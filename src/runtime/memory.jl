# Vulkan memory management for Lava.jl
#
# Device-local buffers with BDA (Buffer Device Address) for kernel arguments.
# Staging buffer for CPU↔GPU transfers.


"""
    VkManagedBuffer

A GPU buffer with a known device address (BDA).
When `mapped_ptr` is non-null, the buffer is in BAR memory (host-visible + device-local)
and can be read/written directly from the CPU without staging copies.
"""
mutable struct PoolBlock
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    base_address::UInt64       # BDA of the start of this block
    capacity::Int              # Total bytes in this block
    bump::Int                  # Next free byte offset (bump pointer for initial carving)
    live_count::Int            # Number of live sub-allocations
end

# Buffer lifecycle states (atomic CAS transitions).  Every VkManagedBuffer
# starts ALIVE.  `unsafe_free!` transitions ALIVE → DEFERRED (queued on a
# bq's deferred_frees list) or ALIVE → DEAD (destroyed immediately).
# `destroy_buffer!` transitions {ALIVE, DEFERRED} → DEAD.  Any call on a
# DEAD buffer is a no-op.  Using an `@atomic` field with CAS guarantees
# idempotent double-free protection across main + finalizer threads.
const BUF_STATE_ALIVE    = UInt8(0)
const BUF_STATE_DEFERRED = UInt8(1)
const BUF_STATE_DEAD     = UInt8(2)

# Lava-only marker bit packed into `extra_usage::UInt32` alongside real
# Vulkan buffer-usage bits.  Vulkan has no usage flag for "this buffer is
# AS build scratch", so we carry our own high bit and strip it before
# `Vulkan.Buffer(...)`.  `bda_alignment_for` keys off this bit to pick the
# cached `ctx.as_scratch_align`.
const LAVA_SCRATCH_BIT = UInt32(1) << 31

mutable struct VkManagedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64     # BDA for PhysicalStorageBuffer access
    mapped_ptr::Ptr{UInt8}  # Non-null for unified/BAR memory
    size::Int
    pool_offset::Int    # Byte offset within pool block (0 for non-pooled)
    pool_block::Union{Nothing, PoolBlock}  # Back-reference for pool free (nothing = non-pooled)
    # Cross-queue synchronization: records which BatchQueue last wrote to
    # this buffer, at which timeline value. Consumed by sync_access! to
    # auto-insert semaphore waits when a dispatch on a different queue
    # takes this buffer as an argument. Nothing = never written.
    # Typed as Any so BatchQueue (defined later) doesn't force a cyclic include.
    # @atomic so the finalizer thread (vk_free!) and main thread (record /
    # sync_access!) can read/write it safely.
    @atomic last_write::Union{Nothing, Tuple{Any, UInt64}}
    # Lifecycle state — see BUF_STATE_* constants above.  @atomic CAS is the
    # single point where double-free / use-after-free is ruled out.
    @atomic state::UInt8
    # Owning VkContext — so upload!/download!/vk_free! don't need the global.
    # Loose type because VkContext is declared in device.jl, included first.
    ctx::Any
end

# Strong references to keep Vulkan handles alive until explicit free
const LIVE_BUFFERS = Set{VkManagedBuffer}()

# Poison value for freed buffer addresses — enables use-after-free detection
# on the GPU side (a shader dereffing a freed BDA traps with this value).
# The CPU-side use-after-free gate is `buf.state`.
const BDA_POISON = 0xDEAD_DEAD_DEAD_DEAD

# Register cleanup callback for vk_reset_device!.  Indirect slabs are per-BQ now
# and die with the old ctx, so only the global memory stats need resetting.
push!(RESET_CALLBACKS, function()
    empty!(LIVE_BUFFERS)
    GPU_LIVE_BYTES[] = 0
    GPU_BYTES_SINCE_LAST_GC[] = 0
    # Per-BQ staging, indirect, and arg slabs die with the old ctx.
end)

# ── GPU memory pressure tracking ──
# Julia's GC doesn't know about GPU memory. LavaArray wrappers are ~50 bytes on
# the CPU heap, but back 100+ MB of VRAM each. Without pressure signals, GC
# never fires and dead GPU buffers accumulate until OOM.
# Solution: track live GPU bytes and trigger GC.gc(false) proactively,
# matching AMDGPU.jl's maybe_collect() pattern.
const GPU_LIVE_BYTES = Threads.Atomic{Int}(0)
const GPU_BYTES_SINCE_LAST_GC = Threads.Atomic{Int}(0)
const GPU_LAST_INCR_GC_TIME = Ref(0.0)
const GPU_LAST_FULL_GC_TIME = Ref(0.0)

"""
    maybe_collect()

Trigger GC if GPU allocation pressure is high. Julia's GC doesn't know about
VRAM — LavaArray wrappers are ~50 bytes on the CPU heap but back hundreds of
MB of GPU memory. Without this, dead GPU buffers accumulate until OOM.

Tracks bytes allocated since last GC (not total live bytes), so steady-state
rendering with stable GPU memory doesn't trigger unnecessary GC cycles.

Two tiers (with separate timers so incremental GC doesn't starve full GC):
- >256 MiB new allocs: incremental GC (rate-limited to every 100ms)
- >512 MiB new allocs: full GC (rate-limited to every 2s)

Called from `vk_alloc` before each allocation.
"""
function maybe_collect()
    since_gc = GPU_BYTES_SINCE_LAST_GC[]
    since_gc < 256 * 1024 * 1024 && return  # <256 MiB new allocs: no pressure
    t = time()
    # Full GC has its own timer — incremental GC must not starve it.
    # atomic_xchg! is the ONLY safe reset: a plain `[]=0` would lose any
    # atomic_add! that races with our read → the bytes-since counter would
    # drift negative (effectively) and future GCs would never fire.
    if since_gc > 512 * 1024 * 1024 && (t - GPU_LAST_FULL_GC_TIME[]) > 2.0
        GPU_LAST_FULL_GC_TIME[] = t
        GPU_LAST_INCR_GC_TIME[] = t
        Threads.atomic_xchg!(GPU_BYTES_SINCE_LAST_GC, 0)
        GC.gc(true)
    elseif (t - GPU_LAST_INCR_GC_TIME[]) > 0.1
        GPU_LAST_INCR_GC_TIME[] = t
        Threads.atomic_xchg!(GPU_BYTES_SINCE_LAST_GC, 0)
        GC.gc(false)
    end
    return
end

"""
    vk_alloc(bq::BatchQueue, nbytes; extra_usage=UInt32(0), unified=false) -> VkManagedBuffer

Allocate a GPU buffer with BDA support.  Takes the queue the allocation is
recorded against — `sweep_retired_batches!` and `drain_deferred_frees!` run
only on that queue's timeline, ready for the multi-queue refactor.

* `extra_usage` — additional `VkBufferUsageFlags` bits (e.g. INDIRECT_BUFFER,
  INDEX_BUFFER, AS_BUILD_INPUT).
* `unified=true` — pick BAR memory (device-local + host-visible + coherent,
  falling back to host-visible only) and map it.  The returned
  `VkManagedBuffer.mapped_ptr` is non-null.  Use for arg/indirect slabs or
  any buffer you want to write from the CPU without staging.
"""
function vk_alloc(bq::BatchQueue, nbytes::Integer;
                  extra_usage::UInt32=UInt32(0), unified::Bool=false)
    if TRACK_ALLOCS[]
        _record_alloc_site!(Int(nbytes))
    end
    # Phase 7 P2: reclaim retired in-flight batches on THIS queue before
    # allocating.  Multi-queue: each caller only drains its own queue's
    # timeline — no implicit reach for `ctx.default_bq` here.
    sweep_retired_batches!(bq)
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    maybe_collect()
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    if result !== nothing
        return result
    end
    GC.gc(true)
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    if result !== nothing
        @info "Lava: GPU allocation succeeded after GC retry" bytes=nbytes
        return result
    end
    live_mb = GPU_LIVE_BYTES[] ÷ (1024 * 1024)
    req_mb = nbytes ÷ (1024 * 1024)
    throw(LavaError("memory allocation",
        "Out of GPU memory. Requested $(req_mb) MiB ($(nbytes) bytes), currently $(live_mb) MiB live in $(length(LIVE_BUFFERS)) buffers.",
        "Free unused LavaArrays, reduce problem size, or check for memory leaks with Lava.gpu_memory_usage()."))
end

"""Attempt GPU buffer allocation, returning nothing on OOM."""
function try_vk_alloc(bq::BatchQueue, nbytes::Integer;
                      extra_usage::UInt32=UInt32(0), unified::Bool=false)
    ctx = bq.ctx::VkContext
    dev = ctx.device
    nbytes = max(nbytes, 16)

    # Strip Lava-only marker bits before passing usage to Vulkan.
    vk_extra = extra_usage & ~LAVA_SCRATCH_BIT

    usage = Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
            Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT
    if vk_extra != UInt32(0)
        usage |= Vulkan.BufferUsageFlag(vk_extra)
    end

    local buf, memory
    mapped_ptr = Ptr{UInt8}(0)
    try
        buf = Vulkan.Buffer(dev, nbytes, usage, Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[])
        mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)

        if unified
            preferred = Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                        Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
            mem_type_idx = find_memory_type_optional(ctx, mem_reqs.memory_type_bits, preferred)
            if mem_type_idx === nothing
                mem_type_idx = find_memory_type(ctx, mem_reqs.memory_type_bits,
                    Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT)
            end
        else
            mem_type_idx = find_memory_type(ctx, mem_reqs.memory_type_bits,
                Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        end

        alloc_flags = Vulkan.MemoryAllocateFlagsInfo(UInt32(0);
            flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
        memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
        unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

        if unified
            mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, nbytes)))
        end
    catch e
        if e isa Vulkan.VulkanError || (e isa LavaError && occursin("memory", e.operation))
            empty!(VALIDATION_MESSAGES)
            return nothing
        end
        rethrow()
    end

    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)

    result = VkManagedBuffer(buf, memory, address, mapped_ptr, Int(nbytes), 0, nothing, nothing, BUF_STATE_ALIVE, ctx)
    push!(LIVE_BUFFERS, result)
    Threads.atomic_add!(GPU_LIVE_BYTES, nbytes)
    Threads.atomic_add!(GPU_BYTES_SINCE_LAST_GC, nbytes)
    return result
end

"""
    vk_free!(buf::VkManagedBuffer)

Free a managed buffer's Vulkan resources.

If a command batch is currently recording or in-flight, the destruction is
deferred until after all batches complete. This prevents DEVICE_LOST from GC
finalizers freeing GPU memory that the in-flight command buffer still
references via BDA addresses.
"""
function vk_free!(buf::VkManagedBuffer)
    # Atomic CAS ALIVE → DEFERRED (optimistic — we haven't yet decided we'll
    # defer; we just need to claim the buffer so no other thread races us).
    # If someone else already transitioned this buffer out of ALIVE, we bail
    # out silently — the work has already been done (or is being done).
    _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEFERRED
    ok || return  # already DEFERRED or DEAD — nothing to do

    delete!(LIVE_BUFFERS, buf)

    # Defer destruction if the GPU still has work in flight that references
    # this buffer. The per-queue timeline semaphore tells us precisely:
    # buf.last_write = (bq, val) means the last dispatch recorded against
    # this buffer will signal `bq.timeline_sem` to `val` when done. If the
    # current counter is already >= val, the GPU is done and free is safe.
    # Otherwise the buffer stays in DEFERRED state on that BQ's deferred-free
    # list; it will be destroyed next time the BQ sweeps completed batches.
    lw = @atomic :acquire buf.last_write
    if lw !== nothing
        bq = lw[1]::BatchQueue
        val = lw[2]::UInt64
        if !device_lost(bq.ctx::VkContext)
            # query_timeline rethrows on healthy-device failure.  We are
            # inside a finalizer-reachable path: a throw here is logged by
            # Julia's finalizer machinery rather than propagating.
            current = query_timeline(bq)
            if current < val
                # Finalizer-thread push into the deferred list — SpinLock so
                # the main thread's drain doesn't race.
                lock(bq.deferred_frees_lock) do
                    push!(bq.deferred_frees, buf)
                end
                return     # state stays DEFERRED; drain_deferred_frees! will transition to DEAD
            end
        end
    end

    destroy_buffer!(buf)
end

"""Actually destroy a buffer's Vulkan resources. Called from `vk_free!` or
`drain_deferred_frees!`.  Atomic CAS ({ALIVE, DEFERRED} → DEAD) ensures the
Vulkan destructor fires exactly once even under racing callers."""
function destroy_buffer!(buf::VkManagedBuffer)
    # Transition {ALIVE, DEFERRED} → DEAD.  First try DEFERRED (the common
    # path — vk_free! or drain always goes through DEFERRED now); fall back
    # to ALIVE for the rare direct-destroy path (e.g. dropping a buffer we
    # never pushed into deferred_frees).
    _, ok = @atomicreplace buf.state BUF_STATE_DEFERRED => BUF_STATE_DEAD
    if !ok
        _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEAD
    end
    ok || return  # was already DEAD; idempotent

    # Pooled chunk: return to pool, don't destroy the shared VkBuffer
    if buf.pool_block !== nothing
        return_to_pool!(buf)
        return
    end

    # Check if the Vulkan device/context is still valid.
    # During Julia shutdown, finalizers fire after the device may be destroyed.
    # Finalizers cannot do context switches, so we only check simple flags here.
    ctx = buf.ctx::VkContext
    if device_lost(ctx)
        # Device is gone — just poison the handle, don't call Vulkan APIs
        buf.mapped_ptr = Ptr{UInt8}(0)
        buf.address = BDA_POISON
        Threads.atomic_sub!(GPU_LIVE_BYTES, buf.size)
        buf.size = 0
        return
    end
    if buf.mapped_ptr != Ptr{UInt8}(0)
        try
            Vulkan.unmap_memory(ctx.device, buf.memory)
        catch
            # unmap may fail if the driver released the memory first — log it
            # loudly (finalizer-safe via jl_safe_printf) so we can notice driver
            # misbehaviour, but don't propagate (finalizers must not throw).
            safe_fin_log("Lava destroy_buffer!: unmap_memory failed\n")
        end
        buf.mapped_ptr = Ptr{UInt8}(0)
    end
    try
        buf.buffer.destructor()
        buf.memory.destructor()
    catch
        safe_fin_log("Lava destroy_buffer!: Vulkan destructor failed\n")
    end
    buf.address = BDA_POISON
    Threads.atomic_sub!(GPU_LIVE_BYTES, buf.size)
    buf.size = 0
end

"""
    drain_deferred_frees!(bq::BatchQueue)

Destroy any buffers in `bq.deferred_frees` whose `last_write` timeline value
has been reached. Safe to call at any time; it only destroys buffers the
GPU is definitely done with. Called at natural sync points: after a flush
completes, and from `sweep_retired!`.
"""
function drain_deferred_frees!(bq::BatchQueue)
    isempty(bq.deferred_frees) && return
    ctx = bq.ctx::VkContext
    # Device-lost shortcut: empty under the lock so no finalizer-thread push
    # survives past our empty!().
    if device_lost(ctx)
        lock(bq.deferred_frees_lock) do
            empty!(bq.deferred_frees)
        end
        return
    end
    current = query_timeline(bq)
    # Hold the lock for the full sweep.  Finalizer-thread pushes are rare
    # (GC pauses only) and the drain body is short; SpinLock contention is
    # negligible.  destroy_buffer! may run Vulkan destructors inside the
    # critical section — that's fine under SpinLock (no yield points).
    lock(bq.deferred_frees_lock) do
        i = 1
        while i <= length(bq.deferred_frees)
            buf = bq.deferred_frees[i]::VkManagedBuffer
            lw = @atomic :acquire buf.last_write
            if lw === nothing || (lw[1]::BatchQueue === bq && lw[2]::UInt64 <= current)
                destroy_buffer!(buf)
                deleteat!(bq.deferred_frees, i)
            else
                i += 1
            end
        end
    end
    return nothing
end

"""Process deferred buffer frees after GPU is idle. Called from vk_flush!()."""
# ── Memory Pool: sub-allocate from large VkBuffer blocks ──
# Eliminates per-array VkBuffer create/destroy overhead (~30μs each).
# All sub-allocations share the parent block's VkBuffer handle.
# Free = return to free list (zero Vulkan API calls).

const POOL_BLOCK_SIZE = 64 * 1024 * 1024  # 64 MiB per block
const POOL_LARGE_THRESHOLD = POOL_BLOCK_SIZE  # Allocs above this bypass the pool
const POOL_MIN_SIZE = 16  # Minimum allocation size (Vulkan requires non-zero)
const POOL_NUM_SIZE_CLASSES = 24  # 2^4=16 to 2^27=128MiB

# Free lists: index i holds reusable VkManagedBuffer objects of size 2^(i+3) bytes
const POOL_BLOCKS = PoolBlock[]
const POOL_FREE_LISTS = [VkManagedBuffer[] for _ in 1:POOL_NUM_SIZE_CLASSES]

push!(RESET_CALLBACKS, function()
    # Destroy all pool blocks on device reset.  The reset callback runs after
    # the device has been marked lost or torn down, so destructor failures are
    # expected (driver may have released the handles already).  Log via
    # jl_safe_printf — we never want to silently lose a destructor error.
    for block in POOL_BLOCKS
        try
            block.buffer.destructor()
            block.memory.destructor()
        catch
            safe_fin_log("Lava POOL_BLOCKS reset: destructor failed (ok during vk_reset_device!)\n")
        end
    end
    empty!(POOL_BLOCKS)
    for fl in POOL_FREE_LISTS
        empty!(fl)
    end
end)

"""Size class index for a given byte size. Returns 1 for 16B, 2 for 32B, etc."""
@inline function size_class_idx(nbytes::Int)
    nbytes = max(nbytes, POOL_MIN_SIZE)
    # Round up to next power of 2
    rounded = nextpow(2, nbytes)
    return trailing_zeros(rounded) - 3  # 16=2^4 → idx 1, 32=2^5 → idx 2, etc.
end

"""Rounded-up allocation size for a given byte count."""
@inline size_class_bytes(nbytes::Int) = nextpow(2, max(nbytes, POOL_MIN_SIZE))

"""Allocate a new pool block (one large VkBuffer)."""
function alloc_pool_block(bq::BatchQueue)
    buf_result = try_vk_alloc(bq, POOL_BLOCK_SIZE)
    if buf_result === nothing
        GC.gc(true)
        drain_deferred_frees!(bq)
        drain_deferred_as_frees!(bq)
        buf_result = try_vk_alloc(bq, POOL_BLOCK_SIZE)
        buf_result === nothing && error("Lava: cannot allocate pool block ($(POOL_BLOCK_SIZE ÷ 1024÷1024) MiB)")
    end
    # Extract Vulkan handles from the VkManagedBuffer, then remove it from LIVE_BUFFERS
    # (the pool block manages its own lifetime, not the per-chunk tracking)
    block = PoolBlock(buf_result.buffer, buf_result.memory, buf_result.address,
                      POOL_BLOCK_SIZE, 0, 0)
    delete!(LIVE_BUFFERS, buf_result)
    # Don't subtract from GPU_LIVE_BYTES — the block IS live memory.
    # Individual chunks don't add to GPU_LIVE_BYTES since the block already accounts for it.
    push!(POOL_BLOCKS, block)
    return block
end

# Diagnostic: track allocation call sites during recording.
# Set TRACK_ALLOCS[] = true to record stack traces of every allocation while the
# active batch is recording. Used to find per-frame allocations leaking into the
# render loop. Reads are merged into ALLOC_TRACE; query via dump_alloc_trace().
const TRACK_ALLOCS = Ref(false)
const ALLOC_TRACE = Dict{Symbol, Int}()
const ALLOC_TRACE_LOCK = ReentrantLock()

function _record_alloc_site!(nbytes::Int)
    bt = stacktrace(backtrace())
    # Capture full stack as a single key (truncated to first 6 user frames)
    user_frames = String[]
    for f in bt
        s = string(f.file)
        # Skip Lava infra and Base
        if occursin("memory.jl", s) || occursin("lavaarray.jl", s) ||
           occursin("launch.jl", s) || occursin("ka_backend.jl", s) ||
           occursin("KernelAbstractions", s) || occursin("Base.jl", s) ||
           occursin("/Base/", s) || occursin("Adapt/src", s) ||
           occursin("gpuarrays", s) || occursin("dict.jl", s) ||
           occursin("./", s) && length(s) < 30
            continue
        end
        push!(user_frames, "$(basename(s)):$(f.line) ($(f.func))")
        length(user_frames) >= 4 && break
    end
    site = Symbol(isempty(user_frames) ? "unknown" : join(user_frames, " <- "))
    lock(ALLOC_TRACE_LOCK) do
        ALLOC_TRACE[site] = get(ALLOC_TRACE, site, 0) + 1
    end
end

function dump_alloc_trace()
    lock(ALLOC_TRACE_LOCK) do
        sorted = sort(collect(ALLOC_TRACE), by=x->x[2], rev=true)
        for (site, count) in sorted
            println("  $count × $site")
        end
    end
end

function clear_alloc_trace!()
    lock(ALLOC_TRACE_LOCK) do
        empty!(ALLOC_TRACE)
    end
end

"""
    pool_alloc(bq::BatchQueue, nbytes; extra_usage=UInt32(0)) -> VkManagedBuffer

Allocate GPU memory on `bq`.  Small device-local allocations come from the
sub-allocation pool (zero Vulkan API calls); large allocations or those
with non-default usage flags bypass the pool via `vk_alloc`.
"""
function pool_alloc(bq::BatchQueue, nbytes::Integer; extra_usage::UInt32=UInt32(0))
    ctx = bq.ctx::VkContext
    nbytes = max(Int(nbytes), POOL_MIN_SIZE)
    if TRACK_ALLOCS[]
        _record_alloc_site!(nbytes)
    end
    sweep_retired_batches!(bq)

    if nbytes > POOL_LARGE_THRESHOLD || extra_usage != UInt32(0)
        return vk_alloc(bq, nbytes; extra_usage)
    end

    alloc_size = size_class_bytes(nbytes)
    idx = size_class_idx(nbytes)
    idx = clamp(idx, 1, POOL_NUM_SIZE_CLASSES)

    # Try the free list first. If empty, run GC (which drains LavaArray
    # finalizers → `return_to_pool!`) and re-check, so we reuse whatever
    # just got freed before burning a new 64-MiB pool block. Without this
    # retry the pool grows monotonically on heavy sim workloads — GC fires
    # sporadically, finalizers enqueue async, and each allocation that
    # races past GC cuts a new block even when thousands of matching
    # buffers are about to return to the free list.
    @inline function try_reuse_or_bump()
        fl = POOL_FREE_LISTS[idx]
        if !isempty(fl)
            buf = pop!(fl)
            block = buf.pool_block::PoolBlock
            block.live_count += 1
            buf.address = block.base_address + UInt64(buf.pool_offset)
            buf.size = alloc_size
            buf.ctx = ctx
            @atomic :release buf.state = BUF_STATE_ALIVE
            return buf
        end
        for block in POOL_BLOCKS
            if block.bump + alloc_size <= block.capacity
                byte_offset = block.bump
                block.bump += alloc_size
                block.live_count += 1
                return VkManagedBuffer(
                    block.buffer, block.memory,
                    block.base_address + UInt64(byte_offset),
                    Ptr{UInt8}(0),
                    alloc_size,
                    byte_offset, block, nothing, BUF_STATE_ALIVE, ctx)
            end
        end
        return nothing
    end

    buf = try_reuse_or_bump()
    buf === nothing || return buf
    # Free list empty + every block full → we are about to cut a new 64-MiB
    # block. Before we do, force a full GC so pending LavaArray finalizers
    # run and return their backing chunks to the free list. `maybe_collect`
    # is rate-limited and may skip, so hit it with `GC.gc(false)` directly —
    # the cost of an incremental GC is much cheaper than a 64 MiB Vulkan
    # alloc we didn't need.
    GC.gc(false)
    buf = try_reuse_or_bump()
    buf === nothing || return buf
    GC.gc(true)  # still nothing? run a full GC to drain any deferred finalizers.
    buf = try_reuse_or_bump()
    buf === nothing || return buf

    block = alloc_pool_block(bq)
    byte_offset = block.bump
    block.bump += alloc_size
    block.live_count += 1
    return VkManagedBuffer(
        block.buffer, block.memory,
        block.base_address + UInt64(byte_offset),
        Ptr{UInt8}(0),
        alloc_size,
        byte_offset, block, nothing, BUF_STATE_ALIVE, ctx)
end

"""Return a pooled chunk to the free list. Keeps VkManagedBuffer object for reuse."""
function return_to_pool!(buf::VkManagedBuffer)
    block = buf.pool_block
    block === nothing && return
    alloc_size = buf.size
    alloc_size == 0 && return
    idx = size_class_idx(alloc_size)
    idx = clamp(idx, 1, POOL_NUM_SIZE_CLASSES)
    block.live_count -= 1
    # Poison address to detect use-after-free, but keep pool_block + pool_offset
    # so pool_alloc can restore the address from block.base_address + pool_offset
    buf.address = BDA_POISON
    buf.mapped_ptr = Ptr{UInt8}(0)
    buf.size = 0
    @atomic :release buf.last_write = nothing  # clear scheduling state so a re-use starts fresh
    # Keep buf.pool_block and buf.pool_offset intact for reuse
    push!(POOL_FREE_LISTS[idx], buf)
end

# ── Staging buffer for CPU↔GPU transfers ──

"""
    get_staging(bq::BatchQueue, nbytes::Integer)
        -> (buf::Vulkan.Buffer, memory::Vulkan.DeviceMemory, mapped_ptr::Ptr, size::Int)

Return `bq`'s staging buffer, growing it to at least `nbytes` if needed.
Backed by a `VkManagedBuffer` whose lifetime follows the normal
timeline-gated free path when re-allocated.
"""
function get_staging(bq::BatchQueue, nbytes::Integer)
    existing = bq.staging
    if existing !== nothing && (existing::VkManagedBuffer).size >= nbytes
        buf = existing::VkManagedBuffer
        return (buf.buffer, buf.memory, buf.mapped_ptr, buf.size)
    end

    # Release the old staging buffer through vk_free! — that routes through
    # the per-BQ deferred queue so in-flight transfers complete first.
    if existing !== nothing
        vk_free!(existing::VkManagedBuffer)
        bq.staging = nothing
    end

    ctx = bq.ctx::VkContext
    dev = bq.device
    alloc_size = max(65536, nextpow(2, nbytes))
    vkbuf = Vulkan.Buffer(
        dev, alloc_size,
        Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[],
    )
    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, vkbuf)
    mem_type_idx = find_memory_type(
        ctx, mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx)
    unwrap(Vulkan.bind_buffer_memory(dev, vkbuf, memory, 0))
    mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, alloc_size)))

    managed = VkManagedBuffer(vkbuf, memory, UInt64(0),   # no BDA needed for staging
                              mapped_ptr, Int(alloc_size),
                              0, nothing, nothing, BUF_STATE_ALIVE, ctx)
    push!(LIVE_BUFFERS, managed)
    Threads.atomic_add!(GPU_LIVE_BYTES, managed.size)
    Threads.atomic_add!(GPU_BYTES_SINCE_LAST_GC, managed.size)
    bq.staging = managed
    return (vkbuf, memory, mapped_ptr, Int(alloc_size))
end

"""
    copy_buffer!(direction, managed, host_ptr, nbytes; offset=0)

Unified CPU↔GPU buffer copy — single entry point for every host-visible
transfer path.  `direction` is `:upload` (host → GPU) or `:download` (GPU → host).

Fast path (BAR memory): direct memcpy via `managed.mapped_ptr` after
`wait_for_write(managed)` drains any in-flight writer.  No staging, no batch.

Slow path (device-local): allocate staging, record `cmd_copy_buffer!` into
the active batch, flush, and memcpy between staging and host.  For downloads
we route the copy through the queue that last wrote `managed` so the copy
piggy-backs on producer dispatches (one fence wait instead of two).  For
uploads we use the buffer's ctx default queue.

The caller is responsible for keeping `host_ptr`'s backing array alive
(typically via `GC.@preserve`).
"""
function copy_buffer!(direction::Symbol, managed::VkManagedBuffer,
                     host_ptr::Ptr{UInt8}, nbytes::Integer; offset::Integer=0)
    nbytes == 0 && return
    @assert direction === :upload || direction === :download  "direction must be :upload or :download"
    buf_offset = managed.pool_offset + Int(offset)

    # BAR fast-path: host-visible mapped memory — direct memcpy, no staging.
    if managed.mapped_ptr != Ptr{UInt8}(0)
        wait_for_write(managed)
        if direction === :upload
            unsafe_copyto!(managed.mapped_ptr + offset, host_ptr, nbytes)
        else
            unsafe_copyto!(host_ptr, managed.mapped_ptr + offset, nbytes)
        end
        return
    end

    # Device-local: route through staging + batched copy.  Pinning `managed`
    # triggers sync_access!'s cross-queue semaphore wait whenever the buffer
    # was last written on a different BatchQueue.
    bq = if direction === :upload
        (managed.ctx::VkContext).default_bq
    else
        lw = @atomic :acquire managed.last_write
        lw !== nothing ? (lw[1]::BatchQueue) : (managed.ctx::VkContext).default_bq
    end
    staging_buf, _, mapped_ptr, _ = get_staging(bq, nbytes)
    if direction === :upload
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr), host_ptr, nbytes)
        cmd_copy_buffer!(bq, staging_buf, managed, nbytes; dst_off=buf_offset)
        flush!(bq, bq.device)
    else
        cmd_copy_buffer!(bq, managed, staging_buf, nbytes; src_off=buf_offset)
        flush!(bq, bq.device)
        unsafe_copyto!(host_ptr, Ptr{UInt8}(mapped_ptr), nbytes)
    end
    return
end

"""Upload host data to a device-local buffer (BAR direct or via staging)."""
function upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)
    GC.@preserve host_data copy_buffer!(:upload, dst, pointer(host_data), length(host_data); offset)
end

"""Download data from a device-local buffer to a host byte vector."""
function download!(host_data::Vector{UInt8}, src::VkManagedBuffer; offset::Int=0)
    GC.@preserve host_data copy_buffer!(:download, src, pointer(host_data), length(host_data); offset)
end

"""Upload a typed array to a device-local buffer."""
function upload_typed!(dst::VkManagedBuffer, data::AbstractVector{T}; offset::Int=0) where T
    bytes = Vector{UInt8}(reinterpret(UInt8, vec(collect(data))))
    upload!(dst, bytes; offset)
end

"""Download data from a device-local buffer into a typed array."""
function download_typed!(data::AbstractVector{T}, src::VkManagedBuffer; offset::Int=0) where T
    GC.@preserve data copy_buffer!(:download, src, Ptr{UInt8}(pointer(data)),
                                    length(data) * sizeof(T); offset)
end

"""
    bda_alignment_for(ctx::VkContext, extra_usage::UInt32) -> UInt64

Required BDA alignment for a buffer created with `extra_usage`.  Keyed
entirely off the usage bits — callers should never pass an `align` kwarg.
`LAVA_SCRATCH_BIT` (a Lava-only marker, stripped before Vulkan sees it)
selects `minAccelerationStructureScratchOffsetAlignment` (cached on ctx).
Everything else defaults to 1 (Vulkan already guarantees the per-usage
minimum alignment via `vkGetBufferDeviceAddress`).
"""
@inline function bda_alignment_for(ctx::VkContext, extra_usage::UInt32)
    if (extra_usage & LAVA_SCRATCH_BIT) != 0
        return ctx.as_scratch_align
    end
    return UInt64(1)
end

# VkMappedBuffer / VkIndirectBuffer / alloc_indirect_slab / vk_alloc_mapped /
# vk_alloc_unified: deleted.  Every GPU buffer allocation goes through
# `vk_alloc(bq, nbytes; extra_usage, unified)` now.  Slab pools (arg buffer
# + indirect dispatch) live on top of `LavaArray`s declared in
# `array/lavaarray.jl` and allocated via `runtime/launch.jl`.

# Indirect buffer slab size — 256 KB of UInt32 storage, enough for ~1000
# 12-byte indirect dispatches.
const INDIRECT_SLAB_SIZE = 256 * 1024

function find_memory_type_optional(ctx::VkContext, type_bits::UInt32, required_flags)
    mem_props = ctx.memory_properties
    for i in 0:(mem_props.memory_type_count - 1)
        if (type_bits & (UInt32(1) << i)) != 0
            mt = mem_props.memory_types[i + 1]
            if (mt.property_flags & required_flags) == required_flags
                return UInt32(i)
            end
        end
    end
    return nothing
end

function find_memory_type(ctx::VkContext, type_bits::UInt32, required_flags)
    mem_props = ctx.memory_properties

    for i in 0:(mem_props.memory_type_count - 1)
        if (type_bits & (UInt32(1) << i)) != 0
            mt = mem_props.memory_types[i + 1]
            if (mt.property_flags & required_flags) == required_flags
                return UInt32(i)
            end
        end
    end
    throw(LavaError(
        "memory allocation",
        "No suitable memory type found for flags $required_flags",
        "Check GPU memory capabilities"))
end
