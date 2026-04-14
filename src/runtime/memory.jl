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

mutable struct VkManagedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64     # BDA for PhysicalStorageBuffer access
    mapped_ptr::Ptr{UInt8}  # Non-null for unified/BAR memory
    size::Int
    pool_offset::Int    # Byte offset within pool block (0 for non-pooled)
    pool_block::Union{Nothing, PoolBlock}  # Back-reference for pool free (nothing = non-pooled)
    # Cross-queue synchronization: records which BatchQueue last wrote to
    # this buffer, at which timeline value. Consumed by record_buffer_access!
    # to auto-insert semaphore waits when a dispatch on a different queue
    # takes this buffer as an argument. Nothing = never written.
    # Typed as Any so BatchQueue (defined later) doesn't force a cyclic include.
    last_write::Union{Nothing, Tuple{Any, UInt64}}
end

# Strong references to keep Vulkan handles alive until explicit free
const LIVE_BUFFERS = Set{VkManagedBuffer}()

# Poison value for freed buffer addresses — enables use-after-free detection.
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

Called from `vk_alloc` / `vk_alloc_unified` before each allocation.
"""
function maybe_collect()
    since_gc = GPU_BYTES_SINCE_LAST_GC[]
    since_gc < 256 * 1024 * 1024 && return  # <256 MiB new allocs: no pressure
    t = time()
    # Full GC has its own timer — incremental GC must not starve it
    if since_gc > 512 * 1024 * 1024 && (t - GPU_LAST_FULL_GC_TIME[]) > 2.0
        GPU_LAST_FULL_GC_TIME[] = t
        GPU_LAST_INCR_GC_TIME[] = t
        GPU_BYTES_SINCE_LAST_GC[] = 0
        GC.gc(true)
    elseif (t - GPU_LAST_INCR_GC_TIME[]) > 0.1
        GPU_LAST_INCR_GC_TIME[] = t
        GPU_BYTES_SINCE_LAST_GC[] = 0
        GC.gc(false)
    end
    return
end

"""
    vk_alloc(nbytes::Integer; extra_usage=nothing) -> VkManagedBuffer

Allocate a device-local buffer with BDA support.
Optional `extra_usage` adds additional `VkBufferUsageFlags` (e.g. for SBT buffers).
"""
function vk_alloc(dev::Vulkan.Device, nbytes::Integer; extra_usage::UInt32=UInt32(0))
    if TRACK_ALLOCS[]
        _record_alloc_site!(Int(nbytes))
    end
    # Proactively drain per-queue deferred frees to keep VRAM steady.
    ctx = VK_CONTEXT_REF[]
    if ctx !== nothing
        drain_deferred_frees!(ctx.default_bq)
        drain_deferred_as_frees!(ctx.default_bq)
    end
    maybe_collect()
    result = try_vk_alloc(dev, nbytes; extra_usage)
    if result !== nothing
        return result
    end
    # OOM -- aggressive GC then retry once
    GC.gc(true)
    if ctx !== nothing
        drain_deferred_frees!(ctx.default_bq)
        drain_deferred_as_frees!(ctx.default_bq)
    end
    result = try_vk_alloc(dev, nbytes; extra_usage)
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

# Convenience: use global device
vk_alloc(nbytes::Integer; extra_usage::UInt32=UInt32(0)) = vk_alloc(vk_device(), nbytes; extra_usage)

"""Attempt GPU buffer allocation, returning nothing on OOM."""
function try_vk_alloc(dev::Vulkan.Device, nbytes::Integer; extra_usage::UInt32=UInt32(0))
    nbytes = max(nbytes, 16)  # Vulkan requires non-zero size

    usage = Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
            Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT
    if extra_usage != UInt32(0)
        usage |= Vulkan.BufferUsageFlag(extra_usage)
    end

    local buf, memory
    try
        buf = Vulkan.Buffer(
            dev, nbytes,
            usage,
            Vulkan.SHARING_MODE_EXCLUSIVE,
            UInt32[]
        )

        mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
        mem_type_idx = find_memory_type(
            mem_reqs.memory_type_bits,
            Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )

        alloc_flags = Vulkan.MemoryAllocateFlagsInfo(
            UInt32(0);  # device_mask (0 = all devices)
            flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
        )
        memory = Vulkan.DeviceMemory(
            dev, mem_reqs.size, mem_type_idx;
            next=alloc_flags
        )
        unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))
    catch e
        # Catch Vulkan OOM errors — let other errors propagate
        if e isa Vulkan.VulkanError || (e isa LavaError && occursin("memory", e.operation))
            # Drain validation messages from the failed allocation — they are expected
            # and should not leak into the next check_validation_errors!() call.
            empty!(VALIDATION_MESSAGES)
            return nothing
        end
        rethrow()
    end

    # Query BDA
    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)

    result = VkManagedBuffer(buf, memory, address, Ptr{UInt8}(0), Int(nbytes), 0, nothing, nothing)
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
    if buf.address == BDA_POISON
        @warn "Lava: double-free detected on VkManagedBuffer (address already poisoned)" maxlog=1
        return
    end
    buf.size == 0 && return  # Already freed

    delete!(LIVE_BUFFERS, buf)

    # Defer destruction if the GPU still has work in flight that references
    # this buffer. The per-queue timeline semaphore tells us precisely:
    # buf.last_write = (bq, val) means the last dispatch recorded against
    # this buffer will signal `bq.timeline_sem` to `val` when done. If the
    # current counter is already >= val, the GPU is done and free is safe.
    # Otherwise queue the buffer on that BQ's deferred-free list; it will
    # be destroyed next time the BQ sweeps completed batches.
    lw = buf.last_write
    if lw !== nothing
        bq = lw[1]::BatchQueue
        val = lw[2]::UInt64
        ctx = VK_CONTEXT_REF[]
        if ctx !== nothing && !device_lost(ctx)
            current = try
                unwrap(Vulkan.get_semaphore_counter_value(bq.device, bq.timeline_sem))
            catch
                typemax(UInt64)  # on error, assume done
            end
            if current < val
                push!(bq.deferred_frees, buf)
                return
            end
        end
    end

    destroy_buffer!(buf)
end

"""Actually destroy a buffer's Vulkan resources. Called from `vk_free!` or `drain_deferred_frees!`."""
function destroy_buffer!(buf::VkManagedBuffer)
    buf.size == 0 && return  # Already destroyed

    # Pooled chunk: return to pool, don't destroy the shared VkBuffer
    if buf.pool_block !== nothing
        return_to_pool!(buf)
        return
    end

    # Check if the Vulkan device/context is still valid.
    # During Julia shutdown, finalizers fire after the device may be destroyed.
    # Finalizers cannot do context switches, so we only check simple flags here.
    ctx = VK_CONTEXT_REF[]
    device_gone = ctx === nothing || device_lost(ctx)

    if device_gone
        # Device is gone — just poison the handle, don't call Vulkan APIs
        buf.mapped_ptr = Ptr{UInt8}(0)
        buf.address = BDA_POISON
        Threads.atomic_sub!(GPU_LIVE_BYTES, buf.size)
        buf.size = 0
        return
    end

    # Device is valid — safe to call Vulkan cleanup
    # Re-check right before Vulkan calls -- a concurrent finalizer may have
    # triggered device_lost between our check above and now. RADV segfaults on
    # vkUnmapMemory after device lost, so we must avoid it.
    if device_lost(ctx)
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
            # unmap can fail if memory was already freed by driver — ignore
        end
        buf.mapped_ptr = Ptr{UInt8}(0)
    end
    try
        buf.buffer.destructor()
        buf.memory.destructor()
    catch
        # Destruction can fail during shutdown — ignore
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
    ctx = VK_CONTEXT_REF[]
    (ctx === nothing || device_lost(ctx)) && (empty!(bq.deferred_frees); return)
    current = try
        unwrap(Vulkan.get_semaphore_counter_value(bq.device, bq.timeline_sem))
    catch
        typemax(UInt64)
    end
    i = 1
    while i <= length(bq.deferred_frees)
        buf = bq.deferred_frees[i]::VkManagedBuffer
        lw = buf.last_write
        if lw === nothing || (lw[1]::BatchQueue === bq && lw[2]::UInt64 <= current)
            destroy_buffer!(buf)
            deleteat!(bq.deferred_frees, i)
        else
            i += 1
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
    # Destroy all pool blocks on device reset
    for block in POOL_BLOCKS
        try
            block.buffer.destructor()
            block.memory.destructor()
        catch
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
function alloc_pool_block()
    buf_result = try_vk_alloc(vk_device(), POOL_BLOCK_SIZE)
    if buf_result === nothing
        # OOM — try GC + drain deferred + retry
        GC.gc(true)
        ctx = VK_CONTEXT_REF[]
        if ctx !== nothing
            drain_deferred_frees!(ctx.default_bq)
            drain_deferred_as_frees!(ctx.default_bq)
        end
        buf_result = try_vk_alloc(vk_device(), POOL_BLOCK_SIZE)
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
    pool_alloc(nbytes::Integer; extra_usage::UInt32=UInt32(0)) -> VkManagedBuffer

Allocate GPU memory. Small allocations come from the pool (zero Vulkan API calls).
Large allocations or those with non-standard usage flags bypass the pool.
"""
function pool_alloc(nbytes::Integer; extra_usage::UInt32=UInt32(0))
    nbytes = max(Int(nbytes), POOL_MIN_SIZE)
    if TRACK_ALLOCS[]
        _record_alloc_site!(nbytes)
    end

    # Large allocations or non-standard usage bypass the pool
    if nbytes > POOL_LARGE_THRESHOLD || extra_usage != UInt32(0)
        return vk_alloc(nbytes; extra_usage)
    end

    maybe_collect()

    alloc_size = size_class_bytes(nbytes)
    idx = size_class_idx(nbytes)
    idx = clamp(idx, 1, POOL_NUM_SIZE_CLASSES)

    # Try free list first — reuse the cached VkManagedBuffer object (zero Julia alloc)
    fl = POOL_FREE_LISTS[idx]
    if !isempty(fl)
        buf = pop!(fl)
        block = buf.pool_block::PoolBlock
        block.live_count += 1
        # Restore the buf fields (they were poisoned on return to pool)
        buf.address = block.base_address + UInt64(buf.pool_offset)
        buf.size = alloc_size
        return buf
    end

    # Try to carve from an existing block's bump pointer
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
                byte_offset, block, nothing)
        end
    end

    # All blocks full — allocate a new one
    block = alloc_pool_block()
    byte_offset = block.bump
    block.bump += alloc_size
    block.live_count += 1
    return VkManagedBuffer(
        block.buffer, block.memory,
        block.base_address + UInt64(byte_offset),
        Ptr{UInt8}(0),
        alloc_size,
        byte_offset, block, nothing)
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
    buf.last_write = nothing  # clear scheduling state so a re-use starts fresh
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
        mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx)
    unwrap(Vulkan.bind_buffer_memory(dev, vkbuf, memory, 0))
    mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, alloc_size)))

    managed = VkManagedBuffer(vkbuf, memory, UInt64(0),   # no BDA needed for staging
                              mapped_ptr, Int(alloc_size),
                              0, nothing, nothing)
    push!(LIVE_BUFFERS, managed)
    Threads.atomic_add!(GPU_LIVE_BYTES, managed.size)
    Threads.atomic_add!(GPU_BYTES_SINCE_LAST_GC, managed.size)
    bq.staging = managed
    return (vkbuf, memory, mapped_ptr, Int(alloc_size))
end

"""
    upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)

Upload host data to a device-local buffer via staging.
"""
function upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)
    nbytes = length(host_data)
    nbytes == 0 && return
    buf_offset = dst.pool_offset + offset

    # Fast path: if buffer is in BAR memory (mapped), write directly via mapped ptr.
    # Any queue still using this buffer must drain first — wait_for_write blocks
    # the CPU until the last write's timeline value has been signalled.
    if dst.mapped_ptr != Ptr{UInt8}(0)
        wait_for_write(dst)
        unsafe_copyto!(dst.mapped_ptr + offset, pointer(host_data), nbytes)
        return
    end

    staging_buf, _, mapped_ptr, _ = get_staging(vk_context().default_bq, nbytes)
    unsafe_copyto!(Ptr{UInt8}(mapped_ptr), pointer(host_data), nbytes)

    one_shot_copy(staging_buf, 0, dst.buffer, buf_offset, nbytes)
end

"""
    download!(host_data::Vector{UInt8}, src::VkManagedBuffer; offset::Int=0)

Download data from a device-local buffer to host via staging.
"""
function download!(host_data::Vector{UInt8}, src::VkManagedBuffer; offset::Int=0)
    nbytes = length(host_data)
    nbytes == 0 && return
    buf_offset = src.pool_offset + offset

    # Fast path: BAR memory — wait for the last write's timeline, then read directly.
    if src.mapped_ptr != Ptr{UInt8}(0)
        wait_for_write(src)
        unsafe_copyto!(pointer(host_data), src.mapped_ptr + offset, nbytes)
        return
    end

    staging_buf, _, mapped_ptr, _ = get_staging(vk_context().default_bq, nbytes)
    one_shot_copy(src.buffer, buf_offset, staging_buf, 0, nbytes)
    unsafe_copyto!(pointer(host_data), Ptr{UInt8}(mapped_ptr), nbytes)
end

"""
    upload_typed!(dst::VkManagedBuffer, data::AbstractVector{T}; offset::Int=0)

Upload a typed array to a device-local buffer.
"""
function upload_typed!(dst::VkManagedBuffer, data::AbstractVector{T}; offset::Int=0) where T
    bytes = Vector{UInt8}(reinterpret(UInt8, vec(collect(data))))
    upload!(dst, bytes; offset)
end

"""
    download_typed!(data::AbstractVector{T}, src::VkManagedBuffer; offset::Int=0)

Download data from a device-local buffer into a typed array.
"""
function download_typed!(data::AbstractVector{T}, src::VkManagedBuffer; offset::Int=0) where T
    nbytes = length(data) * sizeof(T)
    nbytes == 0 && return
    buf_offset = src.pool_offset + offset

    # Fast path: BAR memory — wait for last write's timeline, then direct read.
    if src.mapped_ptr != Ptr{UInt8}(0)
        wait_for_write(src)
        GC.@preserve data begin
            unsafe_copyto!(Ptr{UInt8}(pointer(data)), src.mapped_ptr + offset, nbytes)
        end
        return
    end

    # Device-local: append copy to active batch if possible (one fence wait instead of two).
    # If the buffer has a last_write on some queue, route the append through that queue
    # (so the copy is after the producing dispatches); else use an independent one_shot_copy.
    staging_buf, _, mapped_ptr, _ = get_staging(vk_context().default_bq, nbytes)
    lw = src.last_write
    if lw !== nothing && has_active_recording(lw[1]::BatchQueue)
        append_copy_and_flush!(lw[1]::BatchQueue, src.buffer, buf_offset, staging_buf, nbytes)
    else
        one_shot_copy(src.buffer, buf_offset, staging_buf, 0, nbytes)
    end
    GC.@preserve data begin
        unsafe_copyto!(Ptr{UInt8}(pointer(data)), Ptr{UInt8}(mapped_ptr), nbytes)
    end
end

# ── Internal helpers ──

function one_shot_copy(src::Vulkan.Buffer, src_offset::Integer,
                        dst::Vulkan.Buffer, dst_offset::Integer,
                        nbytes::Integer)
    device_lost() && throw(LavaError("buffer copy", "Vulkan device lost",
        "Call Lava.vk_reset_device!() to reinitialize, or restart Julia session."))
    ctx = vk_context()
    dev = ctx.device

    # Use the current batch queue's dedicated transfer resources (thread-safe)
    bq = ctx.default_bq

    # Flush pending dispatches before transfer to ensure data is written
    if bq.active_batch !== nothing && bq.active_batch.recording
        flush!(bq, dev)
    end

    cmd = bq.xfer_cmd_buf
    fence = bq.xfer_fence

    unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )))

    region = Vulkan.BufferCopy(UInt64(src_offset), UInt64(dst_offset), UInt64(nbytes))
    Vulkan.cmd_copy_buffer(cmd, src, dst, [region])

    unwrap(Vulkan.end_command_buffer(cmd))

    # Submit to the batch queue's own queue and wait
    submit_info = Vulkan.SubmitInfo([], [], [cmd], [])
    submit_result = Vulkan.queue_submit(bq.queue, [submit_info]; fence=fence)
    if iserror(submit_result)
        mark_device_lost!(VK_CONTEXT_REF[])
        unwrap(submit_result)
    end
    fence_result = Vulkan.wait_for_fences(dev, [fence], true, typemax(UInt64))
    if iserror(fence_result)
        mark_device_lost!(VK_CONTEXT_REF[])
        unwrap(fence_result)
    end
    unwrap(Vulkan.reset_fences(dev, [fence]))
end

"""
    VkMappedBuffer

A host-visible, device-accessible buffer with a permanently mapped pointer.
Used for kernel argument buffers — write directly via memcpy, no staging needed.
"""
mutable struct VkMappedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64     # BDA
    mapped_ptr::Ptr{UInt8}
    size::Int
end

"""
    vk_alloc_mapped(nbytes::Integer) -> VkMappedBuffer

Allocate a host-visible, device-accessible buffer with BDA support.
Permanently mapped — write directly to `mapped_ptr`.
On AMD RDNA, uses resizable BAR memory (device-local + host-visible).
Falls back to host-visible only if BAR is unavailable.
"""
function vk_alloc_mapped(nbytes::Integer)
    dev = vk_device()
    nbytes = max(nbytes, 16)

    buf = Vulkan.Buffer(
        dev, nbytes,
        Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)

    # Try device-local + host-visible (BAR memory, fastest)
    preferred_flags = Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                      Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
    mem_type_idx = find_memory_type_optional(mem_reqs.memory_type_bits, preferred_flags)

    # Fall back to host-visible + coherent only
    if mem_type_idx === nothing
        fallback_flags = Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
        mem_type_idx = find_memory_type(mem_reqs.memory_type_bits, fallback_flags)
    end

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(
        UInt32(0);
        flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    )
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)

    mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, nbytes)))

    result = VkMappedBuffer(buf, memory, address, mapped_ptr, Int(nbytes))
    return result
end

"""
    vk_alloc_unified(nbytes::Integer) -> VkManagedBuffer

Allocate a unified (BAR) buffer as VkManagedBuffer with `mapped_ptr` set.
Used by `KA.allocate(; unified=true)` for host-readable GPU buffers.
"""
function vk_alloc_unified(nbytes::Integer)
    maybe_collect()
    mapped = vk_alloc_mapped(max(nbytes, 16))
    managed = VkManagedBuffer(mapped.buffer, mapped.memory, mapped.address,
                               mapped.mapped_ptr, Int(mapped.size), 0, nothing, nothing)
    push!(LIVE_BUFFERS, managed)
    Threads.atomic_add!(GPU_LIVE_BYTES, managed.size)
    Threads.atomic_add!(GPU_BYTES_SINCE_LAST_GC, managed.size)
    return managed
end

# ── Indirect dispatch buffer pool ──

"""
    VkIndirectBuffer — A sub-allocation within an indirect buffer slab.
    Has INDIRECT_BUFFER_BIT + STORAGE_BUFFER_BIT + SHADER_DEVICE_ADDRESS_BIT.
"""
struct VkIndirectBuffer
    buffer::Vulkan.Buffer  # Parent slab's VkBuffer (for cmd_dispatch_indirect)
    buffer_offset::UInt64  # Byte offset within parent slab
    address::UInt64        # BDA of this sub-allocation
    mapped_ptr::Ptr{UInt8}
    size::Int
end

# Indirect buffer slab: similar to arg buffer slabs but with INDIRECT_BUFFER_BIT.
# Each indirect dispatch needs 12 bytes (3×UInt32), aligned to 256 bytes.
# Slab state lives on each BatchQueue; see BatchQueue.indirect_slabs.
const INDIRECT_SLAB_SIZE = 256 * 1024  # 256KB — holds ~1000 indirect dispatches

function alloc_indirect_slab(min_size::Int=INDIRECT_SLAB_SIZE)
    dev = vk_device()
    nbytes = max(min_size, INDIRECT_SLAB_SIZE)

    buf = Vulkan.Buffer(
        dev, nbytes,
        Vulkan.BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)

    preferred = Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
    mem_type_idx = find_memory_type_optional(mem_reqs.memory_type_bits, preferred)
    if mem_type_idx === nothing
        fallback = Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
        mem_type_idx = find_memory_type(mem_reqs.memory_type_bits, fallback)
    end

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(
        UInt32(0); flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)
    mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, nbytes)))

    return VkMappedBuffer(buf, memory, address, mapped_ptr, Int(nbytes))
end

"""Get or create an indirect dispatch buffer from `bq`'s slab pool."""
function get_indirect_buffer(bq::BatchQueue)
    alloc_size = 256  # 12 bytes needed, aligned to 256

    while length(bq.indirect_slabs) < bq.indirect_slab_idx
        push!(bq.indirect_slabs, alloc_indirect_slab())
    end
    slab = bq.indirect_slabs[bq.indirect_slab_idx]::VkMappedBuffer
    if bq.indirect_slab_offset + alloc_size > slab.size
        bq.indirect_slab_idx += 1
        bq.indirect_slab_offset = 0
        while length(bq.indirect_slabs) < bq.indirect_slab_idx
            push!(bq.indirect_slabs, alloc_indirect_slab())
        end
        slab = bq.indirect_slabs[bq.indirect_slab_idx]::VkMappedBuffer
    end

    offset = bq.indirect_slab_offset
    bq.indirect_slab_offset = offset + alloc_size

    return VkIndirectBuffer(
        slab.buffer,
        UInt64(offset),
        slab.address + UInt64(offset),
        slab.mapped_ptr + offset,
        alloc_size
    )
end

"""Reset `bq`'s indirect buffer slab allocator after its in_flight batches drained."""
function reset_indirect_buffer_pool!(bq::BatchQueue)
    bq.indirect_slab_idx = 1
    bq.indirect_slab_offset = 0
end

function find_memory_type_optional(type_bits::UInt32, required_flags)
    ctx = vk_context()
    mem_props = Vulkan.get_physical_device_memory_properties(ctx.physical_device)
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

function find_memory_type(type_bits::UInt32, required_flags)
    ctx = vk_context()
    mem_props = Vulkan.get_physical_device_memory_properties(ctx.physical_device)

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
