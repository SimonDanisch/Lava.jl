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

# Debug instrumentation for iter6 cross-scene cascade investigation.
# All off by default — opt-in via the *_ENABLED Refs.

# Per-finalizer destruction trace
const FREE_DEBUG_ENABLED = Ref{Bool}(false)
const FREE_DEBUG_LOG = NamedTuple[]

# Per-allocation trace
const ALLOC_DEBUG_ENABLED = Ref{Bool}(false)
const ALLOC_DEBUG_LOG = NamedTuple[]

# When enabled, vk_free! scans every live BatchQueue's arg/indirect slabs for
# any UInt64 == buf.address.  Hits are logged + zeroed so the GPU faults on a
# clean null reference instead of corrupting random memory.  Optionally
# throws a LavaError instead of just logging (DESTROY_FREED_BDAS_THROWS[]).
const FREED_BDA_SCAN_ENABLED = Ref{Bool}(false)
const FREED_BDA_SCAN_LOG = NamedTuple[]
const DESTROY_FREED_BDAS_THROWS = Ref{Bool}(false)

# Pre-submit unknown-BDA scan: scans the active arg slab region for any
# BDA-shaped UInt64 that isn't in LIVE_BUFFERS or any pool block / slab.
# Optionally throws a LavaError instead of just warning.
const PRESUBMIT_SCAN_ENABLED = Ref{Bool}(false)
const PRESUBMIT_SCAN_THROWS = Ref{Bool}(false)

# When true, pack_arg!(::VkManagedBuffer, ...) asserts the buffer's state ==
# ALIVE before packing its BDA.  Catches use-after-free where a stale buffer
# reference makes it through to a kernel arg.
const PACK_ARG_ASSERT_LIVE = Ref{Bool}(false)

# When set to a non-zero target BDA, every submit! scans the active arg slab
# for the target value and logs (submit_idx, offsets) to SLAB_DUMP_LOG.  Used
# to track when a known-stale address enters/leaves the arg slab.
const SLAB_DUMP_TARGET = Ref{UInt64}(UInt64(0))
const SLAB_DUMP_LOG = NamedTuple[]

"""
    scan_arg_slabs_for_bda!(buf) -> Int

Scan every live BatchQueue's `arg_slabs` (and `indirect_slabs`) for any
UInt64 word matching `buf.address`.  For each hit, append a record to
`FREED_BDA_SCAN_LOG` and overwrite the slot with 0 so the GPU faults
cleanly on a null reference instead of touching the freed memory.
Returns the number of hits found.

Defined AFTER VkManagedBuffer + BatchQueue (forward-call from vk_free!).
Cost is ~`(slab_size_bytes / 8)` UInt64 reads per live slab — for 4 MiB
slabs that's ~512 K reads, fast enough for debug.
"""
function scan_arg_slabs_for_bda! end

# Buffer lifecycle states (atomic CAS transitions).  Every VkManagedBuffer
# starts ALIVE.  `unsafe_free!` transitions ALIVE → DEFERRED (queued on a
# bq's deferred_frees list) or ALIVE → DEAD (destroyed immediately).
# `destroy_buffer!` transitions {ALIVE, DEFERRED} → DEAD.  Any call on a
# DEAD buffer is a no-op.  Using an `@atomic` field with CAS guarantees
# idempotent double-free protection across main + finalizer threads.
const BUF_STATE_ALIVE    = UInt8(0)
const BUF_STATE_DEFERRED = UInt8(1)
const BUF_STATE_DEAD     = UInt8(2)


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
    # Number of live CommandBatches that have `pin!`ed an array backed by this
    # buffer.  Incremented at pin time, decremented when the batch releases its
    # pins (`release_pinned_refs!`, i.e. the batch completed or its submit
    # failed).  A buffer with pins > 0 is REACHABLE BY A BATCH THAT CAN STILL
    # SUBMIT, so `vk_free!` must not touch it — not even to mark it DEFERRED,
    # because `sync_access!` asserts the buffer is ALIVE at submit.
    #
    # This is what makes the guarantee structural rather than a timing accident:
    # `last_write` only tells us about work already *submitted*, so a buffer
    # pinned into a still-open batch looks idle to the timeline check.
    @atomic pins::Int
    # A free was requested while pins > 0.  The free is not lost, just owed: the
    # last `unpin_buffer!` performs it.
    @atomic free_requested::Bool
    # Owning VkContext — so upload!/download!/vk_free! don't need the global.
    # Loose type because VkContext is declared in device.jl, included first.
    ctx::Any
end

# Strong references to keep Vulkan handles alive until explicit free
const LIVE_BUFFERS = Set{VkManagedBuffer}()

# Poison value for freed buffer addresses — enables use-after-free detection
# on the GPU side (a shader dereffing a freed BDA traps with this value).
# The CPU-side use-after-free gate is `buf.state`.
#
# Poison = 0: no valid GPU BDA is ever 0, so this is unambiguous.  Pointer
# arithmetic on null (ptr + N*sizeof(T)) keeps the result in the unmapped
# low-VA region, which faults clearly.  Detection is also trivial — scan
# arg buffer slots for `== 0`.
const BDA_POISON = UInt64(0)

# Register cleanup callback for vk_reset_device!.  Indirect slabs are per-BQ now
# and die with the old ctx, so only the global memory stats need resetting.
push!(RESET_CALLBACKS, function()
    empty!(LIVE_BUFFERS)
    GPU_LIVE_BYTES[] = 0
    reset_memory_stats!()
    # Per-BQ staging, indirect, and arg slabs die with the old ctx.
end)

# ── GPU memory pressure tracking (ported from AMDGPU.jl) ──
# Julia's GC doesn't know about GPU memory. LavaArray wrappers are ~50 bytes on
# the CPU heap, but back 100+ MB of VRAM each. Without pressure signals, GC
# never fires and dead GPU buffers accumulate until OOM.
#
# Strategy mirrors `AMDGPU.jl/src/memory.jl::maybe_collect`:
#   * Pressure-based trigger (`live / heap_size`), not raw byte counter, so the
#     same logic works for 8 GiB iGPUs and 24 GiB dGPUs without tuning.
#   * GC-rate budget: collector skips itself when it has already spent more than
#     ~5% of wall time in GC (`last_gc_time / dt`).  Budget doubles on high
#     pressure, on blocking calls, and after a productive collection.
#   * Only `GC.gc(false)` (incremental) is ever called automatically — full GC
#     is too expensive to fire from the alloc hot path; if we're truly OOM the
#     caller's retry loop in `vk_alloc` runs the full GC explicitly.
#   * EWMA on `last_gc_time` so a single slow GC doesn't permanently inhibit
#     future GCs.
#
# `GPU_LIVE_BYTES` is incremented by `try_vk_alloc` (real Vulkan allocations,
# including pool blocks) and decremented by `destroy_buffer!`.  Sub-pool chunks
# don't move this counter — the pool block they live in already accounts for
# the VRAM.
const GPU_LIVE_BYTES = Threads.Atomic{Int}(0)

mutable struct MemoryStats
    # Estimated maximum bytes available to us on the device-local heap.
    # Probed lazily from `ctx.memory_properties` and refreshed every 10s.
    @atomic size::Int
    @atomic last_updated::Float64

    # Last `maybe_collect` run + the rolling cost of that GC.
    @atomic last_time::Float64
    @atomic last_gc_time::Float64
    # Bytes freed by the most recent `maybe_collect`-triggered GC.
    @atomic last_freed::Int
end

MemoryStats() = MemoryStats(0, 0.0, 0.0, 0.0, 0)

const MEMORY_STATS = MemoryStats()

const EAGER_GC = Ref{Bool}(true)

"""
    eager_gc!(flag::Bool)

Enable/disable the pressure-driven `maybe_collect`.  Useful when benchmarking,
to take the allocator's GC hooks out of the measurement.
"""
eager_gc!(flag::Bool) = (EAGER_GC[] = flag)

function reset_memory_stats!()
    @atomic MEMORY_STATS.size = 0
    @atomic MEMORY_STATS.last_updated = 0.0
    @atomic MEMORY_STATS.last_time = 0.0
    @atomic MEMORY_STATS.last_gc_time = 0.0
    @atomic MEMORY_STATS.last_freed = 0
    return
end

"""Sum of device-local heap sizes in bytes for `ctx`'s physical device."""
function probe_device_local_heap(ctx::VkContext)
    mem_props = ctx.memory_properties
    total = 0
    for i in 0:(length(mem_props.memory_heaps) - 1)
        heap = mem_props.memory_heaps[i + 1]
        if (UInt32(heap.flags) & UInt32(Vulkan.MEMORY_HEAP_DEVICE_LOCAL_BIT)) != 0
            total += Int(heap.size)
        end
    end
    return total
end

"""
Per-heap snapshot of (device_local, size, budget, usage) in bytes.  `budget`
and `usage` are the driver's view via VK_EXT_memory_budget; both are 0 when the
extension is unavailable.  Used to attach real driver-side memory pressure to
OOM error messages.
"""
function probe_device_memory_budget(ctx::VkContext)
    mem_props = ctx.memory_properties
    n = Int(length(mem_props.memory_heaps))
    sizes = ntuple(i -> Int(mem_props.memory_heaps[i].size), n)
    flags = ntuple(i -> UInt32(mem_props.memory_heaps[i].flags), n)
    device_local = ntuple(i -> (flags[i] & UInt32(Vulkan.MEMORY_HEAP_DEVICE_LOCAL_BIT)) != 0, n)
    if !ctx.memory_budget_available
        return [(heap=i-1, device_local=device_local[i], size=sizes[i],
                 budget=0, usage=0) for i in 1:n]
    end
    # Vulkan.jl's get_physical_device_memory_properties_2 takes the desired
    # chain types as varargs and allocates+populates them itself.
    props2 = Vulkan.get_physical_device_memory_properties_2(
        ctx.physical_device, Vulkan.PhysicalDeviceMemoryBudgetPropertiesEXT)
    bp = props2.next::Vulkan.PhysicalDeviceMemoryBudgetPropertiesEXT
    return [(heap=i-1, device_local=device_local[i], size=sizes[i],
             budget=Int(bp.heap_budget[i]), usage=Int(bp.heap_usage[i])) for i in 1:n]
end

"""
    maybe_collect(ctx::VkContext; blocking::Bool=false)

Trigger an incremental GC if GPU pressure is high.  Ported from
`AMDGPU.jl/src/memory.jl::maybe_collect`.

Called from `vk_alloc` and `pool_alloc` before allocating.  `blocking=true`
lowers the pressure threshold and inflates the rate budget — use it when the
caller is about to do a heavy synchronous operation anyway.
"""
# Absolute-capacity pool trim.
#
# `maybe_collect`'s pressure gate is a *ratio* against the device heap, which is
# the wrong signal for holding on to dead pool blocks on an iGPU: 3 GB of empty
# blocks is only ~20 % of a large shared heap, so the gate never trips — but that
# 3 GB is system RAM the rest of the machine still needs, and the pool is only
# handed back on an OOM retry. Long multi-scene runs therefore accumulate
# gigabytes of blocks that nothing will ever reclaim.
#
# So trim on absolute dead capacity as well, rate-limited so a render loop can't
# pay for it repeatedly. Blocks only become empty once the GC has run their
# sub-allocations' finalizers, hence the collection before the scan.
const POOL_TRIM_THRESHOLD = Ref{Int}(1024 * 1024 * 1024)   # 1 GiB of pool capacity
const POOL_TRIM_MIN_INTERVAL = Ref{Float64}(5.0)           # seconds
const LAST_POOL_TRIM = Ref{Float64}(0.0)

function maybe_trim_pool!(ctx::VkContext)
    GPU_LIVE_BYTES[] < POOL_TRIM_THRESHOLD[] && return
    now = time()
    now - LAST_POOL_TRIM[] < POOL_TRIM_MIN_INTERVAL[] && return
    LAST_POOL_TRIM[] = now

    GC.gc(false)
    any(b -> b.live_count == 0, POOL_BLOCKS) || return
    bq = ctx.default_bq
    quiesce_before_reclaim!(bq)
    n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
    n_blocks > 0 && @debug "Lava: trimmed empty pool blocks" blocks=n_blocks MiB=(bytes_freed >> 20)
    return
end

function maybe_collect(ctx::VkContext; blocking::Bool=false)
    EAGER_GC[] || return
    stats = MEMORY_STATS
    current_time = time()

    # Runs before the ratio gate below: dead pool capacity has to be returned on
    # its own terms, not only when the heap ratio says we are in trouble.
    maybe_trim_pool!(ctx)

    # Refresh device heap estimate every 10s.  The heap size itself doesn't
    # change, but on iGPUs with shared memory another process could shift what
    # we can actually use; a periodic re-probe keeps us honest if we later
    # adopt VK_EXT_memory_budget for real free-memory tracking.
    if current_time - (@atomic stats.last_updated) > 10.0
        max_size = probe_device_local_heap(ctx)
        @atomic stats.size = max_size
        @atomic stats.last_updated = current_time
    end

    size = (@atomic stats.size)
    size > 0 || return  # haven't probed yet

    live = GPU_LIVE_BYTES[]
    pressure = live / size
    min_pressure = blocking ? 0.5 : 0.75
    pressure < min_pressure && return

    # GC rate budget: skip if we've already burned >5% wall time on GC.
    last_time = @atomic stats.last_time
    last_gc_time = @atomic stats.last_gc_time
    dt = current_time - last_time
    gc_rate = dt > 0 ? last_gc_time / dt : 0.0
    max_gc_rate = 0.05
    (@atomic stats.last_freed) > 0.1 * size && (max_gc_rate *= 2)
    blocking && (max_gc_rate *= 2)
    pressure > 0.9 && (max_gc_rate *= 2)
    pressure > 0.95 && (max_gc_rate *= 2)
    gc_rate > max_gc_rate && return

    @atomic stats.last_time = current_time

    pre_gc_live = live
    gc_time = Base.@elapsed GC.gc(false)
    post_gc_live = GPU_LIVE_BYTES[]

    # The GC just returned sub-allocations to their pool blocks, but a block is
    # only handed back to the driver on an OOM retry.  `GPU_LIVE_BYTES` tracks
    # pool *capacity*, so without this the pressure signal never falls: we keep
    # collecting, relieve nothing, and the first thing to notice the pool is
    # holding gigabytes of dead blocks is an allocation failure — or, on an iGPU
    # sharing system RAM, a driver timeout, because the pressure is on memory
    # the rest of the system also needs.
    #
    # Reclaim here, where a collection is already being paid for.  The scan for
    # empty blocks is cheap; only pay `quiesce_before_reclaim!` (which waits for
    # in-flight batches) when a block would actually be returned.
    if any(b -> b.live_count == 0, POOL_BLOCKS)
        bq = ctx.default_bq
        quiesce_before_reclaim!(bq)
        n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
        n_blocks > 0 && @debug "Lava: reclaimed empty pool blocks after GC" blocks=n_blocks MiB=(bytes_freed >> 20)
        post_gc_live = GPU_LIVE_BYTES[]
    end

    @atomic stats.last_freed = pre_gc_live - post_gc_live
    @atomic stats.last_gc_time = 0.75 * last_gc_time + 0.25 * gc_time
    return
end

"""
Captures the exact Vulkan error that caused an allocation to fail.  Returned
from `try_vk_alloc` instead of swallowing the result code, so callers can
include the real `VkResult` and the failing op in the LavaError they throw.
"""
struct AllocFailure
    code::Vulkan.Result
    op::Symbol          # :Buffer, :DeviceMemory, :bind_buffer_memory, :map_memory
    nbytes::Int
    mem_type_idx::Int   # -1 if the failure happened before memory-type selection
end

function format_oom_error(ctx::VkContext, fail::AllocFailure)
    io = IOBuffer()
    println(io, "Out of GPU memory.")
    req_mb = fail.nbytes ÷ (1024 * 1024)
    live_mb = GPU_LIVE_BYTES[] ÷ (1024 * 1024)
    println(io, "  Vulkan returned $(fail.code) from $(fail.op) for $(fail.nbytes) bytes ($(req_mb) MiB).")
    println(io, "  Lava tracked state: $(live_mb) MiB live across $(length(LIVE_BUFFERS)) buffers.")
    if fail.mem_type_idx >= 0
        mem_props = ctx.memory_properties
        mt = mem_props.memory_types[fail.mem_type_idx + 1]
        heap_idx = Int(mt.heap_index)
        println(io, "  Memory type idx=$(fail.mem_type_idx) → heap idx=$(heap_idx), property_flags=0x$(string(UInt32(mt.property_flags), base=16))")
    end
    snapshot = probe_device_memory_budget(ctx)
    if ctx.memory_budget_available
        println(io, "  Driver heap budgets (VK_EXT_memory_budget):")
        for h in snapshot
            sz_mb = h.size ÷ (1024 * 1024)
            bud_mb = h.budget ÷ (1024 * 1024)
            use_mb = h.usage ÷ (1024 * 1024)
            tag = h.device_local ? "DEVICE_LOCAL" : "HOST"
            println(io, "    heap $(h.heap) [$tag] size=$(sz_mb) MiB budget=$(bud_mb) MiB used=$(use_mb) MiB")
        end
    else
        println(io, "  (VK_EXT_memory_budget unavailable — only static heap sizes known)")
        for h in snapshot
            sz_mb = h.size ÷ (1024 * 1024)
            tag = h.device_local ? "DEVICE_LOCAL" : "HOST"
            println(io, "    heap $(h.heap) [$tag] size=$(sz_mb) MiB")
        end
    end
    return String(take!(io))
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

AS-scratch alignment is the caller's responsibility — see
`bda_alignment_for(ctx, scratch::Bool)`, used in `LavaArray(...; scratch=true)`.
"""
function vk_alloc(bq::BatchQueue, nbytes::Integer;
                  extra_usage::UInt32=UInt32(0), unified::Bool=false)
    # Refuse allocation on a lost device — Vulkan calls would either error or
    # (worse) succeed against a torn-down driver state and produce garbage BDAs.
    device_lost(bq.ctx::VkContext) && throw(LavaError(
        "vk_alloc",
        "Vulkan device is lost — cannot allocate new buffers",
        "Call Lava.vk_reset_device!() to reinitialize, or restart Julia."))
    if TRACK_ALLOCS[]
        _record_alloc_site!(Int(nbytes))
    end
    # Phase 7 P2: reclaim retired in-flight batches on THIS queue before
    # allocating.  Multi-queue: each caller only drains its own queue's
    # timeline — no implicit reach for `ctx.default_bq` here.
    sweep_retired_batches!(bq)
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    maybe_collect(bq.ctx::VkContext)
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    result isa VkManagedBuffer && return result
    quiesce_before_reclaim!(bq)
    # Reclaim any pool blocks that are now fully empty after the GC drained
    # their chunks back to the free list.  Without this the pool ratchets
    # up across renders — see Crown 1400×1000 hw_accel=true repro where
    # render 1 left 3.7 GiB of empty 64 MiB blocks pinning the heap.
    n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
    if n_blocks > 0
        @info "Lava: reclaimed empty pool blocks on OOM retry" blocks=n_blocks MiB=(bytes_freed >> 20)
    end
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    if result isa VkManagedBuffer
        @info "Lava: GPU allocation succeeded after GC retry" bytes=nbytes
        return result
    end
    fail = result::AllocFailure
    throw(LavaError("memory allocation",
        format_oom_error(bq.ctx::VkContext, fail),
        "Free unused LavaArrays, reduce problem size, or check for memory leaks with Lava.gpu_memory_usage()."))
end

"""
    quiesce_before_reclaim!(bq)

Submit and wait for everything on `bq`, then collect and drain deferred frees.

The order is the point. `drain_deferred_frees!` releases a chunk once its
**`last_write`** semaphore has signaled, which says nothing about dispatches
that only *read* it, nor about commands already recorded into the batch still
being built — and a graph evaluator does almost nothing else: every weight and
every activation is read by the next layer without being written. Draining (or
reclaiming a block those chunks belong to) while such a batch is open destroys a
VkBuffer out from under queued work, and the damage surfaces later and
elsewhere: `sync_access!: buffer is not ALIVE` on a subsequent submit, a
batch-signal desync, or a segfault inside the driver's `vkCmdPipelineBarrier`.

Flushing first makes the invariant unconditional — after it there is no
recorded-but-unsubmitted work and everything submitted has completed — and it
only runs once an allocation has already failed, so the stall is free in the
steady state. `RECLAIMING` guards the re-entry through `flush!`'s own
allocations.
"""
const RECLAIMING = Threads.Atomic{Bool}(false)

function quiesce_before_reclaim!(bq::BatchQueue)
    if !RECLAIMING[] && !device_lost(bq.ctx::VkContext)
        RECLAIMING[] = true
        try
            flush!(bq, bq.device)
        finally
            RECLAIMING[] = false
        end
    end
    GC.gc(true)
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    return nothing
end

"""Attempt GPU buffer allocation, returning an `AllocFailure` on OOM."""
function try_vk_alloc(bq::BatchQueue, nbytes::Integer;
                      extra_usage::UInt32=UInt32(0), unified::Bool=false)
    ctx = bq.ctx::VkContext
    dev = ctx.device
    nbytes = max(nbytes, 16)

    usage = Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
            Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT
    if extra_usage != UInt32(0)
        usage |= Vulkan.BufferUsageFlag(extra_usage)
    end

    local buf, memory, mem_type_idx
    mapped_ptr = Ptr{UInt8}(0)
    # Track which Vulkan call is in-flight so the catch site can tag the
    # failure precisely.  `mem_type_idx_local` mirrors `mem_type_idx` but
    # stays defined even if the DeviceMemory call throws before assignment.
    op::Symbol = :Buffer
    mem_type_idx_local::Int = -1
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
        mem_type_idx_local = Int(mem_type_idx)

        alloc_flags = Vulkan.MemoryAllocateFlagsInfo(UInt32(0);
            flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
        op = :DeviceMemory
        memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
        op = :bind_buffer_memory
        unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

        if unified
            op = :map_memory
            mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, nbytes)))
        end
    catch e
        # Only swallow honest OOM — every other VulkanError must propagate so
        # the caller sees the real cause (DEVICE_LOST, INVALID_OPAQUE_CAPTURE_ADDRESS,
        # MEMORY_MAP_FAILED, …).  Otherwise vk_alloc's GC-retry would loop and
        # eventually throw a misleading "Out of GPU memory".
        if e isa Vulkan.VulkanError &&
           (e.code == Vulkan.ERROR_OUT_OF_DEVICE_MEMORY ||
            e.code == Vulkan.ERROR_OUT_OF_HOST_MEMORY)
            empty!(VALIDATION_MESSAGES)
            return AllocFailure(e.code, op, Int(nbytes), mem_type_idx_local)
        end
        # DEVICE_LOST during alloc is a hard fault — mark + propagate so the
        # subsequent dispatcher gate fires cleanly.
        if e isa Vulkan.VulkanError && e.code == Vulkan.ERROR_DEVICE_LOST
            mark_device_lost!(ctx)
        end
        rethrow()
    end

    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)

    result = VkManagedBuffer(buf, memory, address, mapped_ptr, Int(nbytes), 0, nothing, nothing, BUF_STATE_ALIVE, 0, false, ctx)
    push!(LIVE_BUFFERS, result)
    Threads.atomic_add!(GPU_LIVE_BYTES, nbytes)
    if ALLOC_DEBUG_ENABLED[]
        push!(ALLOC_DEBUG_LOG,
              (kind=:direct, addr=address, size=Int(nbytes), pool=false,
               mtype=Int(mem_type_idx), unified=unified, usage=UInt32(usage)))
    end
    return result
end

# ── Pin accounting ──
# A CommandBatch that has `pin!`ed an array holds a claim on the backing buffer
# until the batch can no longer submit (it completed, or its submit failed).
# `vk_free!` honours that claim instead of racing it, which makes "pinned by a
# live batch" and "freed" mutually exclusive by construction rather than by
# timing.
function pin_buffer!(buf::VkManagedBuffer)
    @atomic buf.pins += 1
    return nothing
end

function unpin_buffer!(buf::VkManagedBuffer)
    remaining = @atomic buf.pins -= 1
    @assert remaining >= 0 "unpin_buffer!: pins went negative ($remaining) — pin!/release_pinned_refs! are unbalanced"
    # Last batch let go, and a free was owed while we held it: pay it now, at a
    # point where no batch can reference the buffer any more.
    if remaining == 0
        _, owed = @atomicreplace buf.free_requested true => false
        owed && vk_free!(buf)
    end
    return nothing
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
    # A live batch has this buffer pinned, so it can still be submitted with the
    # buffer's BDA baked into an arg slab.  Do NOT touch `state` here: marking a
    # pinned buffer DEFERRED is exactly what made `sync_access!` assert
    # "buffer is not ALIVE (state=1)" at submit.  Record the debt; the last
    # `unpin_buffer!` pays it.
    #
    # The timeline check further down cannot cover this case: it keys off
    # `last_write`, which `sync_access!` only writes at submit, so a buffer
    # pinned into a still-open batch reads as never-written and looks idle.
    if (@atomic :acquire buf.pins) > 0
        @atomic :release buf.free_requested = true
        # The last unpin may have landed between those two lines and seen
        # `free_requested` still false, in which case nobody owes the free.
        # Claim it back and fall through; otherwise the unpin path owns it.
        if (@atomic :acquire buf.pins) > 0
            return
        end
        _, owed = @atomicreplace buf.free_requested true => false
        owed || return
    end

    # Atomic CAS ALIVE → DEFERRED (optimistic — we haven't yet decided we'll
    # defer; we just need to claim the buffer so no other thread races us).
    # If someone else already transitioned this buffer out of ALIVE, we bail
    # out silently — the work has already been done (or is being done).
    _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEFERRED
    ok || return  # already DEFERRED or DEAD — nothing to do

    delete!(LIVE_BUFFERS, buf)

    if FREE_DEBUG_ENABLED[]
        bqd = buf.ctx.default_bq
        active_dbg = (bqd.active_batch !== nothing) && bqd.active_batch.recording
        push!(FREE_DEBUG_LOG,
              (addr=buf.address, size=buf.size,
               pool=buf.pool_block !== nothing,
               lw=@atomic(:acquire, buf.last_write),
               active=active_dbg))
    end

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
            # Defer destruction if EITHER:
            #   (a) GPU still has in-flight work that references this buffer
            #       (current < val — last submitted dispatch still pending), OR
            #   (b) the BQ has an active recording batch — even if all submits
            #       have completed, the recording batch may have captured this
            #       buffer's BDA via a runtime-pinned reference that pin_leaves!
            #       didn't catch (e.g. a closure capture in a kernel that takes
            #       the buffer indirectly). Destroying mid-recording, then
            #       letting the recording submit, races the freed BDA against
            #       the dispatch and trips a RADV GPUVM PERMISSION_FAULT.
            #       The window from finalizer-pause to next batch retirement is
            #       short, so deferring is cheap; reclaiming the buffer happens
            #       via `drain_deferred_frees!` at the next flush/submit boundary.
            active = (bq.active_batch !== nothing) && bq.active_batch.recording
            if current < val || active
                # Finalizer-thread push into the deferred list — SpinLock so
                # the main thread's drain doesn't race.
                lock(bq.deferred_frees_lock) do
                    push!(bq.deferred_frees, buf)
                end
                return     # state stays DEFERRED; drain_deferred_frees! will transition to DEAD
            end
        end
    end

    # Pre-destroy safety scan: if `buf.address` still appears in any live arg
    # slab, that's a use-after-free waiting to happen.  Log it and zero the
    # slot so the GPU faults on a null reference (BDA_POISON) instead of
    # corrupting whatever memory is mapped at the old address.
    if FREED_BDA_SCAN_ENABLED[]
        hits = scan_arg_slabs_for_bda!(buf)
        if hits > 0 && DESTROY_FREED_BDAS_THROWS[]
            throw(LavaError("vk_free!",
                "destroying buffer at 0x$(string(buf.address, base=16, pad=16)) but its BDA still appears $(hits)× in live arg slabs",
                "an unpinned reference is leaking — see FREED_BDA_SCAN_LOG"))
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

# Scanner method — reachable now that VkManagedBuffer + BatchQueue are defined.
function scan_arg_slabs_for_bda!(buf::VkManagedBuffer)
    target = buf.address
    target == UInt64(0) && return 0
    hits = 0
    ctx = buf.ctx
    bq = ctx.default_bq
    for (kind, slabs) in ((:arg, bq.arg_slabs), (:indirect, bq.indirect_slabs))
        for (i, slab) in enumerate(slabs)
            mb = slab.buf[]::VkManagedBuffer
            mb.mapped_ptr == Ptr{UInt8}(0) && continue
            n = mb.size ÷ 8
            p = Ptr{UInt64}(mb.mapped_ptr)
            for k in 0:(n-1)
                v = unsafe_load(p, k+1)
                if v == target
                    push!(FREED_BDA_SCAN_LOG,
                          (slab=kind, idx=i, offset=k*8, freed_bda=target,
                           buf_size=buf.size))
                    unsafe_store!(p, UInt64(0), k+1)
                    hits += 1
                end
            end
        end
    end
    return hits
end

"""
    scan_slabs_for_unknown_bdas() -> Vector{NamedTuple}

Walk every UInt64 in every live arg/indirect slab.  Report any value that
LOOKS like a BDA (in the upper half of the address space, i.e. high bit
of bit 63 set OR top 16 bits = 0xffff) but is NOT the address of any
buffer in `LIVE_BUFFERS` and is NOT 0.  Useful for catching stale BDAs
that pin_leaves! / pack_args_direct! missed.

Call this RIGHT BEFORE submit to catch problems before they reach the GPU.
"""
function scan_slabs_for_unknown_bdas(bq)
    bq === nothing && return NamedTuple[]
    live = Set{UInt64}()
    for buf in LIVE_BUFFERS
        push!(live, buf.address)
    end
    pool_ranges = Tuple{UInt64,UInt64}[]
    for blk in POOL_BLOCKS
        push!(pool_ranges, (blk.base_address, blk.base_address + UInt64(blk.capacity)))
    end
    # Slabs themselves are valid arenas — the arg slab packs nested structs
    # by writing pointers to within the slab itself (a "byval-inline" arg's
    # outer arg pointer is `slab_base + inline_offset`).  Whitelist any value
    # that lands inside a known slab's address range.
    slab_ranges = Tuple{UInt64,UInt64}[]
    for slabs in (bq.arg_slabs, bq.indirect_slabs)
        for slab in slabs
            mb = slab.buf[]::VkManagedBuffer
            push!(slab_ranges, (mb.address, mb.address + UInt64(mb.size)))
        end
    end
    @inline function in_known_range(addr)
        for (lo, hi) in pool_ranges
            lo <= addr < hi && return true
        end
        for (lo, hi) in slab_ranges
            lo <= addr < hi && return true
        end
        return false
    end
    results = NamedTuple[]
    for (kind, slabs) in ((:arg, bq.arg_slabs), (:indirect, bq.indirect_slabs))
        for (i, slab) in enumerate(slabs)
            mb = slab.buf[]::VkManagedBuffer
            mb.mapped_ptr == Ptr{UInt8}(0) && continue
            n_bytes = if kind == :arg && i == bq.arg_slab_idx
                bq.arg_slab_offset
            else
                Int(mb.size)
            end
            n = n_bytes ÷ 8
            p = Ptr{UInt64}(mb.mapped_ptr)
            for k in 0:(n-1)
                v = unsafe_load(p, k+1)
                # Look only for "0xffff8…" sign-extended BDA-shaped values
                # whose 48-bit form is in the high half (bit 47 set).
                v < UInt64(0xffff800000000000) && continue
                v == typemax(UInt64) && continue  # 0xff..ff often appears in scratch
                v in live && continue
                in_known_range(v) && continue
                push!(results, (slab=kind, idx=i, offset=k*8, val=v))
            end
        end
    end
    return results
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

# Debug-only: force every LavaArray onto its own VkBuffer (one vkGetBufferDeviceAddress
# per array). GPU-AV's BDA OOB validation tracks ranges per VkBuffer, so with the pool
# on it cannot see sub-pool overruns; with this flag on, each LavaArray's bounds are
# checked individually. Slow — leave off in production.
const POOL_DISABLED = Ref{Bool}(false)

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
    if buf_result isa AllocFailure
        quiesce_before_reclaim!(bq)
        # Same fallback as vk_alloc: reclaim any pool blocks the GC just
        # emptied before deciding the OOM is real.
        n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
        if n_blocks > 0
            @info "Lava: reclaimed empty pool blocks on pool-block OOM retry" blocks=n_blocks MiB=(bytes_freed >> 20)
        end
        buf_result = try_vk_alloc(bq, POOL_BLOCK_SIZE)
        if buf_result isa AllocFailure
            throw(LavaError("pool block allocation",
                "Cannot allocate $(POOL_BLOCK_SIZE ÷ 1024 ÷ 1024) MiB pool block.\n" *
                format_oom_error(bq.ctx::VkContext, buf_result),
                "Free unused LavaArrays, reduce problem size, or check for memory leaks with Lava.gpu_memory_usage()."))
        end
    end
    # Extract Vulkan handles from the VkManagedBuffer, then remove it from LIVE_BUFFERS
    # (the pool block manages its own lifetime, not the per-chunk tracking)
    block = PoolBlock(buf_result.buffer, buf_result.memory, buf_result.address,
                      POOL_BLOCK_SIZE, 0, 0)
    delete!(LIVE_BUFFERS, buf_result)
    # Don't subtract from GPU_LIVE_BYTES — the block IS live memory.
    # Individual chunks don't add to GPU_LIVE_BYTES since the block already accounts for it.
    push!(POOL_BLOCKS, block)
    if ALLOC_DEBUG_ENABLED[]
        push!(ALLOC_DEBUG_LOG, (kind=:pool_block, addr=buf_result.address,
                                size=POOL_BLOCK_SIZE, pool=true))
    end
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
    # Even pure free-list reuse must refuse a dead device — the pool blocks
    # belong to the old (broken) ctx and would hand back garbage BDAs.
    device_lost(ctx) && throw(LavaError(
        "pool_alloc",
        "Vulkan device is lost — cannot allocate new buffers",
        "Call Lava.vk_reset_device!() to reinitialize, or restart Julia."))
    nbytes = max(Int(nbytes), POOL_MIN_SIZE)
    if TRACK_ALLOCS[]
        _record_alloc_site!(nbytes)
    end
    sweep_retired_batches!(bq)
    drain_deferred_frees!(bq)
    # Pressure-driven GC, identical hook to `vk_alloc`.  Without this the pool
    # fast path silently grows VRAM until the bump pointers exhaust every block,
    # then forces a `GC.gc` from inside the alloc — exactly the 2-5 ms spike the
    # AK benchmarks regressed on.
    maybe_collect(ctx)

    if POOL_DISABLED[] || nbytes > POOL_LARGE_THRESHOLD || extra_usage != UInt32(0)
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
                    byte_offset, block, nothing, BUF_STATE_ALIVE, 0, false, ctx)
            end
        end
        return nothing
    end

    buf = try_reuse_or_bump()
    buf === nothing || return buf
    # Free list empty + every block full → cut a new 64-MiB block.  We used to
    # force a `GC.gc(false)` + `GC.gc(true)` chain here to drain finalizers
    # before growing the pool, but that synchronous GC inside the alloc hot
    # path showed up as 2-5 ms spikes in tight loops (AK benchmarks).  The
    # proactive `maybe_collect(ctx)` at the top now drives finalizer drains via
    # rate-limited incremental GC; if pressure is still low when we land here
    # the right answer is to just commit another pool block rather than pay a
    # stop-the-world full GC the budget would have skipped anyway.
    block = alloc_pool_block(bq)
    byte_offset = block.bump
    block.bump += alloc_size
    block.live_count += 1
    return VkManagedBuffer(
        block.buffer, block.memory,
        block.base_address + UInt64(byte_offset),
        Ptr{UInt8}(0),
        alloc_size,
        byte_offset, block, nothing, BUF_STATE_ALIVE, 0, false, ctx)
end

"""
    reclaim_empty_pool_blocks!(bq::BatchQueue) -> (n_blocks::Int, bytes::Int)

Free every pool block whose `live_count` is 0: drop the block's chunks from
`POOL_FREE_LISTS`, destroy its `VkBuffer` + `VkDeviceMemory`, and remove it
from `POOL_BLOCKS`.  Returns `(n_blocks_reclaimed, bytes_reclaimed)`.

Called from `vk_alloc` / `alloc_pool_block` only on the OOM retry path, so
steady-state allocations don't pay the scan cost.  Not finalizer-safe —
runs only on the main allocator path.

Callers must have run `quiesce_before_reclaim!` first — see there for why.
"""
function reclaim_empty_pool_blocks!(bq::BatchQueue)
    isempty(POOL_BLOCKS) && return (0, 0)
    ctx = bq.ctx::VkContext
    empty_blocks = Set{PoolBlock}()
    kept = PoolBlock[]
    bytes_reclaimed = 0
    for block in POOL_BLOCKS
        if block.live_count == 0
            push!(empty_blocks, block)
            bytes_reclaimed += block.capacity
        else
            push!(kept, block)
        end
    end
    isempty(empty_blocks) && return (0, 0)
    # Drop free-list chunks that belong to any reclaimed block.  Each chunk
    # is a VkManagedBuffer whose `pool_block` field identifies its host.
    for fl in POOL_FREE_LISTS
        filter!(buf -> begin
            pb = buf.pool_block
            pb === nothing || !(pb in empty_blocks)
        end, fl)
    end
    # Swap kept list into POOL_BLOCKS so `pool_alloc` no longer sees the
    # reclaimed blocks (must precede destructor calls — a concurrent bump
    # against a freed handle would corrupt the driver).
    empty!(POOL_BLOCKS)
    append!(POOL_BLOCKS, kept)
    if !device_lost(ctx)
        for block in empty_blocks
            try
                block.buffer.destructor()
                block.memory.destructor()
            catch
                # Match destroy_buffer!: don't propagate from destructors,
                # but log loudly so driver misbehaviour is visible.
                safe_fin_log("Lava reclaim_empty_pool_blocks!: Vulkan destructor failed\n")
            end
        end
    end
    # Pool blocks ARE counted in GPU_LIVE_BYTES (see alloc_pool_block —
    # we intentionally don't subtract per-chunk because the block is the
    # real live memory).  Subtract the reclaimed capacity here.
    Threads.atomic_sub!(GPU_LIVE_BYTES, bytes_reclaimed)
    return (length(empty_blocks), bytes_reclaimed)
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
    # Prefer HOST_CACHED: downloads memcpy FROM this buffer, and CPU reads of
    # write-combined (uncached) host-visible memory run ~70 MB/s vs GB/s cached.
    mem_type_idx = something(
        find_memory_type_optional(ctx, mem_reqs.memory_type_bits,
                                  Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                  Vulkan.MEMORY_PROPERTY_HOST_CACHED_BIT),
        find_memory_type(ctx, mem_reqs.memory_type_bits,
                         Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT))
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx)
    throw_if_error(ctx, "vkBindBufferMemory",
        Vulkan.bind_buffer_memory(dev, vkbuf, memory, 0))
    mapped_ptr = Ptr{UInt8}(throw_if_error(ctx, "vkMapMemory",
        Vulkan.map_memory(dev, memory, 0, alloc_size)))

    managed = VkManagedBuffer(vkbuf, memory, UInt64(0),   # no BDA needed for staging
                              mapped_ptr, Int(alloc_size),
                              0, nothing, nothing, BUF_STATE_ALIVE, 0, false, ctx)
    push!(LIVE_BUFFERS, managed)
    Threads.atomic_add!(GPU_LIVE_BYTES, managed.size)
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
        # wait_for_write reads buf.last_write, which is only populated by
        # sync_access! at submit time — so a dispatch that is recorded but
        # not yet submitted is invisible to it. Flush the active batch of
        # the buffer's ctx first so any pending writer actually lands before
        # we memcpy. (wait_for_write still matters for cross-queue writers
        # and already-in-flight batches.)
        bq = (managed.ctx::VkContext).default_bq
        if bq.active_batch !== nothing
            flush!(bq, bq.device)
        end
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
    bda_alignment_for(ctx::VkContext, scratch::Bool) -> UInt64

Required BDA alignment.  AS-build scratch buffers must be aligned to
`minAccelerationStructureScratchOffsetAlignment` (cached on ctx).
Everything else defaults to 1 (Vulkan already guarantees the per-usage
minimum alignment via `vkGetBufferDeviceAddress`).
"""
@inline function bda_alignment_for(ctx::VkContext, scratch::Bool)
    return scratch ? ctx.as_scratch_align : UInt64(1)
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
    for i in 0:(length(mem_props.memory_types) - 1)
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

    for i in 0:(length(mem_props.memory_types) - 1)
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
