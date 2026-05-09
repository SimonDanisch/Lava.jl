# Command buffer recording and dispatch for Lava.jl
#
# Batch-based command buffer management: each CommandBatch owns its own
# command buffer and strong refs (via `batch.pinned`) to every GPU object
# the dispatch references.  Retirement is synchronous: `sweep_retired_batches!`
# polls each queue's timeline semaphore and recycles batches whose signal
# value has been reached.
#
# GC safety: every dispatch `pin!`s its arg buffers + pipelines + other
# GPU-referenced objects into `batch.pinned`.  Objects remain reachable until
# the timeline semaphore reaches `batch.signal_value`, at which point
# `reclaim_batch!` empties `batch.pinned` and returns the batch to the free
# pool.  Vulkan destructors fired by Julia's GC (via DataRef refcount) hit
# the deferred-free path in `memory.jl`, which itself waits on the same
# timeline before destroying the underlying `VkBuffer`.
#
# Cross-queue sync: at `submit!` time, `sync_access!(batch, buf)` walks every
# pinned object and, for `VkManagedBuffer`s, inserts a timeline-semaphore wait
# on the prior writing queue (if any) and updates `buf.last_write` to this
# batch's (bq, signal_value).

# Flush counter for benchmarking (atomic for thread safety)
const FLUSH_COUNTER = Threads.Atomic{Int}(0)

# Global dispatch counter for debugging (total dispatches across all flushes)
const TOTAL_DISPATCH_COUNTER = Threads.Atomic{Int}(0)

# Dispatch info for debugging DEVICE_LOST
const LAST_DISPATCH_INFO = Ref{String}("")
const PREV_DISPATCH_INFO = Ref{String}("")

# Ring buffer of last N dispatch names for crash debugging
const DISPATCH_LOG = String[]
const MAX_DISPATCH_LOG = 2000

# Toggle dispatch logging (disabled by default for zero-alloc dispatch path).
# Enable with Lava.DISPATCH_LOGGING_ENABLED[] = true for debugging.
# On DEVICE_LOST, the error handler re-enables logging automatically.
const DISPATCH_LOGGING_ENABLED = Ref{Bool}(false)
# TEMP DEBUG: if true, submit! waits for the batch it just submitted to complete
# and records wall-clock GPU time in BATCH_WAIT_TIMES. This SERIALIZES the pipeline
# — only for measurement, not production. Used to identify batches whose actual
# GPU execution time is near the amdgpu TDR threshold (~10 s).
const BATCH_TIMING_ENABLED = Ref{Bool}(false)
const BATCH_WAIT_TIMES = Float64[]  # seconds
const BATCH_WAIT_INFO  = String[]   # last kernel in batch at wait time
const BATCH_WAIT_DISPATCHES = Int[] # dispatch count for each measured batch

function log_dispatch!(info::String)
    DISPATCH_LOGGING_ENABLED[] || return
    if length(DISPATCH_LOG) >= MAX_DISPATCH_LOG
        popfirst!(DISPATCH_LOG)
    end
    push!(DISPATCH_LOG, info)
end

# Register cleanup callback for vk_reset_device!
push!(RESET_CALLBACKS, function()
    FLUSH_COUNTER[] = 0
    TOTAL_DISPATCH_COUNTER[] = 0
    LAST_DISPATCH_INFO[] = ""
    PREV_DISPATCH_INFO[] = ""
    empty!(DISPATCH_LOG)
    DISPATCH_LOGGING_ENABLED[] = false
end)

# Pre-allocated barrier buffer using raw VkMemoryBarrier (isbits).
# Vulkan.jl's MemoryBarrier wrapper allocates ~1.2KB per cmd_pipeline_barrier call
# due to high-level → low-level struct conversion. Using direct ccall with the raw
# VkMemoryBarrier struct is zero-alloc. Saves ~16MB/render for 13k dispatches.
import Vulkan.VkCore: VkMemoryBarrier, VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    VkAccessFlags, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_TRANSFER_READ_BIT,
    VkPipelineStageFlags, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VkDependencyFlags
# Function pointer for vkCmdPipelineBarrier — initialized in _init_vulkan!
const CMD_PIPELINE_BARRIER_FPTR = Ref{Ptr{Nothing}}(C_NULL)

# CB split threshold: seal the current command buffer and start a new one after
# this many dispatches per segment. All segments are submitted in a single
# vkQueueSubmit. This avoids NVIDIA driver crashes from enormous command buffers
# (30k+ dispatches) while maintaining single-submit efficiency.
# Default 3000 ≈ 1 Hikari volpath sample (50 bounces × 60 dispatches/bounce).
# Set to 0 to disable splitting.
const CB_SPLIT_THRESHOLD = Ref{Int}(3000)

# Auto-submit threshold: submit the current batch (starting a new one on the
# next dispatch) when its dispatch count reaches this value. Measured per-batch
# GPU execution time is capped at ~51 ms on dolphin HQ (well under amdgpu's
# ~10 s TDR) so the original TDR hypothesis this was meant to address turned
# out to be wrong. Leaving the mechanism in place but disabled by default;
# enable (e.g. 512) if a future workload actually shows per-batch GPU time
# approaching TDR.
const AUTO_SUBMIT_THRESHOLD = Ref{Int}(0)


# ── Batch lifecycle ──

"""
    ensure_active_batch!(bq::BatchQueue) -> CommandBatch

Get the active batch, allocating one from the free pool if needed.
Begins command buffer recording if not already started.
"""
function ensure_active_batch!(bq::BatchQueue)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread dispatch forbidden"
    # Refuse to start new work on a lost device.  Without this gate, kernel
    # dispatches keep recording into command buffers that will never run,
    # leaking arg slabs/cmd bufs until vk_reset_device! and (worse) hiding
    # the original error behind a flood of follow-on failures.
    device_lost(bq.ctx::VkContext) && throw(LavaError(
        "ensure_active_batch!",
        "Vulkan device is lost — cannot record new dispatches",
        "Call Lava.vk_reset_device!() to reinitialize, or restart Julia. " *
        "All existing LavaArrays are invalid after reset and must be re-allocated."))
    # Sweep any reaper-retired batches first so their resources recycle and
    # their errors surface before we start new work.
    sweep_retired_batches!(bq)

    batch = bq.active_batch
    if batch !== nothing
        if !batch.recording
            throw_if_error(bq, "vkBeginCommandBuffer",
                Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
                    flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                )))
            batch.recording = true
            # Fresh open on reused batch — assign the timeline value it will
            # signal, so record_buffer_access! can write it into buf.last_write.
            batch.signal_value = bq.next_timeline + 1
        end
        return batch
    end

    if !isempty(bq.free_batches)
        batch = pop!(bq.free_batches)
    else
        batch = allocate_batch(bq)
    end

    bq.active_batch = batch

    throw_if_error(bq, "vkBeginCommandBuffer",
        Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
            flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )))
    batch.recording = true
    batch.signal_value = bq.next_timeline + 1
    return batch
end


# VkContext convenience: use default batch queue
function ensure_active_batch!(ctx::VkContext)
    return ensure_active_batch!(ctx.default_bq)
end

"""Allocate a new CommandBatch (new command buffer from the pool)."""
function allocate_batch(bq::BatchQueue)
    cmd_buf = alloc_cmd_buf(bq)
    batch = init_batch(cmd_buf)
    batch.bq = bq
    return batch
end

"""Allocate a command buffer from the free pool, or create a new one."""
function alloc_cmd_buf(bq::BatchQueue)
    if !isempty(bq.free_cmd_bufs)
        return pop!(bq.free_cmd_bufs)
    end
    alloc_info = Vulkan.CommandBufferAllocateInfo(
        bq.cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1
    )
    return throw_if_error(bq, "vkAllocateCommandBuffers",
        Vulkan.allocate_command_buffers(bq.device, alloc_info))[1]
end

"""Reclaim a completed batch: clear pinned set, return to free pool."""
function reclaim_batch!(bq::BatchQueue, batch::CommandBatch)
    batch.recording = false
    batch.dispatch_count = 0
    batch.segment_dispatches = 0
    batch.last_was_rt = false
    empty!(batch.pinned)
    empty!(batch.wait_semaphores)
    empty!(batch.dispatch_log)
    append!(bq.free_cmd_bufs, batch.sealed_cmd_bufs)
    empty!(batch.sealed_cmd_bufs)
    push!(bq.free_batches, batch)
    # Note: pool reset + deferred-free drain are done in `sweep_retired_batches!`
    # AFTER the batch is actually removed from `bq.in_flight` (reclaim_batch! is
    # called with the batch still present in `in_flight`, so `isempty` would
    # always read `false` here and the reset would never fire — that bug leaked
    # ~70 MiB/frame via arg/indirect slabs on long per-frame dispatch streams).
end


"""
    maybe_split_cb!(batch, ctx)

If the current CB segment has reached `CB_SPLIT_THRESHOLD` dispatches, seal it
and start a fresh CB. The sealed CB is stored in `batch.sealed_cmd_bufs` and will
be submitted alongside the active CB in `vk_flush!`.

Barriers work across CB boundaries per Vulkan spec — submission order defines
the scope of pipeline barriers, not command buffer boundaries.
"""
function maybe_split_cb!(batch::CommandBatch, bq::BatchQueue)
    threshold = CB_SPLIT_THRESHOLD[]
    threshold <= 0 && return
    batch.segment_dispatches < threshold && return

    # Seal current CB
    throw_if_error(bq, "vkEndCommandBuffer", Vulkan.end_command_buffer(batch.cmd_buf))
    push!(batch.sealed_cmd_bufs, batch.cmd_buf)

    # Start fresh CB segment
    batch.cmd_buf = alloc_cmd_buf(bq)
    throw_if_error(bq, "vkBeginCommandBuffer",
        Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
            flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )))
    batch.segment_dispatches = 0
    # recording stays true; dispatch_count stays (for barrier logic + total tracking)
end

# ── Recording API ──
#
# All dispatch functions (compute, indirect, RT) use `record_dispatch!` to
# handle the shared boilerplate: get batch, insert barrier, run user code,
# update bookkeeping. The `do` block contains only the dispatch-specific
# Vulkan commands.

"""
    record_dispatch!(f, bq; dst_stage, extra_dst_access=0, is_rt=false, info="")

Record a dispatch into the active command batch. Handles:
1. Getting/creating the active batch
2. Inserting a pipeline barrier if this isn't the first dispatch
3. Calling `f(batch)` with the CommandBatch
4. Updating dispatch count, counter, and debug log

The do-block receives the `CommandBatch`, not a raw command buffer. This lets
dispatch functions `pin!` resources (pipelines, indirect buffers, closures)
into `batch.pinned` alongside recording commands — every pinned object is
kept alive until the timeline signals and also visited by `sync_access!` at
submit.

Example:
    record_dispatch!(bq; dst_stage=PIPELINE_STAGE_COMPUTE_SHADER_BIT) do batch
        Vulkan.cmd_bind_pipeline(batch.cmd_buf, ...)
        pin!(batch, pipeline)
        Vulkan.cmd_dispatch(batch.cmd_buf, ...)
    end
"""
@inline function record_dispatch!(f, bq::BatchQueue;
                           dst_stage::Vulkan.PipelineStageFlag,
                           extra_dst_access::Vulkan.AccessFlag=Vulkan.AccessFlag(0),
                           is_rt::Bool=false,
                           info::String="")
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    # Memory barrier between dispatches (write→read synchronization).
    # Uses direct ccall to avoid Vulkan.jl wrapper allocations (~1.2KB/call).
    if batch.dispatch_count > 0
        src_stage = batch.last_was_rt ?
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        dst_access = VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) | VkAccessFlags(extra_dst_access)
        barrier_ref = Ref(VkMemoryBarrier(
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
            VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT), dst_access))
        GC.@preserve barrier_ref begin
            ccall(CMD_PIPELINE_BARRIER_FPTR[], Cvoid,
                  (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                   UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
                  cmd.vks,
                  VkPipelineStageFlags(src_stage), VkPipelineStageFlags(dst_stage), VkDependencyFlags(0),
                  UInt32(1), barrier_ref,
                  UInt32(0), C_NULL,
                  UInt32(0), C_NULL)
        end
    end

    # Record the actual dispatch commands. The do-block receives the batch
    # so it can pin! resources alongside recording commands.
    f(batch)

    # Bookkeeping
    batch.dispatch_count += 1
    batch.segment_dispatches += 1
    batch.last_was_rt = is_rt
    if DISPATCH_LOGGING_ENABLED[]
        Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, 1)
        log_dispatch!("$(TOTAL_DISPATCH_COUNTER[]) $info")
    end

    # Split to a new CB if this segment is full
    maybe_split_cb!(batch, bq)

    # Auto-submit to avoid TDR when a single submission's GPU execution time
    # approaches amdgpu's ~10 s lockup_timeout.  `submit!` queues the batch
    # onto the in-flight list without blocking — next `record_dispatch!` will
    # `ensure_active_batch!` a fresh batch.  Cross-batch buffer synchronisation
    # is already handled via `sync_access!` writing `buf.last_write` and
    # wait_semaphores picking it up on the next pin.
    threshold = AUTO_SUBMIT_THRESHOLD[]
    if threshold > 0 && batch.dispatch_count >= threshold
        submit!(bq)
    end
end

"""
    push_constants!(cmd, layout, stage_flags, push_data)

Record push constant update. No-op if push_data is empty.
"""
function push_constants!(cmd::Vulkan.CommandBuffer, layout::Vulkan.PipelineLayout,
                          stage_flags, push_data::Vector{UInt8})
    isempty(push_data) && return
    GC.@preserve push_data begin
        Vulkan.cmd_push_constants(cmd, layout, stage_flags,
            UInt32(0), UInt32(length(push_data)), Ptr{Nothing}(pointer(push_data)))
    end
end

"""
    push_constants_bda!(cmd, layout, stage_flags, bda)

Record an 8-byte BDA push constant update. Uses a stack-allocated Ref
(concurrency-safe and still zero-heap-alloc since Refs of primitives are
stack-promoted by the Julia compiler).
"""
@inline function push_constants_bda!(cmd::Vulkan.CommandBuffer, layout::Vulkan.PipelineLayout,
                                      stage_flags, bda::UInt64)
    ref = Ref(bda)
    GC.@preserve ref begin
        Vulkan.cmd_push_constants(cmd, layout, stage_flags,
            UInt32(0), UInt32(8),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, ref)))
    end
end

# ── Compute Dispatch ──

"""
    vk_dispatch!(pipeline, push_bda, groups)

Record a compute dispatch.
`push_bda` is the BDA address of the argument buffer (passed as 8-byte push constant).
"""
function vk_dispatch!(bq::BatchQueue, pipeline::LavaComputePipeline, push_bda::UInt64,
                      groups::NTuple{3, Integer};
                      tlas=nothing)  # Union{Nothing, HWTLAS} — declared later in raytracing/hwtlas.jl
    vk_dispatch_base!(bq, pipeline, push_bda, 0, 0, 0, Int(groups[1]), Int(groups[2]), Int(groups[3]); tlas)
end

"""Record a single compute dispatch with optional base group offset."""
@inline function vk_dispatch_base!(bq::BatchQueue, pipeline::LavaComputePipeline, push_bda::UInt64,
                            base_x::Int, base_y::Int, base_z::Int,
                            gx::Int, gy::Int, gz::Int;
                            tlas=nothing)  # Union{Nothing, HWTLAS} — declared later in raytracing/hwtlas.jl
    dispatch_info = if DISPATCH_LOGGING_ENABLED[]
        info = LAST_DISPATCH_INFO[]
        "$info base=($base_x,$base_y,$base_z) g=($gx,$gy,$gz)"
    else
        ""
    end
    extra_dst_access = pipeline.needs_tlas_descriptor ?
        Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR :
        Vulkan.AccessFlag(0)
    record_dispatch!(bq;
        dst_stage=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        extra_dst_access,
        info=dispatch_info
    ) do batch
        cmd = batch.cmd_buf
        Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        pin!(batch, pipeline)

        # Bind TLAS descriptor set when the pipeline requires it.
        if pipeline.needs_tlas_descriptor
            hw_tlas = tlas::HWTLAS  # caller already verified non-nothing
            lava_tlas = hw_tlas.hw_tlas::LavaTLAS
            dev = bq.ctx.device
            desc_pool, desc_set = alloc_compute_tlas_descriptor_set(dev, pipeline, lava_tlas)
            Vulkan.cmd_bind_descriptor_sets(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline_layout, UInt32(0), [desc_set], UInt32[])
            pin!(batch, desc_pool)
            pin!(batch, lava_tlas.accel)
            pin!(batch, lava_tlas.storage)
            # Pin every BLAS the TLAS references.  rayQuery walks the TLAS into
            # its BLASes and reads their storage; without pinning each BLAS
            # storage, `Raycore.sync!`-driven BLAS swaps can free a BLAS whose
            # GPU memory the GPU is still using through this dispatch.  Pinning
            # the storage updates its `last_write` to this batch's signal,
            # which makes the next `unsafe_free!(blas)` correctly defer until
            # the dispatch has retired.
            for blas in lava_tlas.blases
                pin!(batch, blas.accel)
                pin!(batch, blas.storage)
            end
        end

        push_constants_bda!(cmd, pipeline.pipeline_layout, Vulkan.SHADER_STAGE_COMPUTE_BIT, push_bda)

        if base_x == 0 && base_y == 0 && base_z == 0
            Vulkan.cmd_dispatch(cmd, UInt32(gx), UInt32(gy), UInt32(gz))
        else
            Vulkan.cmd_dispatch_base(cmd,
                UInt32(base_x), UInt32(base_y), UInt32(base_z),
                UInt32(gx), UInt32(gy), UInt32(gz))
        end
    end
end

# ── Indirect Dispatch ──

"""
    vk_dispatch_indirect!(bq, pipeline, push_bda, indirect::LavaArray{UInt32,1})

Record an indirect compute dispatch.  `indirect` is a LavaArray view of 3
UInt32s (groupCountX/Y/Z), typically obtained from `get_indirect_buffer(bq)`
and populated by a prepare-indirect kernel.
"""
@inline function vk_dispatch_indirect!(bq::BatchQueue, pipeline::LavaComputePipeline,
                                       push_bda::UInt64,
                                       indirect;  # LavaArray{UInt32,1} — declared later in array/lavaarray.jl
                                       tlas=nothing)  # Union{Nothing, HWTLAS} — declared later in raytracing/hwtlas.jl
    dispatch_info = DISPATCH_LOGGING_ENABLED[] ?
        "$(LAST_DISPATCH_INFO[]) (indirect)" : ""

    extra_dst_access = Vulkan.ACCESS_INDIRECT_COMMAND_READ_BIT |
        (pipeline.needs_tlas_descriptor ? Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR :
                                          Vulkan.AccessFlag(0))
    record_dispatch!(bq;
        dst_stage=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT | Vulkan.PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        extra_dst_access,
        info=dispatch_info
    ) do batch
        cmd = batch.cmd_buf
        Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        pin!(batch, pipeline)

        # Bind TLAS descriptor set when the pipeline requires it.
        if pipeline.needs_tlas_descriptor
            hw_tlas = tlas::HWTLAS  # caller already verified non-nothing
            lava_tlas = hw_tlas.hw_tlas::LavaTLAS
            dev = bq.ctx.device
            desc_pool, desc_set = alloc_compute_tlas_descriptor_set(dev, pipeline, lava_tlas)
            Vulkan.cmd_bind_descriptor_sets(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline_layout, UInt32(0), [desc_set], UInt32[])
            pin!(batch, desc_pool)
            pin!(batch, lava_tlas.accel)
            pin!(batch, lava_tlas.storage)
            # Pin every BLAS the TLAS references.  rayQuery walks the TLAS into
            # its BLASes and reads their storage; without pinning each BLAS
            # storage, `Raycore.sync!`-driven BLAS swaps can free a BLAS whose
            # GPU memory the GPU is still using through this dispatch.  Pinning
            # the storage updates its `last_write` to this batch's signal,
            # which makes the next `unsafe_free!(blas)` correctly defer until
            # the dispatch has retired.
            for blas in lava_tlas.blases
                pin!(batch, blas.accel)
                pin!(batch, blas.storage)
            end
        end

        push_constants_bda!(cmd, pipeline.pipeline_layout, Vulkan.SHADER_STAGE_COMPUTE_BIT, push_bda)

        mb = indirect.buf[]::VkManagedBuffer
        byte_offset = UInt64(indirect.offset)
        Vulkan.cmd_dispatch_indirect(cmd, mb.buffer, byte_offset)
        pin!(batch, indirect)
    end
end

# ── Flush ──

"""
    query_timeline(bq::BatchQueue) -> UInt64

Return the current counter of `bq.timeline_sem`.  Replaces the old
`try ... catch; typemax(UInt64); end` sentinel:

  * On success, return the real counter value.
  * On any `VulkanError`, set `bq.ctx.device_lost` (if the code is
    `ERROR_DEVICE_LOST`) and throw `LavaVulkanError`.  No silent fallback —
    every caller must either have cleared the device-lost flag upstream
    (so the happy path runs) or be prepared to propagate the throw.

Callers are expected to gate on `device_lost(bq.ctx)` themselves before
calling this.  The only way the throw path fires is the race window where
the device died between that upstream check and this query — and in that
case, loud is correct: we want the dispatcher / finalizer log to show it.
"""
@inline function query_timeline(bq::BatchQueue)::UInt64
    result = Vulkan.get_semaphore_counter_value(bq.device, bq.timeline_sem)
    iserror(result) || return unwrap(result)
    mark_if_lost!(bq, result)
    e = unwrap_error(result)::Vulkan.VulkanError
    throw(LavaVulkanError("get_semaphore_counter_value", Int32(e.code), e.msg,
        e.code == Vulkan.ERROR_DEVICE_LOST ?
            "Device lost. Call Lava.vk_reset_device!() to reinitialize, or restart Julia." :
            "Timeline semaphore query failed with an unexpected VkResult. " *
            "This is a driver bug, stack corruption, or an invalid semaphore handle."))
end

# ── Vulkan call helpers: single source of truth for device-lost handling ──
#
# Every Vulkan call that can fail should funnel through `throw_if_error` (or
# its non-throwing sibling `mark_if_lost!` for callers with custom recovery).
# This is the SOLE place the "VkResult → device_lost flag" rule lives, so
# the dispatcher gate at `ensure_active_batch!` reliably fires after any
# DEVICE_LOST regardless of which low-level call surfaced the error.

"""
    mark_if_lost!(bq::BatchQueue, result)
    mark_if_lost!(ctx::VkContext, result)

Mark `ctx.device_lost = true` iff `result` is a Vulkan `ERROR_DEVICE_LOST`.
Does not unwrap, does not throw.  Use this when a caller needs to do its
own recovery before throwing (`submit!`, `flush!`).  Otherwise prefer
`throw_if_error` which handles the whole pattern.
"""
@inline function mark_if_lost!(ctx::VkContext, result)
    iserror(result) || return
    e = unwrap_error(result)::Vulkan.VulkanError
    e.code == Vulkan.ERROR_DEVICE_LOST && mark_device_lost!(ctx)
    return
end
@inline mark_if_lost!(bq::BatchQueue, result) = mark_if_lost!(bq.ctx::VkContext, result)

device_lost_hint(call) =
    "Vulkan device is lost during $(call). " *
    "Call Lava.vk_reset_device!() to reinitialize, or restart Julia."

"""
    throw_if_error(bq::BatchQueue, result)
    throw_if_error(ctx::VkContext, result)
    throw_if_error(ctx_or_bq, call::String, result)       # adds a call name

Canonical "do everything" Vulkan-call wrapper.
On success → returns the unwrapped value.
On error   → marks `ctx.device_lost` (if the error is `DEVICE_LOST`) and
             throws `LavaVulkanError` with a hint.

Replace any `unwrap(Vulkan.foo(...))` with `throw_if_error(ctx, Vulkan.foo(...))`
to get the device-lost handling for free.  The wrappers below
(`queue_submit!`, `wait_for_fences!`, ...) are thin spellings of this.
"""
@inline function throw_if_error(ctx::VkContext, call::String, result)
    iserror(result) || return unwrap(result)
    mark_if_lost!(ctx, result)
    e = unwrap_error(result)::Vulkan.VulkanError
    suggestion = e.code == Vulkan.ERROR_DEVICE_LOST ? device_lost_hint(call) : ""
    throw(LavaVulkanError(call, Int32(e.code), e.msg, suggestion))
end
@inline throw_if_error(ctx::VkContext, result) = throw_if_error(ctx, "Vulkan call", result)
@inline throw_if_error(bq::BatchQueue, args...) = throw_if_error(bq.ctx::VkContext, args...)

"""
    queue_submit!(bq, submits; fence=Vulkan.Fence(C_NULL))

`vkQueueSubmit` wrapper.  Auto-marks device_lost on failure and throws
`LavaVulkanError`.
"""
@inline queue_submit!(bq::BatchQueue, submits::AbstractVector{Vulkan.SubmitInfo};
                      fence=Vulkan.Fence(C_NULL)) =
    throw_if_error(bq, "vkQueueSubmit", Vulkan.queue_submit(bq.queue, submits; fence=fence))

"""
    queue_submit_2!(bq, submits; fence=Vulkan.Fence(C_NULL))

`vkQueueSubmit2` wrapper.  Auto-marks device_lost on failure.
"""
@inline queue_submit_2!(bq::BatchQueue, submits::AbstractVector{Vulkan.SubmitInfo2};
                        fence=Vulkan.Fence(C_NULL)) =
    throw_if_error(bq, "vkQueueSubmit2", Vulkan.queue_submit_2(bq.queue, submits; fence=fence))

"""
    wait_for_fences!(bq, fences; wait_all=true, timeout=typemax(UInt64))

`vkWaitForFences` wrapper.  Auto-marks device_lost on failure.
"""
@inline wait_for_fences!(bq::BatchQueue, fences;
                         wait_all::Bool=true, timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitForFences", Vulkan.wait_for_fences(bq.device, fences, wait_all, timeout))

"""
    wait_semaphores!(bq, info; timeout=typemax(UInt64))

`vkWaitSemaphores` wrapper.  Auto-marks device_lost on failure.
"""
@inline wait_semaphores!(bq::BatchQueue, info::Vulkan.SemaphoreWaitInfo;
                         timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitSemaphores", Vulkan.wait_semaphores(bq.device, info, timeout))

"""
    submit!(bq::BatchQueue) -> CommandBatch or nothing

Close the active batch's command buffer, run `sync_access!` over every
pinned object to populate `wait_semaphores` + update `last_write`, and
submit to the queue.  Retirement is synchronous: `sweep_retired_batches!`
polls the timeline semaphore at natural sync points (flush, next
`ensure_active_batch!`) and reclaims batches whose signal value has been
reached.

Returns the submitted `CommandBatch` (or `nothing` if no batch was active).
Use `flush!(bq, device)` for the blocking wrapper that waits for GPU
completion before returning.
"""
function submit!(bq::BatchQueue)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread submit forbidden"
    batch = bq.active_batch
    batch === nothing && return nothing
    !batch.recording && return nothing
    @assert batch.bq === bq "batch.bq desync: batch was not bound to this BatchQueue"
    Threads.atomic_add!(FLUSH_COUNTER, 1)

    throw_if_error(bq, "vkEndCommandBuffer", Vulkan.end_command_buffer(batch.cmd_buf))

    n_sealed = length(batch.sealed_cmd_bufs)
    all_cmd_bufs = Vector{Vulkan.CommandBuffer}(undef, n_sealed + 1)
    for i in 1:n_sealed
        all_cmd_bufs[i] = batch.sealed_cmd_bufs[i]
    end
    all_cmd_bufs[n_sealed + 1] = batch.cmd_buf

    saved_dispatch_count = batch.dispatch_count
    saved_last_was_rt = batch.last_was_rt
    PREV_DISPATCH_INFO[] = LAST_DISPATCH_INFO[]

    # ensure_active_batch! pre-assigned batch.signal_value = next_timeline + 1;
    # bump the counter now and assert consistency.
    bq.next_timeline += 1
    @assert batch.signal_value == bq.next_timeline "batch signal desync: $(batch.signal_value) vs $(bq.next_timeline)"

    # Apply per-object access semantics over the full pinned set.  For
    # VkManagedBuffers this populates batch.wait_semaphores and writes
    # (bq, signal_value) into buf.last_write.  Runs ONCE per batch, not per
    # dispatch — the IdSet dedupes multi-dispatch reuse for free.
    for obj in batch.pinned
        sync_access!(batch, obj)
    end

    # Pre-submit safety scan: catch stale-BDA-in-arg-slab corruption BEFORE
    # the GPU sees it.  Off by default (Lava.PRESUBMIT_SCAN_ENABLED[] = true to
    # turn on for debugging).  Cost ~hundreds-of-µs per submit; never on by
    # default.
    if PRESUBMIT_SCAN_ENABLED[]
        unknowns = scan_slabs_for_unknown_bdas(bq)
        if !isempty(unknowns)
            @warn "Pre-submit found $(length(unknowns)) unknown BDA(s) in arg slabs"
            for u in unknowns
                @warn "  STALE: slab=$(u.slab) idx=$(u.idx) offset=$(u.offset) val=0x$(string(u.val, base=16, pad=16))"
            end
            if PRESUBMIT_SCAN_THROWS[]
                throw(LavaError("submit!", "stale BDA in arg slab", "see warnings"))
            end
        end
    end

    # SLAB DUMP for cascade investigation: if SLAB_DUMP_TARGET[] is non-zero,
    # search the active arg slab for any UInt64 == target and log offsets.
    if SLAB_DUMP_TARGET[] != UInt64(0) &&
       !isempty(bq.arg_slabs) && bq.arg_slab_idx <= length(bq.arg_slabs)
        target = SLAB_DUMP_TARGET[]
        slab = bq.arg_slabs[bq.arg_slab_idx]
        mb = slab.buf[]
        mp = mb.mapped_ptr
        if mp != Ptr{UInt8}(0)
            n = bq.arg_slab_offset ÷ 8
            p = Ptr{UInt64}(mp)
            hits = Int[]
            for k in 0:(n-1)
                if unsafe_load(p, k+1) == target
                    push!(hits, k*8)
                end
            end
            if !isempty(hits)
                push!(SLAB_DUMP_LOG, (sub=Int(bq.next_timeline)+1, target=target, offsets=hits))
            end
        end
    end

    cb_infos = [Vulkan.CommandBufferSubmitInfo(cb, UInt32(0)) for cb in all_cmd_bufs]
    wait_infos = [Vulkan.SemaphoreSubmitInfo(s, v, UInt32(0); stage_mask=stage)
                  for (s, v, stage) in batch.wait_semaphores]
    signal_info = Vulkan.SemaphoreSubmitInfo(bq.timeline_sem, batch.signal_value, UInt32(0);
                                             stage_mask=Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    submit_info = Vulkan.SubmitInfo2(wait_infos, cb_infos, [signal_info])
    submit_result = Vulkan.queue_submit_2(bq.queue, [submit_info])
    if iserror(submit_result)
        # `mark_if_lost!` is the canonical place that flips the
        # device_lost flag for any VkResult error.  We then run our own
        # recovery (drop the recording batch) and rethrow with rich
        # validation context, since submit! has dispatch-count / RT-state
        # detail that the generic wrapper can't produce.
        mark_if_lost!(bq, submit_result)
        batch.recording = false
        empty!(batch.pinned)
        empty!(batch.wait_semaphores)
        bq.active_batch = nothing
        throw_with_validation_context("vkQueueSubmit2", submit_result,
            saved_dispatch_count, saved_last_was_rt)
    end

    push!(bq.in_flight, batch)
    bq.active_batch = nothing
    LAST_DISPATCH_INFO[] = "$saved_dispatch_count dispatches ($(saved_last_was_rt ? "RT" : "compute"))"
    Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, saved_dispatch_count)
    if DISPATCH_LOGGING_ENABLED[]
        append!(DISPATCH_LOG, batch.dispatch_log)
    end
    # DEBUG: synchronous per-batch wall-clock timing. Serializes the pipeline
    # but lets us see GPU execution time per batch. Guarded by opt-in flag.
    if BATCH_TIMING_ENABLED[]
        t0 = time()
        wr = Vulkan.wait_semaphores(bq.device,
            Vulkan.SemaphoreWaitInfo([bq.timeline_sem], [batch.signal_value]),
            typemax(UInt64))
        dt = time() - t0
        push!(BATCH_WAIT_TIMES, dt)
        push!(BATCH_WAIT_INFO, LAST_DISPATCH_INFO[])
        push!(BATCH_WAIT_DISPATCHES, saved_dispatch_count)
        # Don't throw from here — let the next call surface it normally — but
        # do mark device_lost so the next dispatcher gate fires.
        mark_if_lost!(bq, wr)
    end
    return batch
end

"""
    sweep_retired_batches!(bq::BatchQueue)

Reclaim any in-flight batches whose signal_value has been reached.  Uses
the non-blocking `get_semaphore_counter_value` to poll the timeline.
Called from the main thread at natural quiescent points.
"""
function sweep_retired_batches!(bq::BatchQueue)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread sweep forbidden"
    isempty(bq.in_flight) && return
    device_lost(bq.ctx::VkContext) && return
    current = query_timeline(bq)
    i = 1
    while i <= length(bq.in_flight)
        b = bq.in_flight[i]
        if b.signal_value <= current
            # Batch has been signalled — safe to reclaim (clears pinned +
            # wait_semaphores inside reclaim_batch!).
            reclaim_batch!(bq, b)
            deleteat!(bq.in_flight, i)
        else
            i += 1
        end
    end
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    # Reset pools once the queue is idle.  Done HERE (not inside reclaim_batch!)
    # because reclaim is called with the batch still in `bq.in_flight` — the
    # isempty check had to run after `deleteat!` to ever return true.
    if isempty(bq.in_flight)
        reset_arg_buffer_pool!(bq)
        reset_indirect_buffer_pool!(bq)
    end
    return nothing
end

"""
    flush!(bq::BatchQueue, device::Vulkan.Device)

Submit the active batch (if any) and block until every in-flight batch on
`bq` has been signalled on the queue's timeline semaphore.  Uses a single
`wait_semaphores` call on the HIGHEST pending signal value — the timeline
is monotonic, so once that value is reached every lower value is too.
"""
function flush!(bq::BatchQueue, device::Vulkan.Device)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread flush forbidden"
    submit!(bq)
    isempty(bq.in_flight) && return
    target = maximum(b.signal_value for b in bq.in_flight)
    wait_result = Vulkan.wait_semaphores(device,
        Vulkan.SemaphoreWaitInfo([bq.timeline_sem], [target]),
        typemax(UInt64))
    if iserror(wait_result)
        # Rich rethrow: flush! gets validation context + dispatcher hints the
        # generic wrapper can't.  The mark_if_dl! call is the single source of
        # truth for the device_lost flag.
        mark_if_lost!(bq, wait_result)
        throw_with_validation_context("vkWaitSemaphores", wait_result, 0, false)
    end
    sweep_retired_batches!(bq)
    check_validation_errors!("vk_flush!")
    return
end

# ── Unified pin + sync primitive ──────────────────────────────────────────
#
# Two phases, two functions.  Every object a dispatch touches goes through:
#
#   1. `pin!(batch, obj)` — called from the @generated arg-pack walker and
#      from dispatch entry points (rt_dispatch!, vk_draw!).  Idempotent
#      insert into `batch.pinned::IdSet`.  User never calls this.
#
#   2. `sync_access!(batch, obj)` — invoked in `submit!` once per pinned
#      object, before queue_submit_2.  Default is no-op.  `VkManagedBuffer`
#      specialization updates `last_write` and inserts a cross-queue timeline-
#      semaphore wait when the prior writer was on a different queue.
#
# Overloadable per type: a new resource kind adds `pin!(batch, ::MyRes)`
# (optional; generic method already handles it) and `sync_access!(batch,
# ::MyRes)` (optional; default no-op).  No edits to the pack walker or to
# submit! required.

"""
    pin!(batch::CommandBatch, obj) -> nothing

Keep `obj` alive until this batch's timeline signals.  Idempotent: repeated
calls with the same `obj` (within one batch) are no-ops.  `VkManagedBuffer`
specialization additionally asserts the ctx invariant so cross-context use
surfaces immediately.

Called from `pack_args_direct!` at each buffer-typed leaf, and directly at
dispatch entry points for objects not in the kernel arg tuple (pipelines,
closures, RT AS handles, indirect buffers).
"""
@inline function pin!(batch::CommandBatch, obj)
    obj in batch.pinned && return nothing
    push!(batch.pinned, obj)
    return nothing
end

@inline function pin!(batch::CommandBatch, buf::VkManagedBuffer)
    bq = batch.bq::BatchQueue
    @assert buf.ctx === bq.ctx  "cross-ctx buffer use forbidden"
    buf in batch.pinned && return nothing
    push!(batch.pinned, buf)
    return nothing
end

"""
    sync_access!(batch::CommandBatch, obj) -> nothing

Apply access semantics for `obj` at `submit!` time.  Default: no-op (plain
GC pinning was enough).  Overload for resource types that need GPU-side
synchronization.

The `VkManagedBuffer` specialization:
  • If `buf.last_write` was set by a different queue, pushes a timeline-
    semaphore wait onto `batch.wait_semaphores` so the submit blocks on the
    prior queue's signal.
  • Updates `buf.last_write = (bq, batch.signal_value)`.
"""
@inline sync_access!(::CommandBatch, _) = nothing

@inline function sync_access!(batch::CommandBatch, buf::VkManagedBuffer)
    # Any buffer reaching sync_access! must be ALIVE at submit time — if a
    # dead buffer's state transitioned to DEFERRED/DEAD before we got here,
    # the GPU is about to read freed memory.  Trip loudly.
    @assert (@atomic :acquire buf.state) == BUF_STATE_ALIVE  "sync_access!: buffer is not ALIVE (state=$(@atomic :acquire buf.state)) — use-after-free"
    bq = batch.bq::BatchQueue
    lw = @atomic :acquire buf.last_write
    if lw !== nothing
        prev_bq, prev_val = lw[1]::BatchQueue, lw[2]::UInt64
        if prev_bq !== bq
            push!(batch.wait_semaphores,
                (prev_bq.timeline_sem, prev_val,
                 Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
        end
    end
    @atomic :release buf.last_write = (bq, batch.signal_value)
    return nothing
end

# LavaArray forwarder lives co-located with its type def in
# array/lavaarray.jl — it unwraps to VkManagedBuffer so the leaf
# pin!/sync_access! do the work.

"""
    wait_for_write(buf::VkManagedBuffer)

Block the calling thread until the GPU has finished the latest write to
`buf` (per `buf.last_write`). Used by CPU-side readbacks. No-op if the
buffer was never written by any queue.
"""
function wait_for_write(buf::VkManagedBuffer)
    lw = @atomic :acquire buf.last_write
    lw === nothing && return nothing
    bq, val = lw[1]::BatchQueue, lw[2]::UInt64
    # If the target signal value hasn't been submitted yet (there's still an
    # active recording on that bq whose signal_value >= val), the semaphore
    # would never reach it.  Flush that bq first to force the submit.
    active = bq.active_batch
    if active !== nothing && active.signal_value <= val
        flush!(bq, bq.device)
    end
    # Any value <= current counter is already signalled; skip the wait to
    # avoid a pointless syscall.  query_timeline rethrows on a
    # healthy-device failure (no silent "pretend not yet" fallback).
    current = query_timeline(bq)
    if current >= val
        return nothing
    end
    wait_semaphores!(bq, Vulkan.SemaphoreWaitInfo([bq.timeline_sem], [val]))
    return nothing
end

"""
    vk_flush!(bq::BatchQueue)
    vk_flush!(ctx::VkContext)       # flushes ctx.default_bq

Flush a specific batch queue.  Always spell the queue (or its ctx) explicitly —
zero-arg convenience forms have been removed per the "Explicit arguments over
implicit state" rule.
"""
function vk_flush!(bq::BatchQueue)
    ctx = bq.ctx::VkContext
    device_lost(ctx) && throw(LavaError("command flush", "Vulkan device lost",
        "Call Lava.vk_reset_device!() to reinitialize, or restart Julia session."))
    flush!(bq, bq.device)
    return
end
vk_flush!(ctx::VkContext) = vk_flush!(ctx.default_bq)

# ── Buffer copy inside the batch ────────────────────────────────────────
#
# Replaces the legacy `one_shot_copy` / `append_copy_and_flush!` pair.  All
# buffer-to-buffer transfers record a `cmd_copy_buffer` command into the
# active CommandBatch, pin both source and destination into `batch.pinned`,
# and insert a shader-write → transfer-read barrier when necessary.  The
# caller decides whether to flush afterwards.
#
# Pinning the VkManagedBuffer wrappers routes through sync_access! at submit,
# which inserts the cross-queue timeline-semaphore wait if src/dst were last
# written by a different queue — so the transfer automatically observes any
# prior producer regardless of which queue it ran on.

"""
    cmd_copy_buffer!(bq, src, dst, nbytes; src_off=0, dst_off=0)

Record a GPU→GPU buffer copy into `bq`'s active CommandBatch.  `src` and
`dst` may be either `VkManagedBuffer` (pinned + sync-tracked) or raw
`Vulkan.Buffer` handles (no pinning — caller must keep them alive).
"""
function cmd_copy_buffer!(bq::BatchQueue, src, dst, nbytes::Integer;
                          src_off::Integer=0, dst_off::Integer=0)
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    # Barrier: shader writes → transfer read. Only needed if we already
    # recorded dispatches into this batch (barrier across those writes).
    if batch.dispatch_count > 0
        src_stage = batch.last_was_rt ?
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        barrier_ref = Ref(VkMemoryBarrier(
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
            VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT),
            VkAccessFlags(VK_ACCESS_TRANSFER_READ_BIT)))
        GC.@preserve barrier_ref begin
            ccall(CMD_PIPELINE_BARRIER_FPTR[], Cvoid,
                  (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                   UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
                  cmd.vks,
                  VkPipelineStageFlags(src_stage),
                  VkPipelineStageFlags(VK_PIPELINE_STAGE_TRANSFER_BIT),
                  VkDependencyFlags(0),
                  UInt32(1), barrier_ref,
                  UInt32(0), C_NULL,
                  UInt32(0), C_NULL)
        end
    end

    src_vkbuf = src isa VkManagedBuffer ? src.buffer : src
    dst_vkbuf = dst isa VkManagedBuffer ? dst.buffer : dst
    region = Vulkan.BufferCopy(UInt64(src_off), UInt64(dst_off), UInt64(nbytes))
    Vulkan.cmd_copy_buffer(cmd, src_vkbuf, dst_vkbuf, [region])

    # Pin + sync-track the VkManagedBuffers so the transfer respects any
    # prior cross-queue writer via batch.wait_semaphores.
    src isa VkManagedBuffer && pin!(batch, src)
    dst isa VkManagedBuffer && pin!(batch, dst)
    return nothing
end

# ── Error Reporting ──

"""Throw a LavaError enriched with recent validation layer messages and dispatch log."""
function throw_with_validation_context(call_name::String, err_result,
        dispatch_count::Int=0, last_was_rt::Bool=false)
    # Re-enable dispatch logging so the next run captures debug info
    DISPATCH_LOGGING_ENABLED[] = true
    vk_err = unwrap_error(err_result)
    msgs = get_validation_messages()
    validation_detail = if isempty(msgs)
        "No validation messages captured. Install vulkan-validationlayers for GPU error diagnostics."
    else
        n = min(length(msgs), 10)
        "Last $n validation message(s):\n" * join(["  [$i] $(msgs[end-n+i])" for i in 1:n], "\n")
    end

    dispatch_detail = if isempty(DISPATCH_LOG)
        "No dispatches logged."
    else
        "Recent dispatch log (last $(length(DISPATCH_LOG))):\n" *
        join(["  $d" for d in DISPATCH_LOG], "\n")
    end

    total = TOTAL_DISPATCH_COUNTER[]
    prev_info = PREV_DISPATCH_INFO[]
    curr_info = LAST_DISPATCH_INFO[]
    throw(LavaError(
        call_name,
        """$vk_err after $dispatch_count dispatches in batch ($total total, last_was_rt=$last_was_rt)
Crashed batch dispatch: $prev_info
Triggered by recording: $curr_info
$validation_detail
$dispatch_detail""",
        "DEVICE_LOST usually means invalid SPIR-V, out-of-bounds BDA access, or GPU timeout (Xid 109). Check dispatch log above for the crashing kernel. Call Lava.vk_reset_device!() to reinitialize."
    ))
end

# ── Debugging API ──

"""
    set_dispatch_logging!(enabled::Bool)

Enable or disable dispatch name logging. When enabled, each dispatch records
its kernel name and parameters for crash debugging. Disabled by default for
zero-alloc performance. Auto-enabled on DEVICE_LOST.
"""
set_dispatch_logging!(enabled::Bool) = (DISPATCH_LOGGING_ENABLED[] = enabled)

"""
    get_dispatch_log() -> Vector{String}

Return a copy of the recent dispatch log (up to $MAX_DISPATCH_LOG entries).
"""
get_dispatch_log() = copy(DISPATCH_LOG)
