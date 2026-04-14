# Command buffer recording and dispatch for Lava.jl
#
# Batch-based command buffer management: each CommandBatch owns its own
# command buffer, fence, and strong refs to in-flight data.
#
# GC safety:
# Every dispatch pushes its buffer/pipeline/descriptor args to `batch.data_refs`
# (via `record_arg_accesses!`). The reaper task clears `data_refs` only after
# the batch's timeline semaphore has been signalled — so Vulkan destructors
# fired by Julia's GC can't run while any in-flight batch references the
# object. See `vk_free!`, `drain_deferred_frees!`, and `reap!`.

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


# ── Batch lifecycle ──

"""
    ensure_active_batch!(bq::BatchQueue) -> CommandBatch

Get the active batch, allocating one from the free pool if needed.
Begins command buffer recording if not already started.
"""
function ensure_active_batch!(bq::BatchQueue)
    # Sweep any reaper-retired batches first so their resources recycle and
    # their errors surface before we start new work.
    sweep_retired_batches!(bq)

    batch = bq.active_batch
    if batch !== nothing
        if !batch.recording
            unwrap(Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
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
        # Reset for reuse — fresh Atomic{Bool}(false) so reap! can re-publish.
        batch.retired = Threads.Atomic{Bool}(false)
        batch.reaper_task = nothing
        batch.error = nothing
    else
        batch = allocate_batch(bq)
    end

    bq.active_batch = batch

    unwrap(Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
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

"""Allocate a new CommandBatch (new command buffer + fence from the pool)."""
function allocate_batch(bq::BatchQueue)
    cmd_buf = alloc_cmd_buf(bq)
    return init_batch(bq.device, cmd_buf)
end

"""Allocate a command buffer from the free pool, or create a new one."""
function alloc_cmd_buf(bq::BatchQueue)
    if !isempty(bq.free_cmd_bufs)
        return pop!(bq.free_cmd_bufs)
    end
    alloc_info = Vulkan.CommandBufferAllocateInfo(
        bq.cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1
    )
    return unwrap(Vulkan.allocate_command_buffers(vk_context().device, alloc_info))[1]
end

"""Reclaim a completed batch: reset fence, clear data refs, return to free pool."""
function reclaim_batch!(bq::BatchQueue, batch::CommandBatch)
    batch.recording = false
    batch.dispatch_count = 0
    batch.segment_dispatches = 0
    batch.last_was_rt = false
    empty!(batch.data_refs)
    empty!(batch.dispatch_log)
    append!(bq.free_cmd_bufs, batch.sealed_cmd_bufs)
    empty!(batch.sealed_cmd_bufs)
    push!(bq.free_batches, batch)

    # When ALL in-flight batches are done, safe to reset pools and drain
    # this queue's deferred-frees lists (which are timeline-gated anyway).
    if isempty(bq.in_flight)
        drain_deferred_frees!(bq)
        drain_deferred_as_frees!(bq)
        reset_arg_buffer_pool!(bq)
        reset_indirect_buffer_pool!(bq)
    end
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
    unwrap(Vulkan.end_command_buffer(batch.cmd_buf))
    push!(batch.sealed_cmd_bufs, batch.cmd_buf)

    # Start fresh CB segment
    batch.cmd_buf = alloc_cmd_buf(bq)
    unwrap(Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
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
3. Calling `f(batch)` with the CommandBatch (provides cmd_buf AND data_refs)
4. Updating dispatch count, counter, and debug log

The do-block receives the `CommandBatch`, not a raw command buffer. This lets
dispatch functions push resources to `batch.data_refs` directly, ensuring all
GPU-referenced objects stay alive until flush.

Example:
    record_dispatch!(bq; dst_stage=PIPELINE_STAGE_COMPUTE_SHADER_BIT) do batch
        Vulkan.cmd_bind_pipeline(batch.cmd_buf, ...)
        push!(batch.data_refs, pipeline)
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
    # so it can push resources to data_refs alongside recording commands.
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
                      groups::NTuple{3, Integer})
    vk_dispatch_base!(bq, pipeline, push_bda, 0, 0, 0, Int(groups[1]), Int(groups[2]), Int(groups[3]))
end

"""Record a single compute dispatch with optional base group offset."""
@inline function vk_dispatch_base!(bq::BatchQueue, pipeline::LavaComputePipeline, push_bda::UInt64,
                            base_x::Int, base_y::Int, base_z::Int,
                            gx::Int, gy::Int, gz::Int)
    dispatch_info = if DISPATCH_LOGGING_ENABLED[]
        info = LAST_DISPATCH_INFO[]
        "$info base=($base_x,$base_y,$base_z) g=($gx,$gy,$gz)"
    else
        ""
    end
    record_dispatch!(bq;
        dst_stage=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        info=dispatch_info
    ) do batch
        cmd = batch.cmd_buf
        Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        push!(batch.data_refs, pipeline)
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
    vk_dispatch_indirect!(pipeline, push_bda, indirect_buf, indirect_offset=0)

Record an indirect compute dispatch. The `indirect_buf` must contain a
VkDispatchIndirectCommand (3×UInt32), written by a previous GPU kernel.
"""
@inline function vk_dispatch_indirect!(bq::BatchQueue, pipeline::LavaComputePipeline, push_bda::UInt64,
                               indirect_buf, indirect_offset::Integer=0)
    dispatch_info = DISPATCH_LOGGING_ENABLED[] ?
        "$(LAST_DISPATCH_INFO[]) (indirect)" : ""

    record_dispatch!(bq;
        dst_stage=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT | Vulkan.PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        extra_dst_access=Vulkan.ACCESS_INDIRECT_COMMAND_READ_BIT,
        info=dispatch_info
    ) do batch
        cmd = batch.cmd_buf
        Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        push!(batch.data_refs, pipeline)
        push_constants_bda!(cmd, pipeline.pipeline_layout, Vulkan.SHADER_STAGE_COMPUTE_BIT, push_bda)

        vk_buf = indirect_buf isa Vulkan.Buffer ? indirect_buf : indirect_buf.buffer
        buf_offset = indirect_buf isa VkIndirectBuffer ? indirect_buf.buffer_offset : UInt64(0)
        Vulkan.cmd_dispatch_indirect(cmd, vk_buf, buf_offset + UInt64(indirect_offset))
        push!(batch.data_refs, indirect_buf)
    end
end

# ── Flush ──

"""
    submit!(bq::BatchQueue) -> CommandBatch or nothing

Close the active batch's command buffer, submit to the queue (with cross-queue
timeline waits + this queue's signal), and return immediately. The batch is
retired asynchronously: we spawn a `Threads.@spawn` reaper that blocks on the
timeline semaphore, clears `data_refs`, runs `reclaim_batch!`, and drains the
queue's deferred-free lists. Returns the submitted `CommandBatch` (or
`nothing` if no batch was active).

Use `flush!(bq, device)` for the synchronous wrapper.
"""
function submit!(bq::BatchQueue)
    batch = bq.active_batch
    batch === nothing && return nothing
    !batch.recording && return nothing
    Threads.atomic_add!(FLUSH_COUNTER, 1)

    unwrap(Vulkan.end_command_buffer(batch.cmd_buf))

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

    cb_infos = [Vulkan.CommandBufferSubmitInfo(cb, UInt32(0)) for cb in all_cmd_bufs]
    wait_infos = [Vulkan.SemaphoreSubmitInfo(s, v, UInt32(0); stage_mask=stage)
                  for (s, v, stage) in batch.wait_semaphores]
    signal_info = Vulkan.SemaphoreSubmitInfo(bq.timeline_sem, batch.signal_value, UInt32(0);
                                             stage_mask=Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    submit_info = Vulkan.SubmitInfo2(wait_infos, cb_infos, [signal_info])
    submit_result = Vulkan.queue_submit_2(bq.queue, [submit_info])
    if iserror(submit_result)
        mark_device_lost!(VK_CONTEXT_REF[])
        # Recover: drop the active batch so future submits can proceed.
        batch.recording = false
        empty!(batch.data_refs)
        empty!(batch.wait_semaphores)
        Threads.atomic_xchg!(batch.retired, true)
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
    return batch
end

"""
    sweep_retired_batches!(bq::BatchQueue)

Reclaim any in-flight batches whose signal_value has been reached.  Uses
the non-blocking `get_semaphore_counter_value` to poll the timeline.
Called from the main thread at natural quiescent points.
"""
function sweep_retired_batches!(bq::BatchQueue)
    isempty(bq.in_flight) && return
    ctx = VK_CONTEXT_REF[]
    (ctx === nothing || device_lost(ctx)) && return
    current = try
        unwrap(Vulkan.get_semaphore_counter_value(bq.device, bq.timeline_sem))
    catch
        typemax(UInt64)
    end
    i = 1
    while i <= length(bq.in_flight)
        b = bq.in_flight[i]
        if b.signal_value <= current
            # Batch has been signalled — safe to clear refs and reclaim.
            empty!(b.data_refs)
            empty!(b.wait_semaphores)
            reclaim_batch!(bq, b)
            deleteat!(bq.in_flight, i)
        else
            i += 1
        end
    end
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
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
    submit!(bq)
    isempty(bq.in_flight) && return
    target = maximum(b.signal_value for b in bq.in_flight)
    wait_result = Vulkan.wait_semaphores(device,
        Vulkan.SemaphoreWaitInfo([bq.timeline_sem], [target]),
        typemax(UInt64))
    if iserror(wait_result)
        ctx = VK_CONTEXT_REF[]
        ctx === nothing || mark_device_lost!(ctx)
        throw_with_validation_context("vkWaitSemaphores", wait_result, 0, false)
    end
    sweep_retired_batches!(bq)
    check_validation_errors!("vk_flush!")
    return
end

# ── Buffer-access tracking for cross-queue sync ──────────────────────────
#
# The new lifetime model: every dispatch declares which buffers it touches.
# `record_buffer_access!(bq, batch, buf)` does three things:
#   1. If `buf.last_write` is on a different queue, append a timeline-semaphore
#      wait so the GPU holds this submit until the prior queue signals.
#   2. Update `buf.last_write = (bq, batch.signal_value)`.
#   3. Push `buf` into `batch.data_refs` so it stays alive until the
#      signalled timeline value is reached (and the batch is reclaimed).
#
# Treats every buffer arg as R/W. Cheap when there is only one queue
# (`last_write[1] === bq` always; no semaphore entry added).

"""
    track_buffer_access!(bq, batch, buf::VkManagedBuffer)

Update `buf.last_write` to `(bq, batch.signal_value)` and, if the previous
write was on a different queue, append a timeline-semaphore wait so this
submit blocks on the GPU until the prior queue signals.

Does NOT push to `batch.data_refs` — that's `record_arg_accesses!`'s job.
"""
@inline function track_buffer_access!(bq::BatchQueue, batch::CommandBatch, buf::VkManagedBuffer)
    lw = buf.last_write
    if lw !== nothing
        prev_bq, prev_val = lw[1]::BatchQueue, lw[2]::UInt64
        if prev_bq !== bq
            push!(batch.wait_semaphores,
                (prev_bq.timeline_sem, prev_val,
                 Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
        end
    end
    buf.last_write = (bq, batch.signal_value)
    return nothing
end

"""
    record_arg_accesses!(bq, batch, things...)

For each `thing`:
  • Pin it into `batch.data_refs` (so it survives until fence).
  • If it's a `VkManagedBuffer` / `LavaArray` / `LavaBuffer`: also call
    `track_buffer_access!` to update `last_write` and insert any
    cross-queue semaphore wait.
  • If it's a `Tuple`: recurse into its contents (walks user args).
  • Otherwise: pin only (closures, pipelines, TLAS handles, etc.).

Use this at every dispatch site instead of bare `push!(batch.data_refs, …)`.
One call per dispatch replaces all the lifetime/sync bookkeeping.
"""
@inline function record_arg_accesses!(bq::BatchQueue, batch::CommandBatch, things...)
    for t in things
        push!(batch.data_refs, t)
        record_one!(bq, batch, t)
    end
    return nothing
end

@inline record_one!(::BatchQueue, ::CommandBatch, _) = nothing
@inline record_one!(bq::BatchQueue, batch::CommandBatch, b::VkManagedBuffer) =
    track_buffer_access!(bq, batch, b)
@inline function record_one!(bq::BatchQueue, batch::CommandBatch, args::Tuple)
    for a in args
        record_one!(bq, batch, a)
    end
    return nothing
end
# LavaArray / LavaBuffer methods are added in array/lavaarray.jl and array/ka_backend.jl,
# after their types are defined.

"""
    wait_for_write(buf::VkManagedBuffer)

Block the calling thread until the GPU has finished the latest write to
`buf` (per `buf.last_write`). Used by CPU-side readbacks. No-op if the
buffer was never written by any queue.
"""
function wait_for_write(buf::VkManagedBuffer)
    lw = buf.last_write
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
    # avoid a pointless syscall.
    current = try
        unwrap(Vulkan.get_semaphore_counter_value(bq.device, bq.timeline_sem))
    catch
        UInt64(0)
    end
    if current >= val
        return nothing
    end
    unwrap(Vulkan.wait_semaphores(bq.device,
        Vulkan.SemaphoreWaitInfo([bq.timeline_sem], [val]),
        typemax(UInt64)))
    return nothing
end

"""
    vk_flush!()

Flush the default batch queue. Convenience wrapper for interactive use.
"""
function vk_flush!()
    device_lost() && throw(LavaError("command flush", "Vulkan device lost",
        "Call Lava.vk_reset_device!() to reinitialize, or restart Julia session."))
    ctx = vk_context()
    flush!(ctx.default_bq, ctx.device)
    return
end

# ── Piggybacked Download ──

"""
    append_copy_and_flush!(ctx, src_buffer, src_offset, dst_staging, nbytes)

Append a GPU→staging buffer copy to the active command batch and flush.
Saves one fence roundtrip compared to vk_flush!() + one_shot_copy() by
combining all dispatch commands and the copy into a single submission.

The caller must ensure an active recording exists on the default batch queue.
"""
function append_copy_and_flush!(bq::BatchQueue, src_buffer::Vulkan.Buffer,
                                  src_offset::Integer, dst_staging::Vulkan.Buffer,
                                  nbytes::Integer)
    batch = bq.active_batch
    cmd = batch.cmd_buf

    # Barrier: shader writes → transfer read
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

    # Append the copy command
    region = Vulkan.BufferCopy(UInt64(src_offset), UInt64(0), UInt64(nbytes))
    Vulkan.cmd_copy_buffer(cmd, src_buffer, dst_staging, [region])

    # Flush the entire batch (dispatches + copy) in one submit
    flush!(bq, bq.device)
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
