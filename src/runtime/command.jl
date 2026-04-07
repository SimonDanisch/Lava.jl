# Command buffer recording and dispatch for Lava.jl
#
# Batch-based command buffer management: each CommandBatch owns its own
# command buffer, fence, and strong refs to in-flight data.
#
# GC safety:
# Layer 1 (proactive): batch.data_refs keeps LavaArray objects alive until flush.
# Layer 2 (structural): DEFERRED_FREES in memory.jl defers Vulkan destroy if
#   a GC finalizer fires during recording.

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
const VK_BARRIER_REF = Ref(VkMemoryBarrier(
    VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
    VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT),
    VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)))
# Function pointer for vkCmdPipelineBarrier — initialized in _init_vulkan!
const CMD_PIPELINE_BARRIER_FPTR = Ref{Ptr{Nothing}}(C_NULL)

# Pre-allocated Ref for BDA push constants (zero-alloc path).
# Used inside push_constants_bda! — set and read synchronously in a single ccall,
# so no aliasing risk from nested dispatches (unlike a shared Vector{UInt8}).
const PUSH_BDA_REF = Ref{UInt64}(0)

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
    batch = bq.active_batch
    if batch !== nothing
        if !batch.recording
            unwrap(Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
                flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            )))
            batch.recording = true
        end
        return batch
    end

    if !isempty(bq.free_batches)
        batch = pop!(bq.free_batches)
    else
        batch = allocate_batch(bq)
    end

    bq.active_batch = batch

    unwrap(Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )))
    batch.recording = true
    return batch
end

# Task-local BatchQueue override: when set, all batch operations on VkContext
# redirect to this BatchQueue instead of default_bq. Used by the RT thread.
const TASK_BATCH_QUEUE_KEY = :lava_batch_queue

"""
    with_batch_queue(f, bq::BatchQueue)

Run `f()` with all batch operations redirected to `bq` instead of the default queue.
The RT thread uses this so kernel dispatches and flushes go to the compute queue.

    with_batch_queue(rt_bq) do
        render!(screen)
        flush!(rt_bq, device)
    end
"""
function with_batch_queue(f, bq::BatchQueue)
    task = current_task()
    if task.storage === nothing
        task.storage = IdDict()
    end
    old = get(task.storage, TASK_BATCH_QUEUE_KEY, nothing)
    task.storage[TASK_BATCH_QUEUE_KEY] = bq
    try
        f()
    finally
        if old === nothing
            delete!(task.storage, TASK_BATCH_QUEUE_KEY)
        else
            task.storage[TASK_BATCH_QUEUE_KEY] = old
        end
    end
end

"""Get the active BatchQueue for the current task (or nothing for default)."""
function current_batch_queue()
    task = current_task()
    task.storage === nothing && return nothing
    return get(task.storage, TASK_BATCH_QUEUE_KEY, nothing)
end

# VkContext convenience: use default batch queue
function ensure_active_batch!(ctx::VkContext)
    return ensure_active_batch!(ctx.default_bq)
end

"""Allocate a new CommandBatch (new command buffer + fence from the pool)."""
function allocate_batch(bq::BatchQueue)
    cmd_buf = alloc_cmd_buf(bq)
    fence = Vulkan.Fence(vk_context().device)
    data_refs = Any[]
    sizehint!(data_refs, 128)
    return CommandBatch(cmd_buf, fence, false, 0, 0, false, data_refs, String[], Vulkan.CommandBuffer[])
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

    # When ALL in-flight batches are done, safe to reset pools and flush deferred frees
    if isempty(bq.in_flight)
        flush_deferred_frees!()
        reset_arg_buffer_pool!()
        reset_indirect_buffer_pool!()
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
        VK_BARRIER_REF[] = VkMemoryBarrier(
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
            VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT), dst_access)
        ccall(CMD_PIPELINE_BARRIER_FPTR[], Cvoid,
              (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
               UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
              cmd.vks,
              VkPipelineStageFlags(src_stage), VkPipelineStageFlags(dst_stage), VkDependencyFlags(0),
              UInt32(1), VK_BARRIER_REF,
              UInt32(0), C_NULL,
              UInt32(0), C_NULL)
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

Record an 8-byte BDA push constant update. Zero-alloc: uses a module-level
Ref{UInt64} that is set and consumed synchronously in a single Vulkan ccall.
"""
@inline function push_constants_bda!(cmd::Vulkan.CommandBuffer, layout::Vulkan.PipelineLayout,
                                      stage_flags, bda::UInt64)
    PUSH_BDA_REF[] = bda
    GC.@preserve PUSH_BDA_REF begin
        Vulkan.cmd_push_constants(cmd, layout, stage_flags,
            UInt32(0), UInt32(8),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, PUSH_BDA_REF)))
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
    flush!(bq::BatchQueue, device::Vulkan.Device)

Submit the active batch on `bq`'s queue and wait for GPU completion.
This is the core flush implementation — `vk_flush!()` delegates here.
"""
function flush!(bq::BatchQueue, device::Vulkan.Device)
    batch = bq.active_batch
    batch === nothing && return
    !batch.recording && return
    Threads.atomic_add!(FLUSH_COUNTER, 1)

    function reset_batch_on_error!()
        batch.recording = false
        batch.dispatch_count = 0
        batch.segment_dispatches = 0
        batch.last_was_rt = false
        empty!(batch.data_refs)
        append!(bq.free_cmd_bufs, batch.sealed_cmd_bufs)
        empty!(batch.sealed_cmd_bufs)
        bq.active_batch = nothing
        push!(bq.free_batches, batch)
    end

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

    submit_info = Vulkan.SubmitInfo([], [], all_cmd_bufs, [])
    submit_result = Vulkan.queue_submit(bq.queue, [submit_info]; fence=batch.fence)
    if iserror(submit_result)
        DEVICE_LOST[] = true
        reset_batch_on_error!()
        throw_with_validation_context("vkQueueSubmit", submit_result,
            saved_dispatch_count, saved_last_was_rt)
    end

    fence_result = Vulkan.wait_for_fences(device, [batch.fence], true, typemax(UInt64))
    if iserror(fence_result)
        DEVICE_LOST[] = true
        reset_batch_on_error!()
        throw_with_validation_context("vkWaitForFences", fence_result,
            saved_dispatch_count, saved_last_was_rt)
    end
    unwrap(Vulkan.reset_fences(device, [batch.fence]))

    LAST_DISPATCH_INFO[] = "$(batch.dispatch_count) dispatches ($(batch.last_was_rt ? "RT" : "compute"))"
    Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, batch.dispatch_count)
    if DISPATCH_LOGGING_ENABLED[]
        append!(DISPATCH_LOG, batch.dispatch_log)
    end

    reclaim_batch!(bq, batch)
    bq.active_batch = nothing
    check_validation_errors!("vk_flush!")
    flush_deferred_frees!()

    return
end

"""
    vk_flush!()

Flush the default batch queue. Convenience wrapper for interactive use.
"""
function vk_flush!()
    DEVICE_LOST[] && throw(LavaError("command flush", "Vulkan device lost",
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
function append_copy_and_flush!(ctx::VkContext, src_buffer::Vulkan.Buffer,
                                  src_offset::Integer, dst_staging::Vulkan.Buffer,
                                  nbytes::Integer)
    batch = ctx.default_bq.active_batch
    cmd = batch.cmd_buf

    # Barrier: shader writes → transfer read
    src_stage = batch.last_was_rt ?
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    VK_BARRIER_REF[] = VkMemoryBarrier(
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
        VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT),
        VkAccessFlags(VK_ACCESS_TRANSFER_READ_BIT))
    ccall(CMD_PIPELINE_BARRIER_FPTR[], Cvoid,
          (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
           UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
          cmd.vks,
          VkPipelineStageFlags(src_stage),
          VkPipelineStageFlags(VK_PIPELINE_STAGE_TRANSFER_BIT),
          VkDependencyFlags(0),
          UInt32(1), VK_BARRIER_REF,
          UInt32(0), C_NULL,
          UInt32(0), C_NULL)

    # Append the copy command
    region = Vulkan.BufferCopy(UInt64(src_offset), UInt64(0), UInt64(nbytes))
    Vulkan.cmd_copy_buffer(cmd, src_buffer, dst_staging, [region])

    # Flush the entire batch (dispatches + copy) in one submit
    vk_flush!()
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
