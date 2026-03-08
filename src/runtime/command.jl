# Command buffer recording and dispatch for Lava.jl
#
# Batch-based command buffer management: each CommandBatch owns its own
# command buffer, fence, and strong refs to in-flight data. Replaces the
# old single cmd_buf/fence + global INFLIGHT_DATA_REFS approach.
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
# _last_dispatch_info: the dispatch currently being recorded
# _prev_dispatch_info: the dispatch in the batch being flushed (what actually crashed)
const _last_dispatch_info = Ref{String}("")
const _prev_dispatch_info = Ref{String}("")

# Ring buffer of last N dispatch names for crash debugging
const _dispatch_log = String[]
const _max_dispatch_log = 50

function _log_dispatch!(info::String)
    if length(_dispatch_log) >= _max_dispatch_log
        popfirst!(_dispatch_log)
    end
    push!(_dispatch_log, info)
end

# Auto-flush threshold: flush command buffer after this many dispatches to prevent
# NVIDIA's GPU watchdog (Xid 109 = CTX SWITCH TIMEOUT) from killing long batches.
# Set to 0 to disable auto-flush. Default tuned for NVIDIA mobile GPUs.
const _auto_flush_threshold = Ref{Int}(16)

# Max workgroups per single dispatch — prevents TDR timeout on NVIDIA GPUs.
# Large dispatches (e.g., 18000 groups of complex material evaluation) are split
# into multiple cmd_dispatch_base calls with flush between chunks.
# 0 = no limit. Default 4096 = ~1M threads at wg256, safe for mobile NVIDIA.
const _max_groups_per_dispatch = Ref{Int}(4096)

"""
    set_max_groups_per_dispatch!(n::Integer)

Set the maximum number of workgroups per single compute dispatch.
Large dispatches are split into chunks using `vkCmdDispatchBase` to avoid
NVIDIA GPU watchdog timeout (Xid 109). Set to 0 to disable. Default is 4096.
"""
set_max_groups_per_dispatch!(n::Integer) = (_max_groups_per_dispatch[] = Int(n))

"""
    set_auto_flush_threshold!(n::Integer)

Set the maximum number of dispatches before an automatic `vk_flush!()`.
Set to 0 to disable. Default is 16 (conservative for NVIDIA mobile GPUs).
"""
set_auto_flush_threshold!(n::Integer) = (_auto_flush_threshold[] = Int(n))

# ── Batch lifecycle ──

"""
    ensure_active_batch!(ctx) -> CommandBatch

Get the active batch, allocating one from the free pool if needed.
Begins command buffer recording if not already started.
"""
function ensure_active_batch!(ctx::VkContext)
    batch = ctx.active_batch
    if batch !== nothing
        if !batch.recording
            unwrap(Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
                flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            )))
            batch.recording = true
        end
        return batch
    end

    # Pop from free pool or allocate new
    if !isempty(ctx.free_batches)
        batch = pop!(ctx.free_batches)
    else
        batch = allocate_batch(ctx)
    end

    ctx.active_batch = batch

    # Begin recording
    unwrap(Vulkan.begin_command_buffer(batch.cmd_buf, Vulkan.CommandBufferBeginInfo(
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )))
    batch.recording = true
    return batch
end

"""Allocate a new CommandBatch (new command buffer + fence from the pool)."""
function allocate_batch(ctx::VkContext)
    dev = ctx.device
    alloc_info = Vulkan.CommandBufferAllocateInfo(
        ctx.cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1
    )
    cmd_bufs = unwrap(Vulkan.allocate_command_buffers(dev, alloc_info))
    fence = Vulkan.Fence(dev)
    return CommandBatch(cmd_bufs[1], fence, false, 0, false, Any[], String[])
end

"""Reclaim a completed batch: reset fence, clear data refs, return to free pool."""
function reclaim_batch!(ctx::VkContext, batch::CommandBatch)
    batch.recording = false
    batch.dispatch_count = 0
    batch.last_was_rt = false
    empty!(batch.data_refs)
    empty!(batch.dispatch_log)
    push!(ctx.free_batches, batch)

    # When ALL in-flight batches are done, safe to reset pools and flush deferred frees
    if isempty(ctx.in_flight)
        flush_deferred_frees!()
        reset_arg_buffer_pool!()
        reset_indirect_buffer_pool!()
    end
end

function _maybe_auto_flush!()
    threshold = _auto_flush_threshold[]
    threshold <= 0 && return
    ctx = vk_context()
    batch = ctx.active_batch
    batch === nothing && return
    if batch.recording && batch.dispatch_count >= threshold
        vk_flush!()
    end
end

# ── Dispatch Recording ──

"""
    vk_dispatch!(pipeline::LavaComputePipeline, push_data::Vector{UInt8},
                 groups::NTuple{3, Integer})

Record a compute dispatch into the batched command buffer.
Call `vk_flush!()` to submit and wait for completion.
"""
function vk_dispatch!(pipeline::LavaComputePipeline, push_data::Vector{UInt8},
                      groups::NTuple{3, Integer})
    max_groups = _max_groups_per_dispatch[]
    gx, gy, gz = Int(groups[1]), Int(groups[2]), Int(groups[3])

    # Split large X-dimension dispatches to avoid GPU watchdog timeout (Xid 109).
    # Uses vkCmdDispatchBase (Vulkan 1.1) which offsets GlobalInvocationID automatically.
    if max_groups > 0 && gx > max_groups && gy == 1 && gz == 1
        base = 0
        while base < gx
            chunk = min(max_groups, gx - base)
            _vk_dispatch_base!(pipeline, push_data, base, 0, 0, chunk, 1, 1)
            base += chunk
            # Flush after each chunk to prevent TDR
            vk_flush!()
        end
        return
    end

    _vk_dispatch_base!(pipeline, push_data, 0, 0, 0, gx, gy, gz)
end

"""Record a single compute dispatch with optional base group offset."""
function _vk_dispatch_base!(pipeline::LavaComputePipeline, push_data::Vector{UInt8},
                             base_x::Int, base_y::Int, base_z::Int,
                             gx::Int, gy::Int, gz::Int)
    # NOTE: _maybe_auto_flush!() is called by callers BEFORE get_arg_buffer(),
    # not here. Flushing here would reset the arg buffer pool after the caller
    # already allocated an arg buffer, causing the next dispatch to overwrite
    # a still-pending dispatch's arg buffer data.
    ctx = vk_context()
    batch = ensure_active_batch!(ctx)
    cmd = batch.cmd_buf

    # Memory barrier between dispatches (write→read synchronization)
    if batch.dispatch_count > 0
        src_stage = batch.last_was_rt ?
            Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT
        barrier = Vulkan.MemoryBarrier(
            C_NULL,
            Vulkan.ACCESS_SHADER_WRITE_BIT,
            Vulkan.ACCESS_SHADER_READ_BIT | Vulkan.ACCESS_SHADER_WRITE_BIT
        )
        Vulkan.cmd_pipeline_barrier(
            cmd, [barrier], [], [];
            src_stage_mask=src_stage,
            dst_stage_mask=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT
        )
    end

    # Bind pipeline
    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)

    # Push constants (BDA pointer)
    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(
                cmd, pipeline.pipeline_layout,
                Vulkan.SHADER_STAGE_COMPUTE_BIT,
                UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data))
            )
        end
    end

    # Dispatch with base offset (Vulkan 1.1 — offsets GlobalInvocationID)
    if base_x == 0 && base_y == 0 && base_z == 0
        Vulkan.cmd_dispatch(cmd, UInt32(gx), UInt32(gy), UInt32(gz))
    else
        Vulkan.cmd_dispatch_base(cmd,
            UInt32(base_x), UInt32(base_y), UInt32(base_z),
            UInt32(gx), UInt32(gy), UInt32(gz))
    end
    batch.dispatch_count += 1
    batch.last_was_rt = false
    Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, 1)
    info = _last_dispatch_info[]  # set by lava_launch!/ka_backend before calling vk_dispatch!
    _log_dispatch!("$(TOTAL_DISPATCH_COUNTER[]) $info base=($base_x,$base_y,$base_z) g=($gx,$gy,$gz)")
end

# ── Flush ──

"""
    vk_flush!()

Submit the active command batch and wait for GPU completion.
"""
function vk_flush!()
    _device_lost[] && error("Cannot flush: Vulkan device lost. Restart Julia session.")
    ctx = vk_context()
    batch = ctx.active_batch
    batch === nothing && return
    !batch.recording && return
    Threads.atomic_add!(FLUSH_COUNTER, 1)

    dev = ctx.device

    # Helper to reset batch state on error so subsequent operations don't
    # try to reuse a command buffer in an invalid state.
    function reset_batch_on_error!()
        batch.recording = false
        batch.dispatch_count = 0
        batch.last_was_rt = false
        empty!(batch.data_refs)
        ctx.active_batch = nothing
        push!(ctx.free_batches, batch)
    end

    unwrap(Vulkan.end_command_buffer(batch.cmd_buf))

    # Save dispatch info before any reset (for error reporting)
    saved_dispatch_count = batch.dispatch_count
    saved_last_was_rt = batch.last_was_rt
    _prev_dispatch_info[] = _last_dispatch_info[]

    submit_info = Vulkan.SubmitInfo([], [], [batch.cmd_buf], [])
    submit_result = Vulkan.queue_submit(ctx.queue, [submit_info]; fence=batch.fence)
    if iserror(submit_result)
        _device_lost[] = true
        reset_batch_on_error!()
        _throw_with_validation_context("vkQueueSubmit", submit_result,
            saved_dispatch_count, saved_last_was_rt)
    end

    fence_result = Vulkan.wait_for_fences(dev, [batch.fence], true, typemax(UInt64))
    if iserror(fence_result)
        _device_lost[] = true
        reset_batch_on_error!()
        _throw_with_validation_context("vkWaitForFences", fence_result,
            saved_dispatch_count, saved_last_was_rt)
    end
    unwrap(Vulkan.reset_fences(dev, [batch.fence]))

    # Detach from context and reclaim — batch state reset happens inside reclaim_batch!
    ctx.active_batch = nothing
    reclaim_batch!(ctx, batch)

    # Check GPU memory pressure after flush — this is a natural boundary
    # where previous render's objects may be garbage-collectible.
    maybe_collect()
end

# ── Indirect Dispatch ──

"""
    vk_dispatch_indirect!(pipeline::LavaComputePipeline, push_data::Vector{UInt8},
                          indirect_buf::VkManagedBuffer, indirect_offset::Integer=0)

Record an indirect compute dispatch into the batched command buffer.
The `indirect_buf` must contain a VkDispatchIndirectCommand at the given offset
(3×UInt32: groupCountX, groupCountY, groupCountZ), written by a previous GPU kernel.
Call `vk_flush!()` to submit and wait for completion.
"""
function vk_dispatch_indirect!(pipeline::LavaComputePipeline, push_data::Vector{UInt8},
                               indirect_buf, indirect_offset::Integer=0)
    # NOTE: _maybe_auto_flush!() called by callers before get_arg_buffer()
    ctx = vk_context()
    batch = ensure_active_batch!(ctx)
    cmd = batch.cmd_buf

    # Memory barrier between dispatches (write→read synchronization)
    # Must include ACCESS_INDIRECT_COMMAND_READ_BIT for vkCmdDispatchIndirect to
    # correctly read group counts written by the prepare-indirect kernel.
    if batch.dispatch_count > 0
        src_stage = batch.last_was_rt ?
            Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT
        barrier = Vulkan.MemoryBarrier(
            C_NULL,
            Vulkan.ACCESS_SHADER_WRITE_BIT,
            Vulkan.ACCESS_SHADER_READ_BIT | Vulkan.ACCESS_SHADER_WRITE_BIT | Vulkan.ACCESS_INDIRECT_COMMAND_READ_BIT
        )
        Vulkan.cmd_pipeline_barrier(
            cmd, [barrier], [], [];
            src_stage_mask=src_stage,
            dst_stage_mask=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT | Vulkan.PIPELINE_STAGE_DRAW_INDIRECT_BIT
        )
    end

    # Bind pipeline
    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)

    # Push constants (BDA pointer)
    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(
                cmd, pipeline.pipeline_layout,
                Vulkan.SHADER_STAGE_COMPUTE_BIT,
                UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data))
            )
        end
    end

    # Indirect dispatch — reads group counts from GPU buffer
    vk_buf = indirect_buf isa Vulkan.Buffer ? indirect_buf : indirect_buf.buffer
    Vulkan.cmd_dispatch_indirect(cmd, vk_buf, UInt64(indirect_offset))
    batch.dispatch_count += 1
    batch.last_was_rt = false
    Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, 1)
    # Preserve caller's _last_dispatch_info (set before calling this function)
    # and log the indirect dispatch with the kernel name
    info = _last_dispatch_info[]
    _log_dispatch!("$(TOTAL_DISPATCH_COUNTER[]) $info (indirect)")
end

# ── Data Lifetime ──

"""
    keep_data_alive!(refs)

Keep Julia objects alive until the next `vk_flush!()` completes.
Prevents GC from freeing LavaArray backing buffers while the GPU is still
reading from them via BDA addresses in the recorded command buffer.

Typically called with the kernel args tuple before dispatch recording.
"""
function keep_data_alive!(refs)
    ctx = vk_context()
    batch = ctx.active_batch
    if batch !== nothing
        push!(batch.data_refs, refs)
    end
end

# ── Error Reporting ──

"""
    _throw_with_validation_context(call_name, err_result)

Throw a LavaVulkanError enriched with recent validation layer messages.
Called when vkQueueSubmit or vkWaitForFences returns an error (typically DEVICE_LOST).
"""
function _throw_with_validation_context(call_name::String, err_result,
        dispatch_count::Int=0, last_was_rt::Bool=false)
    vk_err = unwrap_error(err_result)
    msgs = get_validation_messages()
    validation_detail = if isempty(msgs)
        "No validation messages captured. Install vulkan-validationlayers for GPU error diagnostics."
    else
        n = min(length(msgs), 10)
        "Last $n validation message(s):\n" * join(["  [$i] $(msgs[end-n+i])" for i in 1:n], "\n")
    end

    # Include dispatch log for crash debugging
    dispatch_detail = if isempty(_dispatch_log)
        "No dispatches logged."
    else
        "Recent dispatch log (last $(length(_dispatch_log))):\n" *
        join(["  $d" for d in _dispatch_log], "\n")
    end

    total = TOTAL_DISPATCH_COUNTER[]
    prev_info = _prev_dispatch_info[]
    curr_info = _last_dispatch_info[]
    throw(LavaError(
        call_name,
        """$vk_err after $dispatch_count dispatches in batch ($total total, last_was_rt=$last_was_rt)
Crashed batch dispatch: $prev_info
Triggered by recording: $curr_info
$validation_detail
$dispatch_detail""",
        "DEVICE_LOST usually means invalid SPIR-V, out-of-bounds BDA access, or GPU timeout (Xid 109). Check dispatch log above for the crashing kernel."
    ))
end
