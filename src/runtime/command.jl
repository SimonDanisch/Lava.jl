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
const last_dispatch_info = Ref{String}("")
const prev_dispatch_info = Ref{String}("")

# Ring buffer of last N dispatch names for crash debugging
const dispatch_log = String[]
const MAX_DISPATCH_LOG = 2000

# Toggle dispatch logging (disabled by default for zero-alloc dispatch path).
# Enable with Lava.dispatch_logging_enabled[] = true for debugging.
# On DEVICE_LOST, the error handler re-enables logging automatically.
const dispatch_logging_enabled = Ref{Bool}(false)

function log_dispatch!(info::String)
    dispatch_logging_enabled[] || return
    if length(dispatch_log) >= MAX_DISPATCH_LOG
        popfirst!(dispatch_log)
    end
    push!(dispatch_log, info)
end

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
const _vk_barrier_ref = Ref(VkMemoryBarrier(
    VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
    VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT),
    VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)))
# Function pointer for vkCmdPipelineBarrier — initialized in _init_vulkan!
const _cmd_pipeline_barrier_fptr = Ref{Ptr{Nothing}}(C_NULL)

# Pre-allocated Ref for BDA push constants (zero-alloc path).
# Used inside push_constants_bda! — set and read synchronously in a single ccall,
# so no aliasing risk from nested dispatches (unlike a shared Vector{UInt8}).
const _push_bda_ref = Ref{UInt64}(0)

# Auto-flush threshold: flush command buffer after this many dispatches.
# Set to 0 to disable (default). Use set_auto_flush_threshold!(n) to enable.
const auto_flush_threshold = Ref{Int}(0)

# Max workgroups per single dispatch — splits large dispatches with flush.
# 0 = no limit (default).
const max_groups_per_dispatch = Ref{Int}(0)

"""
    set_max_groups_per_dispatch!(n::Integer)

Set the maximum number of workgroups per single compute dispatch.
Large dispatches are split into chunks using `vkCmdDispatchBase` to avoid
NVIDIA GPU watchdog timeout (Xid 109). Set to 0 to disable.
"""
set_max_groups_per_dispatch!(n::Integer) = (max_groups_per_dispatch[] = Int(n))

"""
    set_auto_flush_threshold!(n::Integer)

Set the maximum number of dispatches before an automatic `vk_flush!()`.
Set to 0 to disable.
"""
set_auto_flush_threshold!(n::Integer) = (auto_flush_threshold[] = Int(n))

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
    data_refs = Any[]
    sizehint!(data_refs, 128)  # Pre-size for typical batch (avoids Vector growth allocs)
    return CommandBatch(cmd_bufs[1], fence, false, 0, false, data_refs, String[])
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

function maybe_auto_flush!()
    threshold = auto_flush_threshold[]
    threshold <= 0 && return
    ctx = vk_context()
    batch = ctx.active_batch
    batch === nothing && return
    if batch.recording && batch.dispatch_count >= threshold
        vk_flush!()
    end
end

# ── Recording API ──
#
# All dispatch functions (compute, indirect, RT) use `record_dispatch!` to
# handle the shared boilerplate: get batch, insert barrier, run user code,
# update bookkeeping. The `do` block contains only the dispatch-specific
# Vulkan commands.

"""
    record_dispatch!(f, ctx; dst_stage, extra_dst_access=0, is_rt=false, info="")

Record a dispatch into the active command batch. Handles:
1. Getting/creating the active batch
2. Inserting a pipeline barrier if this isn't the first dispatch
3. Calling `f(cmd)` with the command buffer
4. Updating dispatch count, counter, and debug log

Example:
    record_dispatch!(ctx; dst_stage=PIPELINE_STAGE_COMPUTE_SHADER_BIT, info="my_kernel") do cmd
        Vulkan.cmd_bind_pipeline(cmd, ...)
        Vulkan.cmd_dispatch(cmd, ...)
    end
"""
function record_dispatch!(f, ctx::VkContext;
                           dst_stage::Vulkan.PipelineStageFlag,
                           extra_dst_access::Vulkan.AccessFlag=Vulkan.AccessFlag(0),
                           is_rt::Bool=false,
                           info::String="")
    batch = ensure_active_batch!(ctx)
    cmd = batch.cmd_buf

    # Memory barrier between dispatches (write→read synchronization).
    # Uses direct ccall to avoid Vulkan.jl wrapper allocations (~1.2KB/call).
    if batch.dispatch_count > 0
        src_stage = batch.last_was_rt ?
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        dst_access = VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) | VkAccessFlags(extra_dst_access)
        _vk_barrier_ref[] = VkMemoryBarrier(
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
            VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT), dst_access)
        ccall(_cmd_pipeline_barrier_fptr[], Cvoid,
              (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
               UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
              cmd.vks,
              VkPipelineStageFlags(src_stage), VkPipelineStageFlags(dst_stage), VkDependencyFlags(0),
              UInt32(1), _vk_barrier_ref,
              UInt32(0), C_NULL,
              UInt32(0), C_NULL)
    end

    # Record the actual dispatch commands
    f(cmd)

    # Bookkeeping
    batch.dispatch_count += 1
    batch.last_was_rt = is_rt
    Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, 1)
    if dispatch_logging_enabled[]
        log_dispatch!("$(TOTAL_DISPATCH_COUNTER[]) $info")
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

Record an 8-byte BDA push constant update. Zero-alloc: uses a module-level
Ref{UInt64} that is set and consumed synchronously in a single Vulkan ccall.
"""
@inline function push_constants_bda!(cmd::Vulkan.CommandBuffer, layout::Vulkan.PipelineLayout,
                                      stage_flags, bda::UInt64)
    _push_bda_ref[] = bda
    GC.@preserve _push_bda_ref begin
        Vulkan.cmd_push_constants(cmd, layout, stage_flags,
            UInt32(0), UInt32(8),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, _push_bda_ref)))
    end
end

# ── Compute Dispatch ──

"""
    vk_dispatch!(pipeline, push_bda, groups)

Record a compute dispatch. Splits large dispatches if max_groups_per_dispatch is set.
`push_bda` is the BDA address of the argument buffer (passed as 8-byte push constant).
"""
function vk_dispatch!(pipeline::LavaComputePipeline, push_bda::UInt64,
                      groups::NTuple{3, Integer})
    limit = max_groups_per_dispatch[]
    gx, gy, gz = Int(groups[1]), Int(groups[2]), Int(groups[3])

    # Split large X-dimension dispatches if configured
    if limit > 0 && gx > limit && gy == 1 && gz == 1
        base = 0
        while base < gx
            chunk = min(limit, gx - base)
            vk_dispatch_base!(pipeline, push_bda, base, 0, 0, chunk, 1, 1)
            base += chunk
            vk_flush!()
        end
        return
    end

    vk_dispatch_base!(pipeline, push_bda, 0, 0, 0, gx, gy, gz)
end

"""Record a single compute dispatch with optional base group offset."""
function vk_dispatch_base!(pipeline::LavaComputePipeline, push_bda::UInt64,
                            base_x::Int, base_y::Int, base_z::Int,
                            gx::Int, gy::Int, gz::Int)
    # NOTE: maybe_auto_flush!() is called by callers BEFORE get_arg_buffer(),
    # not here. Flushing here would reset the arg buffer pool after the caller
    # already allocated an arg buffer.
    ctx = vk_context()
    info = last_dispatch_info[]

    dispatch_info = dispatch_logging_enabled[] ?
        "$info base=($base_x,$base_y,$base_z) g=($gx,$gy,$gz)" : ""
    record_dispatch!(ctx;
        dst_stage=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        info=dispatch_info
    ) do cmd
        Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
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
function vk_dispatch_indirect!(pipeline::LavaComputePipeline, push_bda::UInt64,
                               indirect_buf, indirect_offset::Integer=0)
    ctx = vk_context()
    dispatch_info = dispatch_logging_enabled[] ?
        "$(last_dispatch_info[]) (indirect)" : ""

    record_dispatch!(ctx;
        dst_stage=Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT | Vulkan.PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        extra_dst_access=Vulkan.ACCESS_INDIRECT_COMMAND_READ_BIT,
        info=dispatch_info
    ) do cmd
        Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        push_constants_bda!(cmd, pipeline.pipeline_layout, Vulkan.SHADER_STAGE_COMPUTE_BIT, push_bda)

        vk_buf = indirect_buf isa Vulkan.Buffer ? indirect_buf : indirect_buf.buffer
        Vulkan.cmd_dispatch_indirect(cmd, vk_buf, UInt64(indirect_offset))
    end
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
    prev_dispatch_info[] = last_dispatch_info[]

    submit_info = Vulkan.SubmitInfo([], [], [batch.cmd_buf], [])
    submit_result = Vulkan.queue_submit(ctx.queue, [submit_info]; fence=batch.fence)
    if iserror(submit_result)
        _device_lost[] = true
        reset_batch_on_error!()
        throw_with_validation_context("vkQueueSubmit", submit_result,
            saved_dispatch_count, saved_last_was_rt)
    end

    fence_result = Vulkan.wait_for_fences(dev, [batch.fence], true, typemax(UInt64))
    if iserror(fence_result)
        _device_lost[] = true
        reset_batch_on_error!()
        throw_with_validation_context("vkWaitForFences", fence_result,
            saved_dispatch_count, saved_last_was_rt)
    end
    unwrap(Vulkan.reset_fences(dev, [batch.fence]))

    # Detach from context and reclaim
    ctx.active_batch = nothing
    reclaim_batch!(ctx, batch)
end

# ── Piggybacked Download ──

"""
    _append_copy_and_flush!(ctx, src_buffer, src_offset, dst_staging, nbytes)

Append a GPU→staging buffer copy to the active command batch and flush.
Saves one fence roundtrip compared to vk_flush!() + _one_shot_copy() by
combining all dispatch commands and the copy into a single submission.

The caller must ensure `has_active_recording(ctx)` is true.
"""
function _append_copy_and_flush!(ctx::VkContext, src_buffer::Vulkan.Buffer,
                                  src_offset::Integer, dst_staging::Vulkan.Buffer,
                                  nbytes::Integer)
    batch = ctx.active_batch
    cmd = batch.cmd_buf

    # Barrier: shader writes → transfer read
    src_stage = batch.last_was_rt ?
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    _vk_barrier_ref[] = VkMemoryBarrier(
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
        VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT),
        VkAccessFlags(VK_ACCESS_TRANSFER_READ_BIT))
    ccall(_cmd_pipeline_barrier_fptr[], Cvoid,
          (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
           UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
          cmd.vks,
          VkPipelineStageFlags(src_stage),
          VkPipelineStageFlags(VK_PIPELINE_STAGE_TRANSFER_BIT),
          VkDependencyFlags(0),
          UInt32(1), _vk_barrier_ref,
          UInt32(0), C_NULL,
          UInt32(0), C_NULL)

    # Append the copy command
    region = Vulkan.BufferCopy(UInt64(src_offset), UInt64(0), UInt64(nbytes))
    Vulkan.cmd_copy_buffer(cmd, src_buffer, dst_staging, [region])

    # Flush the entire batch (dispatches + copy) in one submit
    vk_flush!()
end

# ── Data Lifetime ──

"""
    keep_data_alive!(refs)

Keep Julia objects alive until the next `vk_flush!()` completes.
Prevents GC from freeing LavaArray backing buffers while the GPU is still
reading from them via BDA addresses in the recorded command buffer.
"""
function keep_data_alive!(refs)
    ctx = vk_context()
    batch = ctx.active_batch
    if batch !== nothing
        push!(batch.data_refs, refs)
    end
end

# ── Error Reporting ──

"""Throw a LavaError enriched with recent validation layer messages and dispatch log."""
function throw_with_validation_context(call_name::String, err_result,
        dispatch_count::Int=0, last_was_rt::Bool=false)
    # Re-enable dispatch logging so the next run captures debug info
    dispatch_logging_enabled[] = true
    vk_err = unwrap_error(err_result)
    msgs = get_validation_messages()
    validation_detail = if isempty(msgs)
        "No validation messages captured. Install vulkan-validationlayers for GPU error diagnostics."
    else
        n = min(length(msgs), 10)
        "Last $n validation message(s):\n" * join(["  [$i] $(msgs[end-n+i])" for i in 1:n], "\n")
    end

    dispatch_detail = if isempty(dispatch_log)
        "No dispatches logged."
    else
        "Recent dispatch log (last $(length(dispatch_log))):\n" *
        join(["  $d" for d in dispatch_log], "\n")
    end

    total = TOTAL_DISPATCH_COUNTER[]
    prev_info = prev_dispatch_info[]
    curr_info = last_dispatch_info[]
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
