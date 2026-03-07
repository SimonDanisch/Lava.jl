# Command buffer recording and dispatch for Lava.jl
#
# Uses pre-allocated command buffer from VkContext.
# Supports batched dispatches with automatic memory barriers.

# Track in-flight data references to prevent GC from freeing GPU buffers
# between dispatch recording and vk_flush!().
#
# When KA.argconvert converts LavaArray → LavaDeviceArray(Ptr{T}(...), dims),
# the raw Ptr holds no reference to the backing VkManagedBuffer. If the caller
# drops all LavaArray references before vk_flush!(), GC can call vk_free!()
# on the backing buffer while the GPU command buffer still reads from it via BDA.
# Under heavy GC pressure (e.g. thousands of test allocations), this causes
# DEVICE_LOST (GPU page fault on freed memory).
#
# Solution: push the original args tuple here before dispatch. This keeps all
# LavaArray objects (and Broadcasted objects containing them) alive until flush.
#
# This is Layer 1 (proactive) of our GC safety. Layer 2 (structural guard) is
# DEFERRED_FREES in memory.jl — if a GC finalizer fires during recording despite
# Layer 1, the actual Vulkan destroy is deferred until after vk_flush!().
const INFLIGHT_DATA_REFS = Any[]

# Flush counter for benchmarking (atomic for thread safety)
const FLUSH_COUNTER = Threads.Atomic{Int}(0)

# Global dispatch counter for debugging (total dispatches across all flushes)
const TOTAL_DISPATCH_COUNTER = Threads.Atomic{Int}(0)

# Last dispatch info for debugging DEVICE_LOST
const _last_dispatch_info = Ref{String}("")

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

function _maybe_auto_flush!()
    threshold = _auto_flush_threshold[]
    threshold <= 0 && return
    ctx = vk_context()
    if ctx.recording && ctx.dispatch_count >= threshold
        vk_flush!()
    end
end

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
    _maybe_auto_flush!()
    ctx = vk_context()
    cmd = ctx.cmd_buf

    # Begin recording if not already
    if !ctx.recording
        unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
            flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )))
        ctx.recording = true
    end

    # Memory barrier between dispatches (write→read synchronization)
    if ctx.dispatch_count > 0
        src_stage = ctx.last_was_rt ?
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
    ctx.dispatch_count += 1
    ctx.last_was_rt = false
    Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, 1)
    _last_dispatch_info[] = "compute base=($base_x,$base_y,$base_z) groups=($gx,$gy,$gz)"
end

"""
    vk_flush!()

Submit the batched command buffer and wait for GPU completion.
"""
function vk_flush!()
    ctx = vk_context()
    !ctx.recording && return
    Threads.atomic_add!(FLUSH_COUNTER, 1)

    dev = ctx.device

    # Helper to reset state regardless of success/failure.
    # Must be called before throwing so subsequent operations don't
    # try to reuse a command buffer in an invalid state.
    function _reset_flush_state!()
        ctx.recording = false
        ctx.dispatch_count = 0
        ctx.last_was_rt = false
        empty!(INFLIGHT_DATA_REFS)
    end

    unwrap(Vulkan.end_command_buffer(ctx.cmd_buf))

    # Save dispatch info before any reset (for error reporting)
    saved_dispatch_count = ctx.dispatch_count
    saved_last_was_rt = ctx.last_was_rt

    submit_info = Vulkan.SubmitInfo([], [], [ctx.cmd_buf], [])
    submit_result = Vulkan.queue_submit(ctx.queue, [submit_info]; fence=ctx.fence)
    if iserror(submit_result)
        _reset_flush_state!()
        # Wait for device idle to recover, then reset fence
        try; Vulkan.device_wait_idle(dev); catch; end
        try; Vulkan.wait_for_fences(dev, [ctx.fence], true, UInt64(5_000_000_000)); catch; end
        try; Vulkan.reset_fences(dev, [ctx.fence]); catch; end
        _throw_with_validation_context("vkQueueSubmit", submit_result,
            saved_dispatch_count, saved_last_was_rt)
    end

    fence_result = Vulkan.wait_for_fences(dev, [ctx.fence], true, typemax(UInt64))
    if iserror(fence_result)
        _reset_flush_state!()
        # Device lost — wait for idle, then reset fence for future use
        try; Vulkan.device_wait_idle(dev); catch; end
        try; Vulkan.reset_fences(dev, [ctx.fence]); catch; end
        _throw_with_validation_context("vkWaitForFences", fence_result,
            saved_dispatch_count, saved_last_was_rt)
    end
    unwrap(Vulkan.reset_fences(dev, [ctx.fence]))

    # Reset state — recording must be set to false BEFORE flush_deferred_frees!
    # so that any GC finalizers triggered during deferred free processing don't
    # re-defer (we're idle now, safe to free immediately).
    _reset_flush_state!()

    # Destroy buffers whose GC finalizer fired during recording/execution.
    # GPU is idle (fence waited above), safe to destroy now.
    flush_deferred_frees!()
    reset_arg_buffer_pool!()
    reset_indirect_buffer_pool!()
end

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
    _maybe_auto_flush!()
    ctx = vk_context()
    cmd = ctx.cmd_buf

    # Begin recording if not already
    if !ctx.recording
        unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
            flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )))
        ctx.recording = true
    end

    # Memory barrier between dispatches (write→read synchronization)
    # Must include ACCESS_INDIRECT_COMMAND_READ_BIT for vkCmdDispatchIndirect to
    # correctly read group counts written by the prepare-indirect kernel.
    if ctx.dispatch_count > 0
        src_stage = ctx.last_was_rt ?
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
    ctx.dispatch_count += 1
    ctx.last_was_rt = false
    Threads.atomic_add!(TOTAL_DISPATCH_COUNTER, 1)
    _last_dispatch_info[] = "compute_indirect"
end

"""
    keep_data_alive!(refs)

Keep Julia objects alive until the next `vk_flush!()` completes.
Prevents GC from freeing LavaArray backing buffers while the GPU is still
reading from them via BDA addresses in the recorded command buffer.

Typically called with the kernel args tuple before dispatch recording.
"""
function keep_data_alive!(refs)
    push!(INFLIGHT_DATA_REFS, refs)
end

"""
    _throw_with_validation_context(call_name, err_result)

Throw a LavaVulkanError enriched with recent validation layer messages.
Called when vkQueueSubmit or vkWaitForFences returns an error (typically DEVICE_LOST).
"""
function _throw_with_validation_context(call_name::String, err_result,
        dispatch_count::Int=0, last_was_rt::Bool=false)
    vk_err = unwrap_error(err_result)
    msgs = get_validation_messages()
    detail = if isempty(msgs)
        "No validation messages captured. Install vulkan-validationlayers (apt install vulkan-validationlayers) for detailed GPU error diagnostics."
    else
        n = min(length(msgs), 10)
        "Last $n validation message(s):\n" * join(["  [$i] $(msgs[end-n+i])" for i in 1:n], "\n")
    end
    total = TOTAL_DISPATCH_COUNTER[]
    last_info = _last_dispatch_info[]
    throw(LavaError(
        call_name,
        "$vk_err after $dispatch_count dispatches in batch ($total total, last_was_rt=$last_was_rt)\nLast dispatch: $last_info\n$detail",
        "Check validation messages above. DEVICE_LOST usually means invalid SPIR-V, out-of-bounds BDA access, or GPU timeout."
    ))
end
