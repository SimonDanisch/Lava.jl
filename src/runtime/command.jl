# Command buffer recording and dispatch for Lava.jl
#
# Uses pre-allocated command buffer from VkContext.
# Supports batched dispatches with automatic memory barriers.

# Track in-flight argument buffers to prevent GC during GPU execution
const _inflight_arg_bufs = VkManagedBuffer[]

# Flush counter for benchmarking (atomic for thread safety)
const _flush_counter = Threads.Atomic{Int}(0)

"""
    vk_dispatch!(pipeline::LavaComputePipeline, push_data::Vector{UInt8},
                 groups::NTuple{3, Integer})

Record a compute dispatch into the batched command buffer.
Call `vk_flush!()` to submit and wait for completion.
"""
function vk_dispatch!(pipeline::LavaComputePipeline, push_data::Vector{UInt8},
                      groups::NTuple{3, Integer})
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

    # Dispatch
    Vulkan.cmd_dispatch(cmd, UInt32(groups[1]), UInt32(groups[2]), UInt32(groups[3]))
    ctx.dispatch_count += 1
    ctx.last_was_rt = false
end

"""
    vk_flush!()

Submit the batched command buffer and wait for GPU completion.
"""
function vk_flush!()
    ctx = vk_context()
    !ctx.recording && return
    Threads.atomic_add!(_flush_counter, 1)

    dev = ctx.device

    unwrap(Vulkan.end_command_buffer(ctx.cmd_buf))

    submit_info = Vulkan.SubmitInfo([], [], [ctx.cmd_buf], [])
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=ctx.fence))

    unwrap(Vulkan.wait_for_fences(dev, [ctx.fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [ctx.fence]))

    # Reset state
    ctx.recording = false
    ctx.dispatch_count = 0
    ctx.last_was_rt = false
    empty!(_inflight_arg_bufs)
    _reset_arg_buffer_pool!()
    _reset_indirect_buffer_pool!()
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
end

"""
    keep_alive!(buf::VkManagedBuffer)

Keep a buffer alive until the next `vk_flush!()` completes.
Prevents GC from freeing argument buffers during GPU execution.
"""
function keep_alive!(buf::VkManagedBuffer)
    push!(_inflight_arg_bufs, buf)
end
