# Ray tracing pipeline creation and dispatch for Lava.jl
#
# Creates VkRayTracingPipelineKHR from SPIR-V shader modules.
# Manages shader binding table (SBT) layout and dispatch.

const VK_SHADER_UNUSED_KHR = ~UInt32(0)  # 0xFFFFFFFF

"""
    LavaRTPipeline

A compiled ray tracing pipeline with shader binding table.
"""
struct LavaRTPipeline
    pipeline::Vulkan.Pipeline
    pipeline_layout::Vulkan.PipelineLayout
    descriptor_set_layout::Vulkan.DescriptorSetLayout
    # Shader modules (kept alive)
    shader_modules::Vector{Vulkan.ShaderModule}
    # SBT
    sbt_buffer::VkManagedBuffer
    raygen_region::Vulkan.StridedDeviceAddressRegionKHR
    miss_region::Vulkan.StridedDeviceAddressRegionKHR
    hit_region::Vulkan.StridedDeviceAddressRegionKHR
    callable_region::Vulkan.StridedDeviceAddressRegionKHR
    # Push constant size
    push_constant_size::UInt32
end

"""
    create_rt_pipeline(raygen_spirv, miss_spirv, chit_spirv;
                       push_constant_size=8) -> LavaRTPipeline

Create a ray tracing pipeline from 3 SPIR-V binaries (raygen, miss, closest-hit).

Layout:
  - Group 0: raygen (GENERAL)
  - Group 1: miss (GENERAL)
  - Group 2: closest-hit (TRIANGLES_HIT_GROUP)
  - Descriptor set 0, binding 0: AccelerationStructure (TLAS)
  - Push constant: BDA pointer (8 bytes by default)
"""
function create_rt_pipeline(raygen_spirv::Vector{UInt8},
                            miss_spirv::Vector{UInt8},
                            chit_spirv::Vector{UInt8};
                            push_constant_size::Integer=8)
    ctx = vk_context()
    dev = ctx.device
    rt_props = ctx.rt_pipeline_properties
    rt_props === nothing && throw(LavaError(
        "RT pipeline creation",
        "Ray tracing not supported on this device",
        "Ensure VK_KHR_ray_tracing_pipeline is available"))

    # Create shader modules
    raygen_mod = _create_shader_module(dev, raygen_spirv)
    miss_mod = _create_shader_module(dev, miss_spirv)
    chit_mod = _create_shader_module(dev, chit_spirv)

    # Shader stages (index 0=raygen, 1=miss, 2=closest-hit)
    stages = [
        Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_RAYGEN_BIT_KHR, raygen_mod, "main"),
        Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_MISS_BIT_KHR, miss_mod, "main"),
        Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chit_mod, "main"),
    ]

    # Shader groups
    groups = [
        # Group 0: raygen (GENERAL, shader index 0)
        Vulkan.RayTracingShaderGroupCreateInfoKHR(
            Vulkan.RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            UInt32(0),              # general_shader = stage index 0
            VK_SHADER_UNUSED_KHR,  # closest_hit_shader
            VK_SHADER_UNUSED_KHR,  # any_hit_shader
            VK_SHADER_UNUSED_KHR,  # intersection_shader
        ),
        # Group 1: miss (GENERAL, shader index 1)
        Vulkan.RayTracingShaderGroupCreateInfoKHR(
            Vulkan.RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            UInt32(1),
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR,
            VK_SHADER_UNUSED_KHR,
        ),
        # Group 2: closest-hit (TRIANGLES_HIT_GROUP, shader index 2)
        Vulkan.RayTracingShaderGroupCreateInfoKHR(
            Vulkan.RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
            VK_SHADER_UNUSED_KHR,  # general_shader (not used for hit groups)
            UInt32(2),              # closest_hit_shader = stage index 2
            VK_SHADER_UNUSED_KHR,  # any_hit_shader
            VK_SHADER_UNUSED_KHR,  # intersection_shader
        ),
    ]

    # Descriptor set layout: binding 0 = TLAS
    ds_layout = Vulkan.DescriptorSetLayout(dev, [
        Vulkan.DescriptorSetLayoutBinding(
            UInt32(0),
            Vulkan.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            Vulkan.SHADER_STAGE_RAYGEN_BIT_KHR |
            Vulkan.SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            Vulkan.SHADER_STAGE_MISS_BIT_KHR;
            descriptor_count=1,
        ),
    ])

    # Pipeline layout: descriptor set + push constants
    push_ranges = if push_constant_size > 0
        [Vulkan.PushConstantRange(
            Vulkan.SHADER_STAGE_RAYGEN_BIT_KHR |
            Vulkan.SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            Vulkan.SHADER_STAGE_MISS_BIT_KHR,
            UInt32(0),
            UInt32(push_constant_size),
        )]
    else
        Vulkan.PushConstantRange[]
    end

    layout = Vulkan.PipelineLayout(dev, [ds_layout], push_ranges)

    # Create RT pipeline
    rt_ci = Vulkan.RayTracingPipelineCreateInfoKHR(
        stages, groups,
        UInt32(1),  # max_pipeline_ray_recursion_depth
        layout,
        Int32(-1);  # base_pipeline_index
    )

    pipelines, _ = unwrap(Vulkan.create_ray_tracing_pipelines_khr(dev, [rt_ci]))
    pipeline = pipelines[1]

    # Build SBT
    sbt_buf, raygen_region, miss_region, hit_region, callable_region =
        _build_sbt(dev, pipeline, rt_props, 3)

    return LavaRTPipeline(
        pipeline, layout, ds_layout,
        [raygen_mod, miss_mod, chit_mod],
        sbt_buf,
        raygen_region, miss_region, hit_region, callable_region,
        UInt32(push_constant_size),
    )
end

# Descriptor set cache: (pipeline ds_layout handle, tlas accel handle) → (pool, descriptor_set)
# Kept alive as long as the pipeline and TLAS objects exist (GC-safe via Vulkan.jl handles).
const _rt_desc_cache = Dict{Tuple{UInt64, UInt64}, Tuple{Vulkan.DescriptorPool, Vulkan.DescriptorSet}}()

# In-flight descriptor pools to keep alive until vk_flush!()
const _inflight_rt_desc_pools = Vulkan.DescriptorPool[]

function _get_rt_descriptor_set(pipeline::LavaRTPipeline, tlas::LavaTLAS)
    # Cache key: use raw Vulkan handle values for identity
    key = (UInt64(pipeline.descriptor_set_layout.vks),
           UInt64(tlas.accel.vks))
    cached = get(_rt_desc_cache, key, nothing)
    if cached !== nothing
        return cached[2]
    end

    dev = vk_device()
    desc_pool = Vulkan.DescriptorPool(dev, UInt32(1), [
        Vulkan.DescriptorPoolSize(
            Vulkan.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, UInt32(1)),
    ])
    desc_sets = unwrap(Vulkan.allocate_descriptor_sets(dev,
        Vulkan.DescriptorSetAllocateInfo(desc_pool, [pipeline.descriptor_set_layout])))
    desc_set = desc_sets[1]

    as_write = Vulkan.WriteDescriptorSetAccelerationStructureKHR([tlas.accel])
    write_ds = Vulkan.WriteDescriptorSet(
        desc_set, UInt32(0), UInt32(0),
        Vulkan.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        Vulkan.DescriptorImageInfo[],
        Vulkan.DescriptorBufferInfo[],
        Vulkan.BufferView[];
        descriptor_count=UInt32(1),
        next=as_write,
    )
    Vulkan.update_descriptor_sets(dev, [write_ds], [])

    _rt_desc_cache[key] = (desc_pool, desc_set)
    return desc_set
end

"""
    rt_dispatch!(pipeline::LavaRTPipeline, tlas::LavaTLAS,
                 push_data::Vector{UInt8}, width, height; depth=1)

Record an RT trace dispatch into the batched command buffer.
Call `vk_flush!()` to submit and wait for completion.
Compute→RT barriers are inserted automatically.
"""
function rt_dispatch!(pipeline::LavaRTPipeline, tlas::LavaTLAS,
                      push_data::Vector{UInt8}, width::Integer, height::Integer;
                      depth::Integer=1)
    ctx = vk_context()
    cmd = ctx.cmd_buf

    # Begin recording if not already
    if !ctx.recording
        unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
            flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )))
        ctx.recording = true
    end

    # Barrier: previous dispatches → RT
    if ctx.dispatch_count > 0
        src_stage = ctx.last_was_rt ?
            Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT
        barrier = Vulkan.MemoryBarrier(
            C_NULL,
            Vulkan.ACCESS_SHADER_WRITE_BIT,
            Vulkan.ACCESS_SHADER_READ_BIT | Vulkan.ACCESS_SHADER_WRITE_BIT |
            Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        )
        Vulkan.cmd_pipeline_barrier(
            cmd, [barrier], [], [];
            src_stage_mask=src_stage,
            dst_stage_mask=Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        )
    end

    # Get cached descriptor set
    desc_set = _get_rt_descriptor_set(pipeline, tlas)

    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipeline)
    Vulkan.cmd_bind_descriptor_sets(cmd, Vulkan.PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipeline.pipeline_layout, UInt32(0), [desc_set], UInt32[])

    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(
                cmd, pipeline.pipeline_layout,
                Vulkan.SHADER_STAGE_RAYGEN_BIT_KHR |
                Vulkan.SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                Vulkan.SHADER_STAGE_MISS_BIT_KHR,
                UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data)),
            )
        end
    end

    Vulkan.cmd_trace_rays_khr(cmd,
        pipeline.raygen_region,
        pipeline.miss_region,
        pipeline.hit_region,
        pipeline.callable_region,
        UInt32(width), UInt32(height), UInt32(depth),
    )

    ctx.dispatch_count += 1
    ctx.last_was_rt = true
    return nothing
end

"""
    rt_dispatch_indirect!(pipeline::LavaRTPipeline, tlas::LavaTLAS,
                          push_data::Vector{UInt8}, indirect_buf::VkIndirectBuffer;
                          indirect_offset::Integer=0)

Record an indirect RT trace dispatch into the batched command buffer.
The `indirect_buf` must contain a VkTraceRaysIndirectCommandKHR at the given offset
(3×UInt32: width, height, depth), written by a previous GPU kernel.
"""
function rt_dispatch_indirect!(pipeline::LavaRTPipeline, tlas::LavaTLAS,
                               push_data::Vector{UInt8}, indirect_buf;
                               indirect_offset::Integer=0)
    ctx = vk_context()
    cmd = ctx.cmd_buf

    # Begin recording if not already
    if !ctx.recording
        unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
            flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )))
        ctx.recording = true
    end

    # Barrier: previous dispatches → RT
    if ctx.dispatch_count > 0
        src_stage = ctx.last_was_rt ?
            Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT
        barrier = Vulkan.MemoryBarrier(
            C_NULL,
            Vulkan.ACCESS_SHADER_WRITE_BIT,
            Vulkan.ACCESS_SHADER_READ_BIT | Vulkan.ACCESS_SHADER_WRITE_BIT |
            Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
        )
        Vulkan.cmd_pipeline_barrier(
            cmd, [barrier], [], [];
            src_stage_mask=src_stage,
            dst_stage_mask=Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        )
    end

    desc_set = _get_rt_descriptor_set(pipeline, tlas)

    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipeline)
    Vulkan.cmd_bind_descriptor_sets(cmd, Vulkan.PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipeline.pipeline_layout, UInt32(0), [desc_set], UInt32[])

    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(
                cmd, pipeline.pipeline_layout,
                Vulkan.SHADER_STAGE_RAYGEN_BIT_KHR |
                Vulkan.SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                Vulkan.SHADER_STAGE_MISS_BIT_KHR,
                UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data)),
            )
        end
    end

    # Indirect dispatch — reads dimensions from BDA
    indirect_address = indirect_buf isa VkIndirectBuffer ? indirect_buf.address : indirect_buf.address
    Vulkan.cmd_trace_rays_indirect_khr(cmd,
        pipeline.raygen_region,
        pipeline.miss_region,
        pipeline.hit_region,
        pipeline.callable_region,
        UInt64(indirect_address + indirect_offset),
    )

    ctx.dispatch_count += 1
    ctx.last_was_rt = true
    return nothing
end

# ── Internal helpers ──

function _create_shader_module(dev, spirv_bytes::Vector{UInt8})
    @assert length(spirv_bytes) % 4 == 0 "SPIR-V must be 4-byte aligned"
    code_u32 = reinterpret(UInt32, spirv_bytes)
    return Vulkan.ShaderModule(dev, length(spirv_bytes), code_u32)
end

"""Build the shader binding table for a 3-group RT pipeline (raygen, miss, chit)."""
function _build_sbt(dev, pipeline::Vulkan.Pipeline, rt_props::RTPipelineProperties,
                    n_groups::Int)
    handle_size = rt_props.shader_group_handle_size
    base_align = rt_props.shader_group_base_alignment

    # Each group entry must be aligned to shader_group_handle_alignment
    # but stride must be multiple of shader_group_handle_alignment
    handle_size_aligned = _align_up(handle_size, rt_props.shader_group_handle_alignment)

    # Get all group handles
    total_handle_data = handle_size * n_groups
    handles = Vector{UInt8}(undef, total_handle_data)
    unwrap(Vulkan.get_ray_tracing_shader_group_handles_khr(
        dev, pipeline, UInt32(0), UInt32(n_groups), total_handle_data, Ptr{Nothing}(pointer(handles))))

    # SBT layout:
    # [raygen region | miss region | hit region]
    # Each region is base_align-aligned
    raygen_stride = _align_up(handle_size_aligned, base_align)
    miss_stride = handle_size_aligned
    hit_stride = handle_size_aligned

    raygen_size = raygen_stride  # exactly 1 raygen entry
    miss_size = _align_up(miss_stride, base_align)  # 1 miss entry, region aligned
    hit_size = _align_up(hit_stride, base_align)  # 1 hit entry, region aligned

    total_sbt_size = _align_up(raygen_size, base_align) +
                     _align_up(miss_size, base_align) +
                     _align_up(hit_size, base_align)

    sbt_buf = vk_alloc(total_sbt_size)

    # Upload SBT data via staging
    sbt_data = zeros(UInt8, total_sbt_size)
    offset = 0

    # Raygen (group 0)
    raygen_offset = offset
    copyto!(sbt_data, offset + 1, handles, 1, handle_size)
    offset = raygen_offset + _align_up(raygen_size, base_align)

    # Miss (group 1)
    miss_offset = offset
    copyto!(sbt_data, offset + 1, handles, handle_size + 1, handle_size)
    offset = miss_offset + _align_up(miss_size, base_align)

    # Hit (group 2)
    hit_offset = offset
    copyto!(sbt_data, offset + 1, handles, 2 * handle_size + 1, handle_size)

    upload!(sbt_buf, sbt_data)

    raygen_region = Vulkan.StridedDeviceAddressRegionKHR(
        sbt_buf.address + UInt64(raygen_offset),
        UInt64(raygen_stride),
        UInt64(raygen_size),
    )
    miss_region = Vulkan.StridedDeviceAddressRegionKHR(
        sbt_buf.address + UInt64(miss_offset),
        UInt64(miss_stride),
        UInt64(miss_size),
    )
    hit_region = Vulkan.StridedDeviceAddressRegionKHR(
        sbt_buf.address + UInt64(hit_offset),
        UInt64(hit_stride),
        UInt64(hit_size),
    )
    callable_region = Vulkan.StridedDeviceAddressRegionKHR(
        UInt64(0), UInt64(0), UInt64(0),
    )

    return sbt_buf, raygen_region, miss_region, hit_region, callable_region
end

_align_up(x, align) = ((x + align - 1) ÷ align) * align
