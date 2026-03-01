# Compute pipeline creation and caching for Lava.jl
#
# Creates VkShaderModule + VkComputePipeline from validated SPIR-V binary.
# Pipelines are cached by SPIR-V content hash.

"""
    LavaComputePipeline

A compiled compute pipeline ready for dispatch.
"""
struct LavaComputePipeline
    shader_module::Vulkan.ShaderModule
    pipeline_layout::Vulkan.PipelineLayout
    pipeline::Vulkan.Pipeline
    push_constant_size::UInt32
end

const _pipeline_cache = Dict{UInt64, LavaComputePipeline}()

"""
    get_compute_pipeline(spirv_bytes::Vector{UInt8}, entry_name::String;
                         push_constant_size::Integer=8) -> LavaComputePipeline

Get or create a compute pipeline from SPIR-V binary.
Validates SPIR-V before creating the shader module.
"""
function get_compute_pipeline(spirv_bytes::Vector{UInt8}, entry_name::String;
                               push_constant_size::Integer=8)
    cache_key = hash((spirv_bytes, entry_name, push_constant_size))
    cached = get(_pipeline_cache, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    dev = vk_device()

    # Create shader module from SPIR-V
    # SPIR-V binary must be UInt32-aligned
    @assert length(spirv_bytes) % 4 == 0 "SPIR-V binary must be 4-byte aligned"
    code_u32 = reinterpret(UInt32, spirv_bytes)
    shader_mod = Vulkan.ShaderModule(dev, length(spirv_bytes), code_u32)

    # Pipeline layout with push constant range
    push_ranges = if push_constant_size > 0
        [Vulkan.PushConstantRange(
            Vulkan.SHADER_STAGE_COMPUTE_BIT,
            UInt32(0),
            UInt32(push_constant_size)
        )]
    else
        Vulkan.PushConstantRange[]
    end

    layout = Vulkan.PipelineLayout(
        dev,
        Vulkan.DescriptorSetLayout[],
        push_ranges
    )

    # Create compute pipeline
    stage = Vulkan.PipelineShaderStageCreateInfo(
        Vulkan.SHADER_STAGE_COMPUTE_BIT,
        shader_mod,
        entry_name
    )
    ci = Vulkan.ComputePipelineCreateInfo(stage, layout, -1)
    pipelines, _ = unwrap(Vulkan.create_compute_pipelines(dev, [ci]))
    pipeline = pipelines[1]

    result = LavaComputePipeline(shader_mod, layout, pipeline, UInt32(push_constant_size))
    _pipeline_cache[cache_key] = result
    return result
end
