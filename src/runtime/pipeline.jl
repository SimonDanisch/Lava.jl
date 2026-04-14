# Compute pipeline creation and caching for Lava.jl
#
# Creates VkShaderModule + VkComputePipeline from validated SPIR-V binary.
# Pipelines are cached by SPIR-V content hash.
#
# On Windows, vkCreateComputePipelines runs on a dedicated thread with a 256MB
# stack to work around AMD's proprietary Vulkan driver overflowing its internal
# compiler stack when processing certain SPIR-V patterns.

const LARGE_STACK_PIPELINE = Sys.iswindows()
const LARGE_STACK_SIZE = 256 * 1024 * 1024  # 256 MB

if LARGE_STACK_PIPELINE

struct _VkCreatePipelineArgs
    device::Ptr{Cvoid}
    PIPELINE_CACHE::Ptr{Cvoid}
    create_info_count::UInt32
    _pad::UInt32
    p_create_infos::Ptr{Cvoid}
    p_allocator::Ptr{Cvoid}
    p_pipelines::Ptr{Cvoid}
    result::Int32
    _pad2::Int32
end

function vk_pipeline_thread_callback(args_ptr::Ptr{Cvoid})::UInt32
    args = unsafe_load(Ptr{_VkCreatePipelineArgs}(args_ptr))
    result = ccall((:vkCreateComputePipelines, "vulkan-1"),
        Int32,
        (Ptr{Cvoid}, Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        args.device, args.PIPELINE_CACHE, args.create_info_count,
        args.p_create_infos, args.p_allocator, args.p_pipelines)
    unsafe_store!(Ptr{Int32}(args_ptr + 48), result)
    return UInt32(0)
end

end # if LARGE_STACK_PIPELINE

const PIPELINE_THREAD_CFUNC = Ref{Ptr{Nothing}}(C_NULL)

if LARGE_STACK_PIPELINE
function init_pipeline_thread!()
    PIPELINE_THREAD_CFUNC[] = @cfunction(vk_pipeline_thread_callback, UInt32, (Ptr{Cvoid},))
end
else
init_pipeline_thread!() = nothing
end

function create_compute_pipeline_large_stack(device::Ptr{Cvoid}, create_info_ptr::Ptr{Cvoid},
                                               p_pipelines::Ptr{Cvoid})
    args = Ref(_VkCreatePipelineArgs(
        device, C_NULL, UInt32(1), UInt32(0),
        create_info_ptr, C_NULL, p_pipelines,
        Int32(0), Int32(0),
    ))
    GC.@preserve args begin
        args_ptr = Ptr{Cvoid}(pointer_from_objref(args))
        thread_id = Ref(UInt32(0))
        handle = ccall((:CreateThread, "kernel32"), Ptr{Cvoid},
            (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, UInt32, Ptr{UInt32}),
            C_NULL, LARGE_STACK_SIZE, PIPELINE_THREAD_CFUNC[], args_ptr,
            UInt32(0), thread_id)
        handle == C_NULL && error("CreateThread failed for vkCreateComputePipelines")
        # 60 second timeout — AMD's Windows driver can hang on certain SPIR-V patterns
        wait_result = ccall((:WaitForSingleObject, "kernel32"), UInt32,
            (Ptr{Cvoid}, UInt32), handle, UInt32(60_000))
        ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), handle)
        if wait_result == 0x00000102  # WAIT_TIMEOUT
            error("vkCreateComputePipelines timed out after 60s — AMD Windows driver " *
                  "shader compiler hung. This is a known driver bug with certain " *
                  "large SPIR-V modules. Try reducing kernel complexity.")
        end
        wait_result != 0 && error("WaitForSingleObject failed: $wait_result")
        return args[].result
    end
end

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

const PIPELINE_CACHE = Dict{UInt64, LavaComputePipeline}()
const PIPELINE_INSERTION_ORDER = UInt64[]
const MAX_PIPELINE_CACHE_SIZE = Ref(1024)

# Register cleanup callback for vk_reset_device!
push!(RESET_CALLBACKS, function()
    empty!(PIPELINE_CACHE)
    empty!(PIPELINE_INSERTION_ORDER)
end)

"""
    get_compute_pipeline(spirv_bytes::Vector{UInt8}, entry_name::String;
                         push_constant_size::Integer=8) -> LavaComputePipeline

Get or create a compute pipeline from SPIR-V binary.
Validates SPIR-V before creating the shader module.
"""
function get_compute_pipeline(ctx::VkContext, spirv_bytes::Vector{UInt8}, entry_name::String;
                               push_constant_size::Integer=8)
    cache_key = hash((spirv_bytes, entry_name, push_constant_size))
    cached = get(PIPELINE_CACHE, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    dev = ctx.device

    # Create shader module from SPIR-V
    @assert length(spirv_bytes) % 4 == 0 "SPIR-V binary must be 4-byte aligned"
    code_u32 = reinterpret(UInt32, spirv_bytes)
    shader_mod = Vulkan.ShaderModule(dev, length(spirv_bytes), code_u32)
    check_validation_errors!("vkCreateShaderModule (compute)")

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
    ci = Vulkan.ComputePipelineCreateInfo(stage, layout, -1;
        flags=Vulkan.PIPELINE_CREATE_DISPATCH_BASE_BIT)

    pipeline = create_compute_pipeline(dev, ci)

    result = LavaComputePipeline(shader_mod, layout, pipeline, UInt32(push_constant_size))
    PIPELINE_CACHE[cache_key] = result
    push!(PIPELINE_INSERTION_ORDER, cache_key)
    while length(PIPELINE_INSERTION_ORDER) > MAX_PIPELINE_CACHE_SIZE[]
        old_key = popfirst!(PIPELINE_INSERTION_ORDER)
        delete!(PIPELINE_CACHE, old_key)
        # No defensive pin needed: every dispatch pushes its pipeline to
        # `batch.data_refs`, so any in-flight batch that used this pipeline
        # holds a strong ref.  Once all such batches retire the ref drops
        # naturally and GC runs the destructor.
    end
    return result
end

function create_compute_pipeline(dev::Vulkan.Device, ci::Vulkan.ComputePipelineCreateInfo)
    if LARGE_STACK_PIPELINE
        ci_low = convert(Vulkan._ComputePipelineCreateInfo, ci)
        vk_ci_ref = Ref(ci_low.vks)
        pipeline_out = Ref(Ptr{Cvoid}(C_NULL))

        GC.@preserve ci_low vk_ci_ref pipeline_out begin
            vk_result = create_compute_pipeline_large_stack(
                dev.vks,
                Ptr{Cvoid}(pointer_from_objref(vk_ci_ref)),
                Ptr{Cvoid}(pointer_from_objref(pipeline_out)))
            vk_result != 0 && error("vkCreateComputePipelines failed with VkResult $vk_result")
        end

        raw_pipeline = pipeline_out[]
        parent = Vulkan.handle(dev)
        destructor = x -> Vulkan._destroy_pipeline(parent, x)
        return Vulkan.Pipeline(raw_pipeline, destructor, dev)
    else
        pipelines, _ = unwrap(Vulkan.create_compute_pipelines(dev, [ci]))
        return pipelines[1]
    end
end
