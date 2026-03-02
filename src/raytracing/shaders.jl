# High-level Ray Tracing Pipeline API for Lava.jl
#
# Provides a Julia-function-based API:
#   rt = RayTracingPipeline(raygen=f, closest_hit=g, miss=h)
#   trace_rays!(rt, tlas, args...; width=W, height=H)
#
# Compilation is lazy — shaders are compiled on first use and cached.

"""
    RayTracingPipeline

A ray tracing pipeline defined by Julia functions for each shader stage.
Shaders are compiled lazily on first `trace_rays!` call and cached.

# Constructor
    RayTracingPipeline(; raygen, closest_hit, miss, payload_type=:f32)

# Example
```julia
function my_raygen(output::Ptr{Float32})
    lid = Lava.lava_rt_launch_id_x()
    Lava._lava_rt_payload_store_f32(-1f0)
    Lava._lava_rt_trace_ray(...)
    t = Lava._lava_rt_payload_load_f32()
    unsafe_store!(output, t, lid + 1)
end
function my_chit()
    Lava._lava_rt_payload_store_f32(Lava.lava_rt_ray_tmax())
end
function my_miss()
    Lava._lava_rt_payload_store_f32(-1f0)
end

rt = RayTracingPipeline(raygen=my_raygen, closest_hit=my_chit, miss=my_miss)
trace_rays!(rt, tlas, output_buf; width=1920, height=1080)
```
"""
mutable struct RayTracingPipeline
    # User-provided Julia functions
    raygen_func::Any
    closesthit_func::Any
    miss_func::Any
    payload_type::Symbol
    # Compiled state (lazy)
    _compiled::Union{Nothing, NamedTuple}
    _pipeline_cache::Dict{UInt64, Tuple{LavaRTPipeline, LavaRTShader}}
end

function RayTracingPipeline(; raygen, closest_hit, miss, payload_type::Symbol=:f32)
    RayTracingPipeline(raygen, closest_hit, miss, payload_type, nothing,
                        Dict{UInt64, Tuple{LavaRTPipeline, LavaRTShader}}())
end

"""
    trace_rays!(pipeline, tlas, args...; width, height, depth=1)

Dispatch a ray tracing pipeline. The `args` are passed to the raygen shader
via the BDA argument buffer (same as compute kernel arguments).

- `tlas`: A `LavaTLAS` acceleration structure
- `args`: Arguments passed to the raygen function (buffers, scalars, structs)
- `width`, `height`, `depth`: Dispatch dimensions (number of rays per dimension)
"""
function trace_rays!(pipeline::RayTracingPipeline, tlas::LavaTLAS, args...;
                     width::Integer, height::Integer, depth::Integer=1)
    # Build type tuple from arguments (VkManagedBuffer → Ptr{UInt8})
    tt = Tuple{map(_rt_arg_llvm_type, args)...}
    cache_key = hash((tt,))

    # Get or compile pipeline for this argument signature
    cached = get(pipeline._pipeline_cache, cache_key, nothing)
    if cached === nothing
        cached = _compile_rt_pipeline(pipeline, tt)
        pipeline._pipeline_cache[cache_key] = cached
    end
    vk_pipeline, raygen_compiled = cached

    # Pack arguments into BDA arg buffer
    bda_args = _rt_args_to_bda_filtered((pipeline.raygen_func, args...))

    inline_extra = sum(arg isa InlineStructArg ? ((length(arg.bytes) + 7) & ~7) : 0
                       for arg in bda_args; init=0)
    total_size = raygen_compiled.push_info.arg_buffer_size + inline_extra

    # Allocate arg buffer
    arg_buf = vk_alloc(total_size)
    arg_data = pack_kernel_args_inline(bda_args, raygen_compiled.push_info.arg_layout,
                                        raygen_compiled.push_info.arg_buffer_size,
                                        arg_buf.address)
    upload!(arg_buf, arg_data)

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)

    # Dispatch
    rt_dispatch!(vk_pipeline, tlas, push_data, width, height; depth=depth)
end

# ── Internal: Compile RT pipeline ──

function _compile_rt_pipeline(pipeline::RayTracingPipeline, raygen_tt)
    pt = pipeline.payload_type

    # Compile raygen
    raygen_compiled = lava_compile_rt_shader(pipeline.raygen_func, raygen_tt;
        stage=:raygen, push_constant_size=8, payload_type=pt, validate=true)

    # Compile closesthit
    chit_tt = Tuple{}
    chit_compiled = lava_compile_rt_shader(pipeline.closesthit_func, chit_tt;
        stage=:closesthit, push_constant_size=0, payload_type=pt, validate=true)

    # Compile miss
    miss_tt = Tuple{}
    miss_compiled = lava_compile_rt_shader(pipeline.miss_func, miss_tt;
        stage=:miss, push_constant_size=0, payload_type=pt, validate=true)

    # Create Vulkan RT pipeline from compiled SPIR-V
    vk_pipeline = create_rt_pipeline(
        raygen_compiled.spirv_bytes,
        miss_compiled.spirv_bytes,
        chit_compiled.spirv_bytes;
        push_constant_size=8)

    return (vk_pipeline, raygen_compiled)
end

# ── RT Argument Type Mapping ──

# VkManagedBuffer passes its BDA as a Ptr
_rt_arg_llvm_type(::VkManagedBuffer) = Ptr{UInt8}
_rt_arg_llvm_type(buf::LavaBuffer{T}) where T = Ptr{T}
_rt_arg_llvm_type(a::LavaArray{T}) where T = Ptr{T}
_rt_arg_llvm_type(x) = typeof(x)

_rt_arg_to_bda(buf::VkManagedBuffer) = buf.address
_rt_arg_to_bda(buf::LavaBuffer) = buf.buf.address
_rt_arg_to_bda(a::LavaArray) = bda_address(a)
function _rt_arg_to_bda(x)
    T = typeof(x)
    if isbitstype(T) && !isprimitivetype(T)
        data = Vector{UInt8}(undef, sizeof(T))
        unsafe_store!(Ptr{T}(pointer(data)), x)
        return InlineStructArg(data)
    end
    return x
end

function _rt_args_to_bda_filtered(args::Tuple)
    result = Any[]
    for x in args
        T = typeof(x)
        if GPUCompiler.isghosttype(T) || Core.Compiler.isconstType(T)
            continue
        end
        push!(result, _rt_arg_to_bda(x))
    end
    return tuple(result...)
end
