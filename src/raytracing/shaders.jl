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
    anyhit_func::Any          # nothing = no any-hit shader
    payload_type::Symbol
    # Compiled state (lazy)
    _compiled::Union{Nothing, NamedTuple}
    _pipeline_cache::Dict{UInt64, Tuple{LavaRTPipeline, LavaRTShader, Vector{Int}, Vector{Int}}}
    _cache_device_gen::UInt64  # device generation when cache was populated
end

function RayTracingPipeline(; raygen, closest_hit, miss, any_hit=nothing, payload_type::Symbol=:f32)
    RayTracingPipeline(raygen, closest_hit, miss, any_hit, payload_type, nothing,
                        Dict{UInt64, Tuple{LavaRTPipeline, LavaRTShader, Vector{Int}, Vector{Int}}}(),
                        _device_generation[])
end

"""Invalidate RT pipeline cache if the Vulkan device was reset since last use."""
function invalidate_stale_rt_cache!(pipeline::RayTracingPipeline)
    if pipeline._cache_device_gen != _device_generation[]
        empty!(pipeline._pipeline_cache)
        pipeline._compiled = nothing
        pipeline._cache_device_gen = _device_generation[]
    end
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

    # Invalidate cache if device was reset (pipelines reference old VkDevice)
    invalidate_stale_rt_cache!(pipeline)

    # Get or compile pipeline for this argument signature
    cached = get(pipeline._pipeline_cache, cache_key, nothing)
    if cached === nothing
        cached = compile_rt_pipeline(pipeline, tt)
        pipeline._pipeline_cache[cache_key] = cached
    end
    vk_pipeline, raygen_compiled, offsets, byval_sizes = cached

    # Pack args directly to mapped memory (unified path with compute)
    all_args = (pipeline.raygen_func, args...)
    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = raygen_compiled.push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(total_size)
    _pack_args_direct!(arg_buf.mapped_ptr, arg_buf.address, offsets,
                       raygen_compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    # Keep data buffer references alive until flush
    bq = vk_context().default_bq
    batch = ensure_active_batch!(bq)
    push!(batch.data_refs, args)

    rt_dispatch!(bq, vk_pipeline, tlas, arg_buf.address, width, height; depth=depth)
end

"""
    trace_rays_indirect!(pipeline, tlas, args...; n_rays::LavaArray{Int32})

Dispatch a ray tracing pipeline with the ray count read from a GPU buffer.
No CPU readback — a prepare kernel writes the indirect command, then
`cmd_trace_rays_indirect_khr` reads it from GPU memory.
"""
function trace_rays_indirect!(pipeline::RayTracingPipeline, tlas::LavaTLAS, args...;
                              n_rays::LavaArray{Int32})
    # Build type tuple from arguments
    tt = Tuple{map(_rt_arg_llvm_type, args)...}
    cache_key = hash((tt,))

    # Invalidate cache if device was reset (pipelines reference old VkDevice)
    invalidate_stale_rt_cache!(pipeline)

    # Get or compile pipeline for this argument signature
    cached = get(pipeline._pipeline_cache, cache_key, nothing)
    if cached === nothing
        cached = compile_rt_pipeline(pipeline, tt)
        pipeline._pipeline_cache[cache_key] = cached
    end
    vk_pipeline, raygen_compiled, offsets, byval_sizes = cached

    # Prepare the indirect buffer FIRST, before allocating the RT arg buffer.
    # prepare_indirect_rt_dispatch! → lava_launch! may trigger auto-flush which
    # resets slab pools. If we allocated the RT arg buffer first, the flush would
    # invalidate it (DEVICE_LOST from stale BDA).
    bq = vk_context().default_bq

    indirect_buf = get_indirect_buffer()
    prepare_indirect_rt_dispatch!(bq, indirect_buf, n_rays)

    # Now allocate and pack RT args (safe — any auto-flush already happened)
    all_args = (pipeline.raygen_func, args...)
    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = raygen_compiled.push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(total_size)
    _pack_args_direct!(arg_buf.mapped_ptr, arg_buf.address, offsets,
                       raygen_compiled.push_info.arg_buffer_size, byval_sizes, all_args)
    batch = ensure_active_batch!(bq)
    push!(batch.data_refs, args)
    push_bda = arg_buf.address

    rt_dispatch_indirect!(bq, vk_pipeline, tlas, push_bda, indirect_buf)
end

"""Prepare indirect RT dispatch buffer: writes (n_rays, 1, 1) from a GPU-resident count."""
function prepare_indirect_rt_dispatch!(bq::BatchQueue, indirect_buf::VkIndirectBuffer, n_rays_buf::LavaArray{Int32})
    lava_launch!(bq, prepare_indirect_rt_kernel,
                 Ptr{UInt32}(indirect_buf.address), n_rays_buf;
                 ndrange=1, workgroup_size=(1, 1, 1))
end

function prepare_indirect_rt_kernel(indirect::Ptr{UInt32}, n_rays_buf::Ptr{Int32})
    n = UInt32(unsafe_load(n_rays_buf, 1))
    unsafe_store!(indirect, n, 1)          # width = n_rays
    unsafe_store!(indirect, UInt32(1), 2)  # height = 1
    unsafe_store!(indirect, UInt32(1), 3)  # depth = 1
    return nothing
end

# ── Internal: Compile RT pipeline ──

function compile_rt_pipeline(pipeline::RayTracingPipeline, raygen_tt)
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

    # Compile any-hit (optional)
    anyhit_spirv = nothing
    if pipeline.anyhit_func !== nothing
        # Any-hit gets same args as raygen (shares BDA arg buffer via push constant)
        anyhit_compiled = lava_compile_rt_shader(pipeline.anyhit_func, raygen_tt;
            stage=:anyhit, push_constant_size=8, payload_type=pt, validate=true)
        anyhit_spirv = anyhit_compiled.spirv_bytes
    end

    # Create Vulkan RT pipeline from compiled SPIR-V
    vk_pipeline = create_rt_pipeline(
        raygen_compiled.spirv_bytes,
        miss_compiled.spirv_bytes,
        chit_compiled.spirv_bytes;
        anyhit_spirv=anyhit_spirv,
        push_constant_size=8)

    # Cache arg layout offsets and byval sizes for zero-alloc packing
    offsets = Int[p.first for p in raygen_compiled.push_info.arg_layout]
    byval_sizes = raygen_compiled.push_info.byval_llvm_sizes

    return (vk_pipeline, raygen_compiled, offsets, byval_sizes)
end

# ── RT Argument Type Mapping ──

# VkManagedBuffer passes its BDA as a Ptr
_rt_arg_llvm_type(::VkManagedBuffer) = Ptr{UInt8}
_rt_arg_llvm_type(buf::LavaBuffer{T}) where T = Ptr{T}
_rt_arg_llvm_type(a::LavaArray{T}) where T = Ptr{T}
_rt_arg_llvm_type(x) = typeof(x)
