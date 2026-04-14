# RayCore-compatible hardware ray tracing interface for Lava.jl
#
# Provides HardwareAccel and trace_closest_hits!() as a drop-in replacement
# for Raycore's software BVH traversal closest_hit().

# Use Raycore's RTRay and RTHitResult (same memory layout, single definition)
using Raycore: RTRay, RTHitResult

"""
    HardwareAccel

Hardware-accelerated ray tracing context. Built from a Raycore-compatible TLAS.

# Fields
- `tlas::LavaTLAS` — Vulkan top-level acceleration structure
- `triangle_data::Vector` — CPU vector of all primitives (for lookup after trace)
- `blas_offsets::Vector{UInt32}` — Per-BLAS offset into triangle_data
- `rt_pipeline::RayTracingPipeline` — Pre-compiled raygen+closesthit+miss

# Usage
```julia
hw = HardwareAccel(raycore_tlas)
results = LavaArray{RTHitResult}(n_rays)
trace_closest_hits!(results, rays, hw)
```
"""
mutable struct HardwareAccel
    tlas::LavaTLAS
    triangle_data::Vector           # CPU primitives for lookup
    blas_offsets::Vector{UInt32}
    rt_pipeline::RayTracingPipeline
    # Optional any-hit pipeline (lazy — created on first use via set_anyhit_pipeline!)
    anyhit_pipeline::Union{Nothing, RayTracingPipeline}
    # BatchQueue this accel's RT dispatches run on.  Stored explicitly so
    # callers don't reach for an implicit `vk_context().default_bq`.
    bq::BatchQueue
end

"""
    HardwareAccel(tlas; bq=vk_context().default_bq) -> HardwareAccel

Build a HardwareAccel from a Raycore-compatible TLAS.
The TLAS must have `.blas_array` and `.instances` fields.
"""
function HardwareAccel(tlas; bq::BatchQueue=vk_context().default_bq)
    hw_tlas, tri_data, offsets = build_hw_accel_from_tlas(tlas)
    HardwareAccel(hw_tlas, tri_data, offsets; bq)
end

"""
    HardwareAccel(hw_tlas::LavaTLAS, triangle_data, blas_offsets; bq=vk_context().default_bq) -> HardwareAccel

Build a HardwareAccel from a pre-built Vulkan TLAS + triangle data + offsets.
"""
function HardwareAccel(hw_tlas::LavaTLAS, triangle_data, blas_offsets;
                       bq::BatchQueue=vk_context().default_bq)
    rt = RayTracingPipeline(
        raygen=hw_raygen,
        closest_hit=hw_closesthit,
        miss=hw_miss,
        payload_type=:f32_6,
    )
    HardwareAccel(hw_tlas, triangle_data, blas_offsets, rt, nothing, bq)
end

"""
    set_anyhit_pipeline!(accel::HardwareAccel, anyhit_func, raygen_func)

Create and cache an any-hit pipeline variant for this HardwareAccel.
The `anyhit_func` and `raygen_func` must have matching BDA arg signatures.
"""
function set_anyhit_pipeline!(accel::HardwareAccel, anyhit_func, raygen_func)
    accel.anyhit_pipeline = RayTracingPipeline(
        raygen=raygen_func,
        closest_hit=hw_closesthit,
        miss=hw_miss,
        any_hit=anyhit_func,
        payload_type=:f32_6,
    )
    return accel
end

"""
    trace_closest_hits!(results, rays, accel::HardwareAccel, n_rays::Integer)

Trace `n_rays` rays against the hardware acceleration structure.
Results are written to `results` buffer (one RTHitResult per ray).

`results` and `rays` can be `LavaBuffer{RTHitResult}`/`LavaBuffer{RTRay}`,
`LavaArray{RTHitResult}`/`LavaArray{RTRay}`, or any type accepted by `trace_rays!`.
"""
function trace_closest_hits!(results, rays, accel::HardwareAccel, n_rays::Integer)
    trace_rays!(accel.bq, accel.rt_pipeline, accel.tlas, rays, results;
                width=Int(n_rays), height=1)
end

"""
    trace_closest_hits_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32})

Indirect RT trace — reads ray count from GPU buffer. No CPU readback.
"""
function trace_closest_hits_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32})
    trace_rays_indirect!(accel.bq, accel.rt_pipeline, accel.tlas, rays, results; n_rays=n_rays)
end

"""
    trace_closest_hits_anyhit!(results, rays, accel, n_rays, extra_args...)

RT trace with any-hit shader. `extra_args` are passed after (rays, results) to the
raygen and any-hit functions via the shared BDA arg buffer.
"""
function trace_closest_hits_anyhit!(results, rays, accel::HardwareAccel, n_rays::Integer, extra_args...)
    pipeline = accel.anyhit_pipeline
    pipeline === nothing && error("No any-hit pipeline set. Call set_anyhit_pipeline! first.")
    trace_rays!(accel.bq, pipeline, accel.tlas, rays, results, extra_args...;
                width=Int(n_rays), height=1)
end

"""
    trace_closest_hits_anyhit_indirect!(results, rays, accel, n_rays_buf, extra_args...)

Indirect RT trace with any-hit shader — reads ray count from GPU buffer.
"""
function trace_closest_hits_anyhit_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32}, extra_args...)
    pipeline = accel.anyhit_pipeline
    pipeline === nothing && error("No any-hit pipeline set. Call set_anyhit_pipeline! first.")
    trace_rays_indirect!(accel.bq, pipeline, accel.tlas, rays, results, extra_args...; n_rays=n_rays)
end

# ── Built-in RT Shaders ──

function hw_raygen(rays::Ptr{RTRay}, results::Ptr{RTHitResult})
    lid = lava_rt_launch_id_x()

    # Read ray from input buffer
    ray = unsafe_load(rays, lid + 1)

    # Initialize payload to miss
    lava_rt_payload_store_f32_at(0f0, UInt32(0))   # hit=0
    lava_rt_payload_store_f32_at(-1f0, UInt32(1))   # t=-1

    # Trace
    lava_rt_trace_ray(
        UInt32(0),    # flags
        UInt32(0xFF), # cull mask
        UInt32(0),    # sbt offset
        UInt32(0),    # sbt stride
        UInt32(0),    # miss index
        ray.origin_x, ray.origin_y, ray.origin_z, ray.tmin,
        ray.dir_x, ray.dir_y, ray.dir_z, ray.tmax
    )

    # Read payload results
    hit  = lava_rt_payload_load_f32_at(UInt32(0))
    t    = lava_rt_payload_load_f32_at(UInt32(1))
    pid  = lava_rt_payload_load_f32_at(UInt32(2))
    ci   = lava_rt_payload_load_f32_at(UInt32(3))
    bu   = lava_rt_payload_load_f32_at(UInt32(4))
    bv   = lava_rt_payload_load_f32_at(UInt32(5))

    # Write result to output buffer
    result = RTHitResult(
        reinterpret(UInt32, hit),
        t,
        reinterpret(UInt32, pid),
        reinterpret(UInt32, ci),
        bu, bv,
        UInt32(0), UInt32(0)
    )
    unsafe_store!(results, result, lid + 1)
    return nothing
end

function hw_closesthit()
    t = lava_rt_ray_tmax()
    ci = lava_rt_instance_custom_index()
    pid = lava_rt_primitive_id()
    bu = lava_rt_hit_bary_u()
    bv = lava_rt_hit_bary_v()

    lava_rt_payload_store_f32_at(reinterpret(Float32, UInt32(1)), UInt32(0))  # hit=1
    lava_rt_payload_store_f32_at(t, UInt32(1))
    lava_rt_payload_store_f32_at(reinterpret(Float32, pid), UInt32(2))
    lava_rt_payload_store_f32_at(reinterpret(Float32, ci), UInt32(3))
    lava_rt_payload_store_f32_at(bu, UInt32(4))
    lava_rt_payload_store_f32_at(bv, UInt32(5))
    return nothing
end

function hw_miss()
    lava_rt_payload_store_f32_at(0f0, UInt32(0))   # hit=0
    lava_rt_payload_store_f32_at(-1f0, UInt32(1))   # t=-1
    lava_rt_payload_store_f32_at(0f0, UInt32(2))
    lava_rt_payload_store_f32_at(0f0, UInt32(3))
    lava_rt_payload_store_f32_at(0f0, UInt32(4))
    lava_rt_payload_store_f32_at(0f0, UInt32(5))
    return nothing
end
