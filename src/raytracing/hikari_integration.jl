# Hikari Hardware RT Integration for Lava.jl
#
# This file bridges Lava's hardware RT infrastructure with Hikari's wavefront
# renderer. It is NOT included from Lava.jl (no Hikari/Raycore dependency).
# Users include it after loading both Lava, Hikari, and Raycore:
#
#   using Lava, Hikari, Raycore
#   include("path/to/hikari_integration.jl")
#
# Usage:
#   scene = Hikari.Scene(...)
#   Hikari.sync!(scene)
#   hw_accel = Lava.HardwareAccel(scene.accel)
#   # In render loop, replace vp_trace_rays! with:
#   hw_vp_trace_rays!(state, hw_accel, media_interfaces, materials)
#
# Current limitations:
# - Alpha transparency (cutout textures) is NOT supported — surfaces are treated
#   as fully opaque. Full alpha support requires an any_hit shader (future work).
# - Shadow ray tracing still uses software BVH (vp_trace_shadow_rays! unchanged).

using Base: @propagate_inbounds
using Lava: RTRay, RTHitResult, LavaArray, LavaBackend,
            lava_global_invocation_id_x, trace_closest_hits!, vk_flush!
using Adapt
using KernelAbstractions
using StaticArrays

import Raycore

# ── PrecomputedHitsAccel ──
# A "fake" acceleration structure that returns pre-computed RT results.
# Used as the `accel` argument to vp_trace_rays_kernel!, replacing the
# software BVH. The `closest_hit` method reads from the results buffer
# using the thread's GlobalInvocationID as the index.

struct PrecomputedHitsAccel{R, T, O}
    results::R       # RTHitResult array (1 per ray)
    triangles::T     # Triangle{TMetadata} array (all primitives, flat)
    offsets::O       # UInt32 array (per-BLAS offset into triangles)
end

function Adapt.adapt_structure(to, p::PrecomputedHitsAccel)
    PrecomputedHitsAccel(
        Adapt.adapt(to, p.results),
        Adapt.adapt(to, p.triangles),
        Adapt.adapt(to, p.offsets),
    )
end

# closest_hit for pre-computed results — reads from buffer by thread index.
# The thread index matches the work queue item index because foreach() launches
# one thread per queue item in order.
@propagate_inbounds function Raycore.closest_hit(accel::PrecomputedHitsAccel, ray)
    tid = lava_global_invocation_id_x() + UInt32(1)  # 1-based
    result = accel.results[tid]

    if result.hit == UInt32(0)
        # Miss — return a dummy triangle (caller checks `hit` flag first)
        dummy = accel.triangles[1]
        return (false, dummy, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0))
    end

    # Look up the original triangle from triangle_data
    tri_idx = Int(accel.offsets[result.instance_custom_index + UInt32(1)]) +
              Int(result.primitive_id) + 1
    tri = accel.triangles[tri_idx]

    # Reconstruct barycentric coordinates (w, u, v)
    w = 1f0 - result.bary_u - result.bary_v
    bary = SVector{3,Float32}(w, result.bary_u, result.bary_v)

    return (true, tri, result.t, bary)
end

# ── Ray Extraction Kernel ──
# Extracts rays from a Hikari work queue into a flat RTRay buffer.

@kernel function _extract_rays_kernel!(ray_buf, @Const(queue_items), @Const(queue_size))
    i = @index(Global)
    if i <= queue_size[1]
        work = queue_items[i]
        ray = work.ray
        ray_buf[i] = RTRay(
            Float32(ray.o[1]), Float32(ray.o[2]), Float32(ray.o[3]),
            0.001f0,  # tmin (small offset to avoid self-intersection)
            Float32(ray.d[1]), Float32(ray.d[2]), Float32(ray.d[3]),
            ray.t_max,
        )
    end
end

# ── Hardware Ray Trace Function ──

"""
    hw_vp_trace_rays!(state, hw_accel, media_interfaces, materials)

Hardware-accelerated replacement for `Hikari.VolPath.vp_trace_rays!`.

Traces rays from the current work queue using the GPU's RT hardware
(VK_KHR_ray_tracing_pipeline) instead of software BVH traversal.

# Arguments
- `state`: VolPathState (Hikari's wavefront state)
- `hw_accel`: HardwareAccel built from the scene's TLAS
- `media_interfaces`: Media interface array (passed through to kernel)
- `materials`: Material array (passed through to kernel)

# Limitations
- Alpha transparency (cutout textures) NOT supported — treated as opaque.
  The alpha testing loop in vp_trace_rays_kernel! may return incorrect results
  for transparent surfaces. Full support requires any_hit shader.
- Only primary ray tracing is accelerated; shadow rays still use software BVH.
"""
function hw_vp_trace_rays!(state, hw_accel::Lava.HardwareAccel,
                            media_interfaces, materials;
                            _tri_gpu=nothing, _off_gpu=nothing)
    # Access Hikari's internal work queue
    input_queue = state.current_ray_queue == :a ? state.ray_queue_a : state.ray_queue_b
    n_rays = Int(Array(input_queue.size)[1])
    n_rays == 0 && return nothing

    backend = KernelAbstractions.get_backend(input_queue.items)

    # Phase 1: Extract rays from work queue → flat RTRay buffer (GPU kernel)
    ray_buf = LavaArray{RTRay}(undef, n_rays)
    extract_kernel! = _extract_rays_kernel!(backend, 256)
    extract_kernel!(ray_buf, input_queue.items, input_queue.size; ndrange=n_rays)
    vk_flush!()

    # Phase 2: RT dispatch — trace all rays at once via hardware
    result_buf = LavaArray{RTHitResult}(undef, n_rays)
    trace_closest_hits!(result_buf, ray_buf, hw_accel, n_rays)

    # Phase 3: Create precomputed accel with triangle data on GPU
    # Use cached GPU arrays if provided (avoids re-uploading each frame)
    tri_gpu = _tri_gpu !== nothing ? _tri_gpu : LavaArray(hw_accel.triangle_data)
    off_gpu = _off_gpu !== nothing ? _off_gpu : LavaArray(hw_accel.blas_offsets)
    precomputed = PrecomputedHitsAccel(result_buf, tri_gpu, off_gpu)

    # Phase 4: Run original trace kernel with precomputed results
    # The kernel calls Raycore.closest_hit(precomputed, ray) which reads
    # from the pre-computed results buffer instead of traversing BVH.
    foreach(Hikari.vp_trace_rays_kernel!,
        input_queue,
        state.medium_sample_queue,
        state.escaped_queue,
        state.hit_surface_queue,
        precomputed,
        media_interfaces,
        materials,
    )
    return nothing
end

"""
    build_hardware_accel(scene_or_tlas) -> HardwareAccel

Build a HardwareAccel from a Hikari Scene or a Raycore TLAS.

For GPU-backed scenes: the TLAS stores LavaDeviceArrays (GPU pointers)
which can't be directly downloaded. You should either:
1. Build from a CPU-backed scene: `Scene(pairs; backend=CPU())`
2. Build from the CPU TLAS before GPU upload

Returns a `HardwareAccel` that can be used with `hw_vp_trace_rays!`.
"""
function build_hardware_accel(scene)
    return Lava.HardwareAccel(scene.accel)
end

"""
    build_hardware_accel(mesh_material_pairs; backend=CPU()) -> HardwareAccel

Build a HardwareAccel from mesh-material pairs by constructing a temporary
CPU-backed scene. This is the recommended way when using a GPU backend,
since it avoids the GPU↔CPU data extraction issue.

# Example
```julia
hw_accel = build_hardware_accel([(mesh1, mat1), (mesh2, mat2)])
```
"""
function build_hardware_accel(pairs::Vector{<:Tuple}; backend=KernelAbstractions.CPU())
    cpu_scene = Hikari.Scene(pairs; backend=backend)
    return Lava.HardwareAccel(cpu_scene.accel)
end

"""
    HWAccelCache

Caches GPU-side triangle data and offsets to avoid re-uploading every frame.
Create once, pass to `hw_vp_trace_rays!` via keyword arguments.

    cache = HWAccelCache(hw_accel)
    hw_vp_trace_rays!(state, hw_accel, mi, mat; _tri_gpu=cache.tri_gpu, _off_gpu=cache.off_gpu)
"""
struct HWAccelCache
    tri_gpu::LavaArray
    off_gpu::LavaArray{UInt32, 1}
end

function HWAccelCache(hw_accel::Lava.HardwareAccel)
    HWAccelCache(
        LavaArray(hw_accel.triangle_data),
        LavaArray(hw_accel.blas_offsets),
    )
end
