# Ray Tracing

Lava exposes `VK_KHR_ray_tracing_pipeline` with all stages — raygen, closest-hit, any-hit, miss, intersection — implemented as Julia functions. The TLAS and BLAS lifecycle is managed by `HWTLAS`, which is the recommended entry point for production code.

!!! note "API stability"
    The ray-tracing surface area is functional and exercised by Hikari, but the public API is still being refined. Expect minor breaking changes between 0.x releases.

## HWTLAS — recommended high-level entry

`HWTLAS` owns the Vulkan TLAS + BLAS lifecycle, supports incremental geometry updates via `push!` / `delete!` / `update_transform!`, and implements the `Raycore.AbstractAccel` contract so the same client code works against software and hardware backends.

```julia
using Lava, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore: RTRay, RTHitResult

hwtlas = HWTLAS(LavaBackend())

# Add a mesh instance at the identity transform
push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))

# After staging changes, sync to the device
Raycore.sync!(hwtlas)

# Build a ray buffer and a hit-result buffer
rays = LavaArray([RTRay(0, 0, 5,    # origin
                        0,
                        0, 0, -1,   # direction
                        1f3)])      # t_max
hits = LavaArray(fill(RTHitResult(0,0,0,0,0,0,0,0), 1))

trace_closest_hits!(hits, rays, hwtlas.hw_accel, 1)
```

`HWTLAS` exposes `tlas.hw_accel::HardwareAccel`, which is the lower-level handle for direct pipeline/SBT control if you need it. Most users do not.

## Incremental updates

Modifying geometry between frames does not rebuild the whole acceleration structure — only the dirty BLAS and a TLAS refit:

```julia
# Swap one instance's mesh (e.g. an animated character)
delete!(hwtlas, instance_id)
push!(hwtlas, new_mesh, transform; instance_id=instance_id)

# Move an instance without rebuilding its BLAS
update_transform!(hwtlas, instance_id, new_transform)

Raycore.sync!(hwtlas)
```

`sync!` is nonblocking: it submits the build commands on the dedicated queue and returns. The next `trace_closest_hits!` synchronises automatically before tracing.

## Writing RT shaders

Julia-side RT shaders look like regular kernels but use the RT intrinsics from `Lava.rt_intrinsics`. Pipelines are constructed by name:

```julia
function my_raygen(image, accel, camera)
    px, py = lava_launch_id_2d()
    ray = primary_ray(camera, px, py)
    payload = lava_rt_trace_ray(accel, ray)
    image[px, py] = payload.color
end

function my_closesthit()
    t = lava_ray_query_get_t(true)
    prim = lava_ray_query_get_primitive_index(true)
    lava_rt_payload_store_f32(t, 0)
end

function my_miss()
    lava_rt_payload_store_f32(0, 0)
end

pipeline = RayTracingPipeline(
    raygen       = my_raygen,
    closest_hit  = my_closesthit,
    miss         = my_miss,
)
```

`RayTracingPipeline` builds the SBT, manages the pipeline cache, and dispatches via `vkCmdTraceRaysKHR`.

## Inline ray queries

For simpler workloads — most volumetric integrators, AO, shadow rays — `VK_KHR_ray_query` is often a better fit than a full RT pipeline. Inline ray queries run inside compute shaders and have lower setup overhead:

```julia
@kernel function shadow_kernel!(visibility, origins, dirs, accel)
    i = @index(Global)
    o = origins[i]; d = dirs[i]
    lava_ray_query_init(accel, UInt32(0), UInt32(0xFF),
                        o[1], o[2], o[3], 0f0,
                        d[1], d[2], d[3], 1f3)
    while lava_ray_query_proceed()
        # opaque triangles auto-commit
    end
    visibility[i] = lava_ray_query_get_type(true) == UInt32(1) ? 0f0 : 1f0
end
```

This is the path Hikari uses for its shadow rays.

## Hikari and Raycore integration

`HWTLAS` implements `Raycore.AbstractAccel`, so any renderer that talks to Raycore (notably [Hikari](https://github.com/SimonDanisch/Hikari.jl)) gets hardware ray tracing for free by passing `hw_accel=true` to its integrators. The same scene file runs through Hikari's software BVH for ground-truth correctness checks.
