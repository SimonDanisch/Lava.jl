# Lava - Julia intrinsics for VK_KHR_ray_query.
#
# Two layers:
#   1. Typed wrappers (Ray, Vec3f, etc.) -- what user kernels call.
#   2. Scalar `llvmcall` stubs -- what the SPIR-V emitter recognizes by name.
# Multiple dispatch picks the wrapper for `::Ray` and the wrapper destructures
# into scalars for the stub.

"""
    lava_ray_query_init(ray::Ray; flags=UInt32(0), mask=UInt32(0xFF))

Initialize the implicit per-function ray-query variable. Lowers to
`OpRayQueryInitializeKHR` over the bound TLAS descriptor. `ray` is a
`Raycore.Ray` (re-exported by Lava).
"""
@inline function lava_ray_query_init(ray::Ray;
                                     flags::UInt32 = UInt32(0),
                                     mask::UInt32  = UInt32(0xFF))
    o = ray.o; d = ray.d
    lava_ray_query_init(flags, mask, o[1], o[2], o[3], ray.t_min,
                        d[1], d[2], d[3], ray.t_max)
end

# Scalar method -- the symbol `lava_ray_query_init` from this `llvmcall` is
# what the SPIR-V emitter dispatches on.
@inline function lava_ray_query_init(
    flags::UInt32, mask::UInt32,
    ox::Float32, oy::Float32, oz::Float32, tmin::Float32,
    dx::Float32, dy::Float32, dz::Float32, tmax::Float32,
)
    Base.llvmcall(("""
        declare void @lava_ray_query_init(i32, i32,
            float, float, float, float,
            float, float, float, float) #0
        define void @entry(i32 %a0, i32 %a1,
                            float %a2, float %a3, float %a4, float %a5,
                            float %a6, float %a7, float %a8, float %a9) #0 {
            call void @lava_ray_query_init(i32 %a0, i32 %a1,
                float %a2, float %a3, float %a4, float %a5,
                float %a6, float %a7, float %a8, float %a9)
            ret void
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Cvoid,
        Tuple{UInt32, UInt32,
              Float32, Float32, Float32, Float32,
              Float32, Float32, Float32, Float32},
        flags, mask, ox, oy, oz, tmin, dx, dy, dz, tmax)
    return nothing
end

# Register with GPUCompiler validation so it is not rejected as unknown.
push!(KNOWN_INTRINSICS, "lava_ray_query_init")
