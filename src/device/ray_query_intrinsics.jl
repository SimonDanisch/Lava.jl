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

# Helper: convert the Bool committed flag to the Int32 operand expected by SPIR-V.
@inline committed_to_int32(b::Bool) = b ? Int32(1) : Int32(0)

# Control-flow ops (no operands besides the implicit query var).

@inline function lava_ray_query_proceed()::Bool
    Base.llvmcall(("""
        declare i1 @lava_ray_query_proceed() #0
        define i1 @entry() #0 {
            %r = call i1 @lava_ray_query_proceed()
            ret i1 %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Bool, Tuple{})
end

@inline function lava_ray_query_confirm()::Nothing
    Base.llvmcall(("""
        declare void @lava_ray_query_confirm() #0
        define void @entry() #0 {
            call void @lava_ray_query_confirm()
            ret void
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Cvoid, Tuple{})
    return nothing
end

@inline function lava_ray_query_terminate()::Nothing
    Base.llvmcall(("""
        declare void @lava_ray_query_terminate() #0
        define void @entry() #0 {
            call void @lava_ray_query_terminate()
            ret void
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Cvoid, Tuple{})
    return nothing
end

# Typed user-facing getters: take `committed::Bool`, dispatch to scalar form.

lava_ray_query_get_type(committed::Bool)                  = lava_ray_query_get_type(committed_to_int32(committed))
lava_ray_query_get_t(committed::Bool)                     = lava_ray_query_get_t(committed_to_int32(committed))
lava_ray_query_get_instance_id(committed::Bool)           = lava_ray_query_get_instance_id(committed_to_int32(committed))
lava_ray_query_get_instance_custom_index(committed::Bool) = lava_ray_query_get_instance_custom_index(committed_to_int32(committed))
lava_ray_query_get_primitive_index(committed::Bool)       = lava_ray_query_get_primitive_index(committed_to_int32(committed))
lava_ray_query_get_barycentrics(committed::Bool)          = lava_ray_query_get_barycentrics(committed_to_int32(committed))

# Scalar `llvmcall` stubs (the SPIR-V emitter dispatches by these symbol names).

@inline function lava_ray_query_get_type(committed::Int32)::UInt32
    Base.llvmcall(("""
        declare i32 @lava_ray_query_get_type(i32) #0
        define i32 @entry(i32 %c) #0 {
            %r = call i32 @lava_ray_query_get_type(i32 %c)
            ret i32 %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), UInt32, Tuple{Int32}, committed)
end

@inline function lava_ray_query_get_t(committed::Int32)::Float32
    Base.llvmcall(("""
        declare float @lava_ray_query_get_t(i32) #0
        define float @entry(i32 %c) #0 {
            %r = call float @lava_ray_query_get_t(i32 %c)
            ret float %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Float32, Tuple{Int32}, committed)
end

@inline function lava_ray_query_get_instance_id(committed::Int32)::UInt32
    Base.llvmcall(("""
        declare i32 @lava_ray_query_get_instance_id(i32) #0
        define i32 @entry(i32 %c) #0 {
            %r = call i32 @lava_ray_query_get_instance_id(i32 %c)
            ret i32 %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), UInt32, Tuple{Int32}, committed)
end

@inline function lava_ray_query_get_instance_custom_index(committed::Int32)::UInt32
    Base.llvmcall(("""
        declare i32 @lava_ray_query_get_instance_custom_index(i32) #0
        define i32 @entry(i32 %c) #0 {
            %r = call i32 @lava_ray_query_get_instance_custom_index(i32 %c)
            ret i32 %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), UInt32, Tuple{Int32}, committed)
end

@inline function lava_ray_query_get_primitive_index(committed::Int32)::UInt32
    Base.llvmcall(("""
        declare i32 @lava_ray_query_get_primitive_index(i32) #0
        define i32 @entry(i32 %c) #0 {
            %r = call i32 @lava_ray_query_get_primitive_index(i32 %c)
            ret i32 %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), UInt32, Tuple{Int32}, committed)
end

# Barycentrics: split into two scalar llvmcalls so we don't have to return a
# struct from llvmcall. The SPIR-V emitter recognizes both names and emits
# OpRayQueryGetIntersectionBarycentricsKHR with OpCompositeExtract for each component.
@inline function lava_ray_query_get_barycentrics(committed::Int32)::NTuple{2,Float32}
    a = Base.llvmcall(("""
        declare float @lava_ray_query_get_barycentrics_x(i32) #0
        define float @entry(i32 %c) #0 {
            %r = call float @lava_ray_query_get_barycentrics_x(i32 %c)
            ret float %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Float32, Tuple{Int32}, committed)
    b = Base.llvmcall(("""
        declare float @lava_ray_query_get_barycentrics_y(i32) #0
        define float @entry(i32 %c) #0 {
            %r = call float @lava_ray_query_get_barycentrics_y(i32 %c)
            ret float %r
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Float32, Tuple{Int32}, committed)
    return (a, b)
end

# Register all new intrinsics with GPUCompiler validation.
push!(KNOWN_INTRINSICS, "lava_ray_query_proceed")
push!(KNOWN_INTRINSICS, "lava_ray_query_confirm")
push!(KNOWN_INTRINSICS, "lava_ray_query_terminate")
push!(KNOWN_INTRINSICS, "lava_ray_query_get_type")
push!(KNOWN_INTRINSICS, "lava_ray_query_get_t")
push!(KNOWN_INTRINSICS, "lava_ray_query_get_instance_id")
push!(KNOWN_INTRINSICS, "lava_ray_query_get_instance_custom_index")
push!(KNOWN_INTRINSICS, "lava_ray_query_get_primitive_index")
push!(KNOWN_INTRINSICS, "lava_ray_query_get_barycentrics_x")
push!(KNOWN_INTRINSICS, "lava_ray_query_get_barycentrics_y")
