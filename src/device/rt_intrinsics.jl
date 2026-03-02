# Device-side ray tracing intrinsics for Lava.jl
#
# RT builtins via llvmcall, following the same pattern as compute builtins
# in runtime/intrinsics.jl. Each builtin produces loads from addrspace(7)
# globals with __spirv_BuiltIn* names. The SPIR-V emitter recognizes these
# and creates Input variables with BuiltIn decorations.
#
# RT-specific intrinsics (_lava_rt_trace_ray) use addrspace(8) globals
# for payload and addrspace(9) for acceleration structure descriptors.
# The RT emitter handles these specially.

# ── RT 3D Builtins (uvec3) ──
# Available in raygen shaders: LaunchIdKHR, LaunchSizeKHR
# Available in hit/miss shaders: WorldRayOriginKHR, WorldRayDirectionKHR, etc.

const _RT_BUILTIN_3D_U32 = (
    :lava_rt_launch_id       => :__spirv_BuiltInLaunchIdKHR,
    :lava_rt_launch_size     => :__spirv_BuiltInLaunchSizeKHR,
)

for (jl_name, spirv_name) in _RT_BUILTIN_3D_U32
    gvar = "@$spirv_name"
    ir = """
        $gvar = external addrspace(7) global <3 x i32>
        define i32 @entry(i32 %dim) #0 {
            %vec = load <3 x i32>, ptr addrspace(7) $gvar, align 16
            %x = extractelement <3 x i32> %vec, i32 %dim
            ret i32 %x
        }
        attributes #0 = { alwaysinline }
    """
    @eval @inline function $jl_name(dim::Integer=1)
        Base.llvmcall(($ir, "entry"), UInt32, Tuple{UInt32}, UInt32(dim - 1))
    end
    x_name = Symbol(jl_name, :_x)
    y_name = Symbol(jl_name, :_y)
    z_name = Symbol(jl_name, :_z)
    @eval @inline $x_name() = $jl_name(1)
    @eval @inline $y_name() = $jl_name(2)
    @eval @inline $z_name() = $jl_name(3)
end

# ── RT 3D Builtins (vec3 float) ──
# Available in closest-hit/miss: WorldRayOriginKHR, WorldRayDirectionKHR,
# ObjectRayOriginKHR, ObjectRayDirectionKHR

const _RT_BUILTIN_3D_F32 = (
    :lava_rt_world_ray_origin    => :__spirv_BuiltInWorldRayOriginKHR,
    :lava_rt_world_ray_direction => :__spirv_BuiltInWorldRayDirectionKHR,
    :lava_rt_object_ray_origin   => :__spirv_BuiltInObjectRayOriginKHR,
    :lava_rt_object_ray_direction => :__spirv_BuiltInObjectRayDirectionKHR,
)

for (jl_name, spirv_name) in _RT_BUILTIN_3D_F32
    gvar = "@$spirv_name"
    ir = """
        $gvar = external addrspace(7) global <3 x float>
        define float @entry(i32 %dim) #0 {
            %vec = load <3 x float>, ptr addrspace(7) $gvar, align 16
            %x = extractelement <3 x float> %vec, i32 %dim
            ret float %x
        }
        attributes #0 = { alwaysinline }
    """
    @eval @inline function $jl_name(dim::Integer=1)
        Base.llvmcall(($ir, "entry"), Float32, Tuple{UInt32}, UInt32(dim - 1))
    end
    x_name = Symbol(jl_name, :_x)
    y_name = Symbol(jl_name, :_y)
    z_name = Symbol(jl_name, :_z)
    @eval @inline $x_name() = $jl_name(1)
    @eval @inline $y_name() = $jl_name(2)
    @eval @inline $z_name() = $jl_name(3)
end

# ── RT Scalar Builtins ──
# RayTminKHR, RayTmaxKHR (float) — available in intersection/anyhit/closesthit/miss
# HitKindKHR (uint) — available in anyhit/closesthit
# InstanceCustomIndexKHR (uint) — available in intersection/anyhit/closesthit
# PrimitiveId (uint) — available in intersection/anyhit/closesthit
# InstanceId (uint) — available in intersection/anyhit/closesthit

const _RT_BUILTIN_SCALAR_F32 = (
    :lava_rt_ray_tmin  => :__spirv_BuiltInRayTminKHR,
    :lava_rt_ray_tmax  => :__spirv_BuiltInRayTmaxKHR,
)

for (jl_name, spirv_name) in _RT_BUILTIN_SCALAR_F32
    gvar = "@$spirv_name"
    ir = """
        $gvar = external addrspace(7) global float
        define float @entry() #0 {
            %val = load float, ptr addrspace(7) $gvar, align 4
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """
    @eval @inline function $jl_name()
        Base.llvmcall(($ir, "entry"), Float32, Tuple{})
    end
end

const _RT_BUILTIN_SCALAR_U32 = (
    :lava_rt_hit_kind             => :__spirv_BuiltInHitKindKHR,
    :lava_rt_instance_custom_index => :__spirv_BuiltInInstanceCustomIndexKHR,
    :lava_rt_primitive_id         => :__spirv_BuiltInPrimitiveId,
    :lava_rt_instance_id          => :__spirv_BuiltInInstanceId,
    :lava_rt_incoming_ray_flags   => :__spirv_BuiltInIncomingRayFlagsKHR,
)

for (jl_name, spirv_name) in _RT_BUILTIN_SCALAR_U32
    gvar = "@$spirv_name"
    ir = """
        $gvar = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) $gvar, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """
    @eval @inline function $jl_name()
        Base.llvmcall(($ir, "entry"), UInt32, Tuple{})
    end
end

# ── RT 4x3 Matrix Builtins ──
# ObjectToWorldKHR, WorldToObjectKHR — mat4x3 (12 floats)
# Available in intersection/anyhit/closesthit
# We expose individual element access: lava_rt_object_to_world(row, col)

const _RT_BUILTIN_MAT4X3 = (
    :lava_rt_object_to_world => :__spirv_BuiltInObjectToWorldKHR,
    :lava_rt_world_to_object => :__spirv_BuiltInWorldToObjectKHR,
)

for (jl_name, spirv_name) in _RT_BUILTIN_MAT4X3
    gvar = "@$spirv_name"
    # SPIR-V mat4x3 = 4 columns of vec3 = [12 x float]
    # Access as flat array: index = col * 3 + row
    ir = """
        $gvar = external addrspace(7) global [12 x float]
        define float @entry(i32 %idx) #0 {
            %ptr = getelementptr [12 x float], ptr addrspace(7) $gvar, i32 0, i32 %idx
            %val = load float, ptr addrspace(7) %ptr, align 4
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """
    @eval @inline function $jl_name(row::Integer, col::Integer)
        Base.llvmcall(($ir, "entry"), Float32, Tuple{UInt32}, UInt32((col - 1) * 3 + (row - 1)))
    end
end

# ── OpTraceRayKHR Intrinsic ──
# This is the main RT trace call. The emitter recognizes "_lava_rt_trace_ray"
# and emits OpTraceRayKHR.
#
# SPIR-V signature: OpTraceRayKHR %accel %ray_flags %cull_mask
#   %sbt_offset %sbt_stride %miss_index
#   %origin %tmin %direction %tmax %payload_idx
#
# The accel structure and payload are handled implicitly by the emitter
# (they're global variables, not function parameters). What we pass here
# are the ray parameters.
#
# The function is a no-op at LLVM level — the emitter replaces the call
# with OpTraceRayKHR referencing the implicit TLAS and payload variables.

@inline function _lava_rt_trace_ray(
    ray_flags::UInt32, cull_mask::UInt32,
    sbt_offset::UInt32, sbt_stride::UInt32, miss_index::UInt32,
    origin_x::Float32, origin_y::Float32, origin_z::Float32, tmin::Float32,
    dir_x::Float32, dir_y::Float32, dir_z::Float32, tmax::Float32
)
    Base.llvmcall(("""
        declare void @_lava_rt_trace_ray(
            i32, i32, i32, i32, i32,
            float, float, float, float,
            float, float, float, float) #0
        define void @entry(
            i32 %flags, i32 %mask, i32 %sbt_off, i32 %sbt_stride, i32 %miss_idx,
            float %ox, float %oy, float %oz, float %tmin,
            float %dx, float %dy, float %dz, float %tmax) #0 {
            call void @_lava_rt_trace_ray(
                i32 %flags, i32 %mask, i32 %sbt_off, i32 %sbt_stride, i32 %miss_idx,
                float %ox, float %oy, float %oz, float %tmin,
                float %dx, float %dy, float %dz, float %tmax)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, UInt32, UInt32, UInt32, UInt32,
                                Float32, Float32, Float32, Float32,
                                Float32, Float32, Float32, Float32},
    ray_flags, cull_mask, sbt_offset, sbt_stride, miss_index,
    origin_x, origin_y, origin_z, tmin,
    dir_x, dir_y, dir_z, tmax)
end

# High-level trace_ray! wrapper
@inline function lava_rt_trace_ray!(
    ray_flags::UInt32, cull_mask::UInt32,
    sbt_offset::UInt32, sbt_stride::UInt32, miss_index::UInt32,
    origin_x::Float32, origin_y::Float32, origin_z::Float32, tmin::Float32,
    dir_x::Float32, dir_y::Float32, dir_z::Float32, tmax::Float32
)
    _lava_rt_trace_ray(
        ray_flags, cull_mask, sbt_offset, sbt_stride, miss_index,
        origin_x, origin_y, origin_z, tmin,
        dir_x, dir_y, dir_z, tmax)
end

# ── Payload Load/Store Intrinsics ──
# These read/write the ray payload global variable.
# The emitter maps these to OpLoad/OpStore on the RayPayloadKHR or
# IncomingRayPayloadKHR variable.

@inline function _lava_rt_payload_store_f32(val::Float32)
    Base.llvmcall(("""
        declare void @_lava_rt_payload_store_f32(float) #0
        define void @entry(float %val) #0 {
            call void @_lava_rt_payload_store_f32(float %val)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{Float32}, val)
end

@inline function _lava_rt_payload_load_f32()
    Base.llvmcall(("""
        declare float @_lava_rt_payload_load_f32() #0
        define float @entry() #0 {
            %val = call float @_lava_rt_payload_load_f32()
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{})
end

# ── Indexed Payload Load/Store ──
# For multi-field payloads (payload_type = :f32_N).
# Store/load at a specific index in the payload array.

@inline function _lava_rt_payload_store_f32_at(val::Float32, idx::UInt32)
    Base.llvmcall(("""
        declare void @_lava_rt_payload_store_f32_at(float, i32) #0
        define void @entry(float %val, i32 %idx) #0 {
            call void @_lava_rt_payload_store_f32_at(float %val, i32 %idx)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{Float32, UInt32}, val, idx)
end

@inline function _lava_rt_payload_load_f32_at(idx::UInt32)
    Base.llvmcall(("""
        declare float @_lava_rt_payload_load_f32_at(i32) #0
        define float @entry(i32 %idx) #0 {
            %val = call float @_lava_rt_payload_load_f32_at(i32 %idx)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, idx)
end

# ── Hit Attribute (Barycentric) Load Intrinsic ──
# In closesthit shaders, gl_HitAttributeEXT contains vec2 barycentrics (u, v).
# w = 1 - u - v is the weight for vertex 0.

@inline function _lava_rt_hit_attrib_load_f32_at(idx::UInt32)
    Base.llvmcall(("""
        declare float @_lava_rt_hit_attrib_load_f32_at(i32) #0
        define float @entry(i32 %idx) #0 {
            %val = call float @_lava_rt_hit_attrib_load_f32_at(i32 %idx)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, idx)
end

# Convenience wrappers
@inline lava_rt_hit_bary_u() = _lava_rt_hit_attrib_load_f32_at(UInt32(0))
@inline lava_rt_hit_bary_v() = _lava_rt_hit_attrib_load_f32_at(UInt32(1))

# ── OpIgnoreIntersectionKHR / OpTerminateRayKHR Intrinsics ──
# These are SPIR-V block terminators — valid only in any-hit shaders.
# OpIgnoreIntersectionKHR: reject the current intersection, continue traversal.
# OpTerminateRayKHR: accept the current hit and stop traversal immediately.

@inline function _lava_rt_ignore_intersection()
    Base.llvmcall(("""
        declare void @_lava_rt_ignore_intersection() #0
        define void @entry() #0 {
            call void @_lava_rt_ignore_intersection()
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{})
end

@inline function _lava_rt_terminate_ray()
    Base.llvmcall(("""
        declare void @_lava_rt_terminate_ray() #0
        define void @entry() #0 {
            call void @_lava_rt_terminate_ray()
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{})
end

# Register RT intrinsic names for GPUCompiler validation
push!(known_intrinsics, "_lava_rt_trace_ray")
push!(known_intrinsics, "_lava_rt_payload_store_f32")
push!(known_intrinsics, "_lava_rt_payload_load_f32")
push!(known_intrinsics, "_lava_rt_payload_store_f32_at")
push!(known_intrinsics, "_lava_rt_payload_load_f32_at")
push!(known_intrinsics, "_lava_rt_hit_attrib_load_f32_at")
push!(known_intrinsics, "_lava_rt_ignore_intersection")
push!(known_intrinsics, "_lava_rt_terminate_ray")
