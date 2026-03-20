# Device-side graphics shader intrinsics for Lava.jl
#
# Graphics builtins via llvmcall, following the same addrspace(7) pattern
# as compute and RT builtins. The SPIR-V emitter recognizes the internal
# LLVM function names and creates Input/Output variables with BuiltIn decorations.
#
# The public Julia API uses clean names (vertex_index, gfx_output, etc.)
# with dispatch on GeometryBasics Vec types.

using GeometryBasics: Vec2f, Vec3f, Vec4f

# ── Vertex Builtins ──

@inline function vertex_index()
    raw = Base.llvmcall(("""
        @__spirv_BuiltInVertexIndex = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInVertexIndex, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
    return Int32(raw) + Int32(1)
end

@inline function instance_index()
    raw = Base.llvmcall(("""
        @__spirv_BuiltInInstanceIndex = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInInstanceIndex, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
    return Int32(raw) + Int32(1)
end

# ── Fragment Builtins ──

@inline function frag_coord(dim::Integer=1)
    Base.llvmcall(("""
        @__spirv_BuiltInFragCoord = external addrspace(7) global <4 x float>
        define float @entry(i32 %dim) #0 {
            %vec = load <4 x float>, ptr addrspace(7) @__spirv_BuiltInFragCoord, align 16
            %val = extractelement <4 x float> %vec, i32 %dim
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, UInt32(dim - 1))
end

@inline frag_coord_x() = frag_coord(1)
@inline frag_coord_y() = frag_coord(2)
@inline frag_coord_z() = frag_coord(3)
@inline frag_coord_w() = frag_coord(4)
@inline frag_coord_xy() = Vec2f(frag_coord(1), frag_coord(2))

@inline function front_facing()
    # Return UInt8 (not Bool/i1) to match Julia's ABI convention.
    # Julia passes Bool as i8 at call sites, so returning i1 creates a type
    # mismatch that prevents LLVM's AlwaysInliner from inlining this helper.
    raw = Base.llvmcall(("""
        @__spirv_BuiltInFrontFacing = external addrspace(7) global i1
        define i8 @entry() #0 {
            %val = load i1, ptr addrspace(7) @__spirv_BuiltInFrontFacing, align 1
            %ext = zext i1 %val to i8
            ret i8 %ext
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt8, Tuple{})
    raw != 0x00
end

# ── Output Builtins (Vertex/Geometry/TessEval) ──

# Low-level: 4 scalars
@inline function set_position!(x::Float32, y::Float32, z::Float32, w::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_set_position(float, float, float, float) #0
        define void @entry(float %x, float %y, float %z, float %w) #0 {
            call void @_lava_gfx_set_position(float %x, float %y, float %z, float %w)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{Float32, Float32, Float32, Float32}, x, y, z, w)
end

# High-level: Vec4f
@inline set_position!(v::Vec4f) = set_position!(v[1], v[2], v[3], v[4])

@inline function set_point_size!(s::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_set_point_size(float) #0
        define void @entry(float %s) #0 {
            call void @_lava_gfx_set_point_size(float %s)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{Float32}, s)
end

# ── User-Defined I/O (location-based) ──
# Unified gfx_output / gfx_input with dispatch on Vec types

# Low-level output intrinsics (scalar arguments for llvmcall)
@inline function gfx_output_f32(location::UInt32, val::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_f32(i32, float) #0
        define void @entry(i32 %loc, float %val) #0 {
            call void @_lava_gfx_output_f32(i32 %loc, float %val)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32}, location, val)
end

@inline function gfx_output_vec4(location::UInt32,
        x::Float32, y::Float32, z::Float32, w::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_vec4(i32, float, float, float, float) #0
        define void @entry(i32 %loc, float %x, float %y, float %z, float %w) #0 {
            call void @_lava_gfx_output_vec4(i32 %loc, float %x, float %y, float %z, float %w)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32, Float32, Float32, Float32},
    location, x, y, z, w)
end

@inline function gfx_output_vec3(location::UInt32,
        x::Float32, y::Float32, z::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_vec3(i32, float, float, float) #0
        define void @entry(i32 %loc, float %x, float %y, float %z) #0 {
            call void @_lava_gfx_output_vec3(i32 %loc, float %x, float %y, float %z)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32, Float32, Float32},
    location, x, y, z)
end

@inline function gfx_output_vec2(location::UInt32, x::Float32, y::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_vec2(i32, float, float) #0
        define void @entry(i32 %loc, float %x, float %y) #0 {
            call void @_lava_gfx_output_vec2(i32 %loc, float %x, float %y)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32, Float32}, location, x, y)
end

# High-level dispatch: gfx_output(location, value)
@inline gfx_output(loc::Integer, v::Vec4f) = gfx_output_vec4(UInt32(loc), v[1], v[2], v[3], v[4])
@inline gfx_output(loc::Integer, v::Vec3f) = gfx_output_vec3(UInt32(loc), v[1], v[2], v[3])
@inline gfx_output(loc::Integer, v::Vec2f) = gfx_output_vec2(UInt32(loc), v[1], v[2])
@inline gfx_output(loc::Integer, v::Float32) = gfx_output_f32(UInt32(loc), v)

# Low-level input intrinsics (return one component at a time)
@inline function gfx_input_vec4(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_vec4(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_vec4(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

@inline function gfx_input_vec3(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_vec3(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_vec3(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

@inline function gfx_input_vec2(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_vec2(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_vec2(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

@inline function gfx_input_f32(location::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_f32(i32) #0
        define float @entry(i32 %loc) #0 {
            %val = call float @_lava_gfx_input_f32(i32 %loc)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, location)
end

# High-level dispatch: gfx_input(Type, location) → Vec
@inline gfx_input(::Type{Vec4f}, loc::Integer) = Vec4f(
    gfx_input_vec4(UInt32(loc), UInt32(0)),
    gfx_input_vec4(UInt32(loc), UInt32(1)),
    gfx_input_vec4(UInt32(loc), UInt32(2)),
    gfx_input_vec4(UInt32(loc), UInt32(3)),
)
@inline gfx_input(::Type{Vec3f}, loc::Integer) = Vec3f(
    gfx_input_vec3(UInt32(loc), UInt32(0)),
    gfx_input_vec3(UInt32(loc), UInt32(1)),
    gfx_input_vec3(UInt32(loc), UInt32(2)),
)
@inline gfx_input(::Type{Vec2f}, loc::Integer) = Vec2f(
    gfx_input_vec2(UInt32(loc), UInt32(0)),
    gfx_input_vec2(UInt32(loc), UInt32(1)),
)
@inline gfx_input(::Type{Float32}, loc::Integer) = gfx_input_f32(UInt32(loc))

# ── Geometry Shader ──

@inline function emit_vertex!()
    Base.llvmcall(("""
        declare void @_lava_gfx_emit_vertex() #0
        define void @entry() #0 {
            call void @_lava_gfx_emit_vertex()
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{})
end

@inline function end_primitive!()
    Base.llvmcall(("""
        declare void @_lava_gfx_end_primitive() #0
        define void @entry() #0 {
            call void @_lava_gfx_end_primitive()
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{})
end

@inline function invocation_id()
    Base.llvmcall(("""
        @__spirv_BuiltInInvocationId = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInInvocationId, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

@inline function primitive_id_in()
    Base.llvmcall(("""
        @__spirv_BuiltInPrimitiveId = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInPrimitiveId, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

# ── Tessellation ──

@inline function tess_coord(dim::Integer=1)
    Base.llvmcall(("""
        @__spirv_BuiltInTessCoord = external addrspace(7) global <3 x float>
        define float @entry(i32 %dim) #0 {
            %vec = load <3 x float>, ptr addrspace(7) @__spirv_BuiltInTessCoord, align 16
            %val = extractelement <3 x float> %vec, i32 %dim
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, UInt32(dim - 1))
end

@inline tess_coord_u() = tess_coord(1)
@inline tess_coord_v() = tess_coord(2)
@inline tess_coord_w() = tess_coord(3)
@inline tess_coord_uvw() = Vec3f(tess_coord(1), tess_coord(2), tess_coord(3))

@inline function set_tess_level_outer!(idx::UInt32, val::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_set_tess_level_outer(i32, float) #0
        define void @entry(i32 %idx, float %val) #0 {
            call void @_lava_gfx_set_tess_level_outer(i32 %idx, float %val)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32}, idx, val)
end

@inline function set_tess_level_inner!(idx::UInt32, val::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_set_tess_level_inner(i32, float) #0
        define void @entry(i32 %idx, float %val) #0 {
            call void @_lava_gfx_set_tess_level_inner(i32 %idx, float %val)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32}, idx, val)
end

# ── Texture Sampling ──

@inline function sample_texture_2d(binding::UInt32, u::Float32, v::Float32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_sample_2d(i32, float, float, i32) #0
        define float @entry(i32 %binding, float %u, float %v, i32 %comp) #0 {
            %val = call float @_lava_gfx_sample_2d(i32 %binding, float %u, float %v, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, Float32, Float32, UInt32}, binding, u, v, component)
end

# ── Register intrinsic names for GPUCompiler validation ──
# These are the LLVM IR function names (not the Julia names)
push!(known_intrinsics, "_lava_gfx_set_position")
push!(known_intrinsics, "_lava_gfx_set_point_size")
push!(known_intrinsics, "_lava_gfx_output_f32")
push!(known_intrinsics, "_lava_gfx_output_vec4")
push!(known_intrinsics, "_lava_gfx_output_vec3")
push!(known_intrinsics, "_lava_gfx_output_vec2")
push!(known_intrinsics, "_lava_gfx_input_vec4")
push!(known_intrinsics, "_lava_gfx_input_vec3")
push!(known_intrinsics, "_lava_gfx_input_vec2")
push!(known_intrinsics, "_lava_gfx_input_f32")
push!(known_intrinsics, "_lava_gfx_emit_vertex")
push!(known_intrinsics, "_lava_gfx_end_primitive")
push!(known_intrinsics, "_lava_gfx_set_tess_level_outer")
push!(known_intrinsics, "_lava_gfx_set_tess_level_inner")
push!(known_intrinsics, "_lava_gfx_sample_2d")
