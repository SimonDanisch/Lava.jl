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

# ── Flat (non-interpolated) output intrinsics for geometry→fragment ──

@inline function gfx_output_flat_f32(location::UInt32, val::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_flat_f32(i32, float) #0
        define void @entry(i32 %loc, float %val) #0 {
            call void @_lava_gfx_output_flat_f32(i32 %loc, float %val)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32}, location, val)
end

@inline function gfx_output_flat_vec4(location::UInt32,
        x::Float32, y::Float32, z::Float32, w::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_flat_vec4(i32, float, float, float, float) #0
        define void @entry(i32 %loc, float %x, float %y, float %z, float %w) #0 {
            call void @_lava_gfx_output_flat_vec4(i32 %loc, float %x, float %y, float %z, float %w)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32, Float32, Float32, Float32},
    location, x, y, z, w)
end

@inline function gfx_output_flat_vec3(location::UInt32,
        x::Float32, y::Float32, z::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_flat_vec3(i32, float, float, float) #0
        define void @entry(i32 %loc, float %x, float %y, float %z) #0 {
            call void @_lava_gfx_output_flat_vec3(i32 %loc, float %x, float %y, float %z)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32, Float32, Float32},
    location, x, y, z)
end

@inline function gfx_output_flat_vec2(location::UInt32, x::Float32, y::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_flat_vec2(i32, float, float) #0
        define void @entry(i32 %loc, float %x, float %y) #0 {
            call void @_lava_gfx_output_flat_vec2(i32 %loc, float %x, float %y)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32, Float32}, location, x, y)
end

# High-level dispatch: gfx_output_flat(location, value)
@inline gfx_output_flat(loc::Integer, v::Vec4f) = gfx_output_flat_vec4(UInt32(loc), v[1], v[2], v[3], v[4])
@inline gfx_output_flat(loc::Integer, v::Vec3f) = gfx_output_flat_vec3(UInt32(loc), v[1], v[2], v[3])
@inline gfx_output_flat(loc::Integer, v::Vec2f) = gfx_output_flat_vec2(UInt32(loc), v[1], v[2])
@inline gfx_output_flat(loc::Integer, v::Float32) = gfx_output_flat_f32(UInt32(loc), v)
@inline gfx_output_flat(loc::Integer, v::Int32) = gfx_output_flat_f32(UInt32(loc), Float32(v))

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

# ── Fragment Derivatives (dFdx, dFdy) ──

@inline function dFdx(v::Float32)
    Base.llvmcall(("""
        declare float @_lava_gfx_dFdx_f32(float) #0
        define float @entry(float %v) #0 {
            %r = call float @_lava_gfx_dFdx_f32(float %v)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{Float32}, v)
end

@inline function dFdy(v::Float32)
    Base.llvmcall(("""
        declare float @_lava_gfx_dFdy_f32(float) #0
        define float @entry(float %v) #0 {
            %r = call float @_lava_gfx_dFdy_f32(float %v)
            ret float %r
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{Float32}, v)
end

@inline dFdx(v::Vec2f) = Vec2f(dFdx(v[1]), dFdx(v[2]))
@inline dFdy(v::Vec2f) = Vec2f(dFdy(v[1]), dFdy(v[2]))

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

# ── Julian Texture API ──
# tex[uv::Vec2f] → Vec4f (interpolated/filtered sampling)
# tex[idx::Vec2i] → Vec4f (texel fetch, no interpolation) — TODO

"""Texture handle for use in GPU shaders. Wraps a descriptor set binding index."""
struct GfxTexture2D
    binding::UInt32
end

@inline function Base.getindex(tex::GfxTexture2D, uv::Vec2f)
    b = tex.binding
    u = uv[1]; v = uv[2]
    return Vec4f(
        sample_texture_2d(b, u, v, UInt32(0)),
        sample_texture_2d(b, u, v, UInt32(1)),
        sample_texture_2d(b, u, v, UInt32(2)),
        sample_texture_2d(b, u, v, UInt32(3)),
    )
end

# ── Flat input intrinsics (fragment reads flat-interpolated geometry outputs) ──

@inline function gfx_input_flat_vec4(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_flat_vec4(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_flat_vec4(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

@inline function gfx_input_flat_vec2(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_flat_vec2(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_flat_vec2(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

@inline function gfx_input_flat_f32(location::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_flat_f32(i32) #0
        define float @entry(i32 %loc) #0 {
            %val = call float @_lava_gfx_input_flat_f32(i32 %loc)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, location)
end

# High-level dispatch: gfx_input_flat(Type, location) → Vec
@inline gfx_input_flat(::Type{Vec4f}, loc::Integer) = Vec4f(
    gfx_input_flat_vec4(UInt32(loc), UInt32(0)),
    gfx_input_flat_vec4(UInt32(loc), UInt32(1)),
    gfx_input_flat_vec4(UInt32(loc), UInt32(2)),
    gfx_input_flat_vec4(UInt32(loc), UInt32(3)))
@inline gfx_input_flat(::Type{Vec2f}, loc::Integer) = Vec2f(
    gfx_input_flat_vec2(UInt32(loc), UInt32(0)),
    gfx_input_flat_vec2(UInt32(loc), UInt32(1)))
@inline gfx_input_flat(::Type{Float32}, loc::Integer) = gfx_input_flat_f32(UInt32(loc))

# ── NamedTuple-based I/O ──
# Vertex shaders can return a NamedTuple with :position and varying fields.
# Fragment shaders receive a NamedTuple of interpolated varyings as first arg.
# Location assignment: non-position fields in declaration order, starting at 0.

"""Unpack a vertex shader's NamedTuple return value into set_position! + gfx_output calls."""
@generated function emit_vertex_outputs(result::NT) where NT <: NamedTuple
    exprs = Expr[]
    if hasfield(NT, :position)
        push!(exprs, :(set_position!(result.position)))
    end
    loc = 0
    for name in fieldnames(NT)
        name === :position && continue
        push!(exprs, :(gfx_output($loc, result.$name)))
        loc += 1
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

"""Build a NamedTuple of fragment inputs from gfx_input calls, matching the vertex output type."""
@generated function load_fragment_inputs(::Type{NT}) where NT <: NamedTuple
    exprs = Expr[]
    names_list = Symbol[]
    loc = 0
    for name in fieldnames(NT)
        name === :position && continue
        T = fieldtype(NT, name)
        push!(names_list, name)
        push!(exprs, :(gfx_input($T, $loc)))
        loc += 1
    end
    keys_expr = Expr(:tuple, QuoteNode.(names_list)...)
    vals_expr = Expr(:tuple, exprs...)
    return :(NamedTuple{$keys_expr}($vals_expr))
end

"""Emit fragment output: Vec4f → location 0, NamedTuple with :color → location 0."""
@inline emit_fragment_output(color::Vec4f) = gfx_output(0, color)
@generated function emit_fragment_output(result::NT) where NT <: NamedTuple
    if hasfield(NT, :color)
        return :(gfx_output(0, result.color))
    else
        error("Fragment NamedTuple must have a :color field, got fields: $(fieldnames(NT))")
    end
end

"""Infer the vertex shader's return type from its function and device-side arg types."""
function infer_vertex_output_type(@nospecialize(f), @nospecialize(tt::Type))
    rts = Base.return_types(f, tt)
    length(rts) == 1 || error("Cannot infer unique return type for vertex shader (got $(length(rts)) possibilities)")
    RT = rts[1]
    RT <: NamedTuple || return nothing  # not using NamedTuple API
    return RT
end

# ── Geometry Shader Arrayed Inputs ──
# In geometry shaders, vertex shader outputs become arrayed inputs.
# `geom_input_vec4(location, vertex_idx, component)` reads `in vec4 var[vertex_idx][component]`
# `geom_input_position(vertex_idx, component)` reads `gl_in[vertex_idx].gl_Position[component]`
# vertex_idx is 0-based (matching GLSL gl_in[0..N-1]).

@inline function geom_input_position(vertex_idx::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_geom_input_position(i32, i32) #0
        define float @entry(i32 %vidx, i32 %comp) #0 {
            %val = call float @_lava_geom_input_position(i32 %vidx, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, vertex_idx, component)
end

@inline function geom_input_vec4(location::UInt32, vertex_idx::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_geom_input_vec4(i32, i32, i32) #0
        define float @entry(i32 %loc, i32 %vidx, i32 %comp) #0 {
            %val = call float @_lava_geom_input_vec4(i32 %loc, i32 %vidx, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32, UInt32}, location, vertex_idx, component)
end

@inline function geom_input_vec3(location::UInt32, vertex_idx::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_geom_input_vec3(i32, i32, i32) #0
        define float @entry(i32 %loc, i32 %vidx, i32 %comp) #0 {
            %val = call float @_lava_geom_input_vec3(i32 %loc, i32 %vidx, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32, UInt32}, location, vertex_idx, component)
end

@inline function geom_input_vec2(location::UInt32, vertex_idx::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_geom_input_vec2(i32, i32, i32) #0
        define float @entry(i32 %loc, i32 %vidx, i32 %comp) #0 {
            %val = call float @_lava_geom_input_vec2(i32 %loc, i32 %vidx, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32, UInt32}, location, vertex_idx, component)
end

@inline function geom_input_f32(location::UInt32, vertex_idx::UInt32)
    Base.llvmcall(("""
        declare float @_lava_geom_input_f32(i32, i32) #0
        define float @entry(i32 %loc, i32 %vidx) #0 {
            %val = call float @_lava_geom_input_f32(i32 %loc, i32 %vidx)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, vertex_idx)
end

@inline function geom_input_i32(location::UInt32, vertex_idx::UInt32)
    Base.llvmcall(("""
        declare i32 @_lava_geom_input_i32(i32, i32) #0
        define i32 @entry(i32 %loc, i32 %vidx) #0 {
            %val = call i32 @_lava_geom_input_i32(i32 %loc, i32 %vidx)
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Int32, Tuple{UInt32, UInt32}, location, vertex_idx)
end

# High-level dispatch: geom_input(Type, location, vertex_idx)
# vertex_idx is 0-based (matches GLSL gl_in[0], gl_in[1], ...)
@inline geom_input(::Type{Vec4f}, loc::Integer, vidx::Integer) = Vec4f(
    geom_input_vec4(UInt32(loc), UInt32(vidx), UInt32(0)),
    geom_input_vec4(UInt32(loc), UInt32(vidx), UInt32(1)),
    geom_input_vec4(UInt32(loc), UInt32(vidx), UInt32(2)),
    geom_input_vec4(UInt32(loc), UInt32(vidx), UInt32(3)))
@inline geom_input(::Type{Vec3f}, loc::Integer, vidx::Integer) = Vec3f(
    geom_input_vec3(UInt32(loc), UInt32(vidx), UInt32(0)),
    geom_input_vec3(UInt32(loc), UInt32(vidx), UInt32(1)),
    geom_input_vec3(UInt32(loc), UInt32(vidx), UInt32(2)))
@inline geom_input(::Type{Vec2f}, loc::Integer, vidx::Integer) = Vec2f(
    geom_input_vec2(UInt32(loc), UInt32(vidx), UInt32(0)),
    geom_input_vec2(UInt32(loc), UInt32(vidx), UInt32(1)))
@inline geom_input(::Type{Float32}, loc::Integer, vidx::Integer) =
    geom_input_f32(UInt32(loc), UInt32(vidx))
@inline geom_input(::Type{Int32}, loc::Integer, vidx::Integer) =
    geom_input_i32(UInt32(loc), UInt32(vidx))

# Read gl_in[vidx].gl_Position
@inline geom_input_position(vidx::Integer) = Vec4f(
    geom_input_position(UInt32(vidx), UInt32(0)),
    geom_input_position(UInt32(vidx), UInt32(1)),
    geom_input_position(UInt32(vidx), UInt32(2)),
    geom_input_position(UInt32(vidx), UInt32(3)))

# ── Shader Wrappers for NamedTuple I/O ──
# These callable structs wrap user functions to translate NamedTuple returns
# into the low-level gfx_output/gfx_input intrinsic calls.

"""Wraps a vertex shader that returns a NamedTuple."""
struct VertexWrapper{F} end

@generated function (::VertexWrapper{F})(args...) where F
    quote
        result = F.instance(args...)
        emit_vertex_outputs(result)
    end
end

"""Wraps a fragment shader that takes a NamedTuple of varyings as first arg."""
struct FragmentWrapper{F, VOut} end

@generated function (::FragmentWrapper{F, VOut})(args...) where {F, VOut}
    quote
        inputs = load_fragment_inputs($VOut)
        result = F.instance(inputs, args...)
        emit_fragment_output(result)
    end
end

# ── Register intrinsic names for GPUCompiler validation ──
# These are the LLVM IR function names (not the Julia names)
push!(known_intrinsics, "_lava_gfx_set_position")
push!(known_intrinsics, "_lava_gfx_set_point_size")
push!(known_intrinsics, "_lava_gfx_output_f32")
push!(known_intrinsics, "_lava_gfx_output_vec4")
push!(known_intrinsics, "_lava_gfx_output_vec3")
push!(known_intrinsics, "_lava_gfx_output_vec2")
push!(known_intrinsics, "_lava_gfx_output_flat_f32")
push!(known_intrinsics, "_lava_gfx_output_flat_vec4")
push!(known_intrinsics, "_lava_gfx_output_flat_vec3")
push!(known_intrinsics, "_lava_gfx_output_flat_vec2")
push!(known_intrinsics, "_lava_gfx_input_vec4")
push!(known_intrinsics, "_lava_gfx_input_vec3")
push!(known_intrinsics, "_lava_gfx_input_vec2")
push!(known_intrinsics, "_lava_gfx_input_f32")
push!(known_intrinsics, "_lava_gfx_input_flat_vec4")
push!(known_intrinsics, "_lava_gfx_input_flat_vec2")
push!(known_intrinsics, "_lava_gfx_input_flat_f32")
push!(known_intrinsics, "_lava_gfx_dFdx_f32")
push!(known_intrinsics, "_lava_gfx_dFdy_f32")
push!(known_intrinsics, "_lava_gfx_emit_vertex")
push!(known_intrinsics, "_lava_gfx_end_primitive")
push!(known_intrinsics, "_lava_gfx_set_tess_level_outer")
push!(known_intrinsics, "_lava_gfx_set_tess_level_inner")
push!(known_intrinsics, "_lava_gfx_sample_2d")
# Geometry shader arrayed inputs
push!(known_intrinsics, "_lava_geom_input_position")
push!(known_intrinsics, "_lava_geom_input_vec4")
push!(known_intrinsics, "_lava_geom_input_vec3")
push!(known_intrinsics, "_lava_geom_input_vec2")
push!(known_intrinsics, "_lava_geom_input_f32")
push!(known_intrinsics, "_lava_geom_input_i32")
