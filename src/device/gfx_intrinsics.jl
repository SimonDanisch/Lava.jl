# Device-side graphics shader intrinsics for Lava.jl
#
# Graphics builtins via llvmcall, following the same addrspace(7) pattern
# as compute and RT builtins. The SPIR-V emitter recognizes these global names
# and creates Input/Output variables with BuiltIn decorations.
#
# Graphics-specific intrinsics (_lava_gfx_*) use special call names
# that the graphics emitter intercepts.

# ── Vertex Builtins ──

@inline function _lava_gfx_vertex_index()
    Base.llvmcall(("""
        @__spirv_BuiltInVertexIndex = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInVertexIndex, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

@inline function _lava_gfx_instance_index()
    Base.llvmcall(("""
        @__spirv_BuiltInInstanceIndex = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInInstanceIndex, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

# ── Fragment Builtins ──

@inline function _lava_gfx_frag_coord(dim::Integer=1)
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

@inline _lava_gfx_frag_coord_x() = _lava_gfx_frag_coord(1)
@inline _lava_gfx_frag_coord_y() = _lava_gfx_frag_coord(2)
@inline _lava_gfx_frag_coord_z() = _lava_gfx_frag_coord(3)
@inline _lava_gfx_frag_coord_w() = _lava_gfx_frag_coord(4)

@inline function _lava_gfx_front_facing()
    Base.llvmcall(("""
        @__spirv_BuiltInFrontFacing = external addrspace(7) global i1
        define i1 @entry() #0 {
            %val = load i1, ptr addrspace(7) @__spirv_BuiltInFrontFacing, align 1
            ret i1 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Bool, Tuple{})
end

# ── Output Builtins (Vertex/Geometry/TessEval) ──

@inline function _lava_gfx_set_position(x::Float32, y::Float32, z::Float32, w::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_set_position(float, float, float, float) #0
        define void @entry(float %x, float %y, float %z, float %w) #0 {
            call void @_lava_gfx_set_position(float %x, float %y, float %z, float %w)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{Float32, Float32, Float32, Float32}, x, y, z, w)
end

@inline function _lava_gfx_set_point_size(s::Float32)
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

# Output a single Float32 at a given location
@inline function _lava_gfx_output_f32(location::UInt32, val::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_f32(i32, float) #0
        define void @entry(i32 %loc, float %val) #0 {
            call void @_lava_gfx_output_f32(i32 %loc, float %val)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32}, location, val)
end

# Output a vec4 at a given location (most common: color output)
@inline function _lava_gfx_output_vec4(location::UInt32,
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

# Output a vec3 at a given location
@inline function _lava_gfx_output_vec3(location::UInt32,
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

# Output a vec2 at a given location
@inline function _lava_gfx_output_vec2(location::UInt32, x::Float32, y::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_output_vec2(i32, float, float) #0
        define void @entry(i32 %loc, float %x, float %y) #0 {
            call void @_lava_gfx_output_vec2(i32 %loc, float %x, float %y)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32, Float32}, location, x, y)
end

# Input a vec4 from a given location
@inline function _lava_gfx_input_vec4(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_vec4(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_vec4(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

# Input a vec3 from a given location
@inline function _lava_gfx_input_vec3(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_vec3(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_vec3(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

# Input a vec2 from a given location
@inline function _lava_gfx_input_vec2(location::UInt32, component::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_vec2(i32, i32) #0
        define float @entry(i32 %loc, i32 %comp) #0 {
            %val = call float @_lava_gfx_input_vec2(i32 %loc, i32 %comp)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32, UInt32}, location, component)
end

# Input a scalar float from a given location
@inline function _lava_gfx_input_f32(location::UInt32)
    Base.llvmcall(("""
        declare float @_lava_gfx_input_f32(i32) #0
        define float @entry(i32 %loc) #0 {
            %val = call float @_lava_gfx_input_f32(i32 %loc)
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, location)
end

# ── Geometry Shader ──

@inline function _lava_gfx_emit_vertex()
    Base.llvmcall(("""
        declare void @_lava_gfx_emit_vertex() #0
        define void @entry() #0 {
            call void @_lava_gfx_emit_vertex()
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{})
end

@inline function _lava_gfx_end_primitive()
    Base.llvmcall(("""
        declare void @_lava_gfx_end_primitive() #0
        define void @entry() #0 {
            call void @_lava_gfx_end_primitive()
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{})
end

@inline function _lava_gfx_invocation_id()
    Base.llvmcall(("""
        @__spirv_BuiltInInvocationId = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInInvocationId, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

@inline function _lava_gfx_primitive_id_in()
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

@inline function _lava_gfx_tess_coord(dim::Integer=1)
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

@inline _lava_gfx_tess_coord_u() = _lava_gfx_tess_coord(1)
@inline _lava_gfx_tess_coord_v() = _lava_gfx_tess_coord(2)
@inline _lava_gfx_tess_coord_w() = _lava_gfx_tess_coord(3)

@inline function _lava_gfx_set_tess_level_outer(idx::UInt32, val::Float32)
    Base.llvmcall(("""
        declare void @_lava_gfx_set_tess_level_outer(i32, float) #0
        define void @entry(i32 %idx, float %val) #0 {
            call void @_lava_gfx_set_tess_level_outer(i32 %idx, float %val)
            ret void
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Cvoid, Tuple{UInt32, Float32}, idx, val)
end

@inline function _lava_gfx_set_tess_level_inner(idx::UInt32, val::Float32)
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

@inline function _lava_gfx_sample_2d(binding::UInt32, u::Float32, v::Float32, component::UInt32)
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
