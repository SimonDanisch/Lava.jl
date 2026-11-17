# The graphics stage intrinsics a shader body calls.
#
# Part of the device vocabulary: a device-side name that a COMPILER lowers belongs
# under everything that compiles or runs a shader. Each backend overrides the
# function declared here, and a downstream package imports one name from one place.
#
# What is NOT here, and deliberately:
#
#   * `set_position!`, `gfx_output`/`gfx_input` and their flat variants,
#     `geom_input`, `geom_input_position` — the older location-based way of
#     moving values between stages. A pipeline declares `varyings = (albedo =
#     Vec4f, …)`, a vertex stage returns `(position = …, albedo = …)` and a
#     fragment stage receives them by name. That is one declaration instead of
#     matched pairs of numbered slots, it works on both backends today, and
#     `bench/showcase.jl` is written in it. Seven names removed rather than
#     moved.
#   * A geometry stage's `emit_vertex!`/`end_primitive!` as the vocabulary a
#     shader body writes. They are still declared below, but as the LEAF a
#     native geometry lowering reaches, not as something an author calls: a body
#     emits through `emit!(gs, vertex)`, and the emitter it was handed decides
#     whether that becomes these or the mesh stage's indexed writes. See
#     `mesh.jl`, which explains why an argument-free pair cannot be lowered two
#     ways and why the mesh pipeline is the more general of the two.
#
# Indices are ONE-BASED on every backend. A shader that has to remember which
# convention a builtin follows will eventually get it wrong.

const _GFX_OUTSIDE = " is a shader builtin: it reads a value the rasteriser " *
                     "supplies, and only a compiled stage has one. Calling it " *
                     "on the host cannot mean anything."

"""
    vertex_index()::Int32

Which vertex this invocation is for, counting from one.

`gl_VertexIndex` in GLSL, `[[vertex_id]]` in MSL. Vertex stage only.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override vertex_index()::Int32
    ```
"""
function vertex_index end

"""
    VertexIndex(value::Int32)

Which vertex a vertex body is being evaluated for, handed to it EXPLICITLY.

Declared as a leading parameter:

    function myvertex(vid::VertexIndex, positions, colours)
        i = vid.value
        …
    end

`vertex_index()` above is the same number and is what a body uses when it runs as
a real vertex stage: the rasteriser supplies it and there is nothing to pass. This
marker exists because a mesh stage HAS no such builtin, and a geometry stage
lowered onto one (see Mantle's `lower_geometry_to_mesh`) has to evaluate the
vertex body once per input vertex of its primitive — several times, in one
invocation, at indices it computes. A zero-argument builtin cannot answer
differently on each of those calls; a parameter can.

For an INDEXED draw the value is what the index buffer holds, matching
`vertex_index()`'s meaning on a real vertex stage rather than the position in the
draw. So the same body reads the same vertex either way.

Opt in per shader. A body that does not declare it keeps using `vertex_index()`
and nothing about it changes; a backend passes this only to bodies that ask, so
the two spellings coexist and neither is privileged.
"""
struct VertexIndex
    value::Int32
end

VertexIndex(i::Integer) = VertexIndex(Int32(i))

"""
    wantsvertexindex(f, argtypes::Tuple) -> Bool

Whether `f` declares a leading [`VertexIndex`](@ref) for these argument types.

Asked of the METHOD TABLE rather than recorded on the `VertexShader`, so a shader
declares the parameter in the one place a reader looks — its own signature — and
cannot have the declaration and the flag disagree.
"""
wantsvertexindex(@nospecialize(f), @nospecialize(argtypes::Tuple)) =
    hasmethod(f, Tuple{VertexIndex, argtypes...})

"""
    instance_index()::Int32

Which instance this invocation is for, counting from one.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override instance_index()::Int32
    ```
"""
function instance_index end

"""
    frag_coord(dim = 1)::Float32

One component of the interpolated fragment position, counting from one: 1 is x,
4 is w. `gl_FragCoord` in GLSL, `[[position]]` in MSL. Fragment stage only.

`frag_coord_x`, `frag_coord_y`, `frag_coord_z`, `frag_coord_w` and
`frag_coord_xy` name the components a shader usually wants, so the common case
needs no literal.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override frag_coord(dim::Integer = 1)::Float32
    ```
"""
function frag_coord end

"""x of [`frag_coord`](@ref)."""
function frag_coord_x end
"""y of [`frag_coord`](@ref)."""
function frag_coord_y end
"""z of [`frag_coord`](@ref), the depth this fragment writes."""
function frag_coord_z end
"""w of [`frag_coord`](@ref)."""
function frag_coord_w end
"""x and y of [`frag_coord`](@ref), as a tuple."""
function frag_coord_xy end

"""
    dFdx(v::Float32)::Float32
    dFdy(v::Float32)::Float32

How `v` changes across the fragment quad in x and in y. Fragment stage only:
the value is a difference between neighbouring invocations, so it exists only
where invocations run in a quad.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override dFdx(::Float32)::Float32
    @device_override dFdy(::Float32)::Float32
    ```
"""
function dFdx end
@doc (@doc dFdx) function dFdy end

"""
    discard()

Throw this fragment away: it contributes no colour and, crucially, no DEPTH.
Fragment stage only.

The depth half is the reason this exists. A blended pass that writes depth has
an alpha-zero corner of a glyph or marker quad occlude whatever should have
shown through it, because depth is written before blending ever looks at alpha.
Discarding is what lets a blended pass keep a depth buffer at all.

Execution CONTINUES past the call and the shader still returns a colour, which
is simply not used. This is not a `return`, and every backend's primitive
behaves that way: MSL's `discard_fragment()` and SPIR-V's demote both mark the
invocation and carry on.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override discard()
    ```
"""
function discard end

"""
    set_point_size!(s::Float32)

The size, in pixels, of the point this vertex becomes. Only meaningful when the
pipeline's topology is a point list.

A setter and not a returned field, unlike `position` and the varyings, because
it is written by a minority of shaders and threading it through every vertex
stage's return type to carry `nothing` almost always is the worse trade.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override set_point_size!(::Float32)
    ```
"""
function set_point_size! end

"""
    sample_texture_2d(binding::UInt32, u::Float32, v::Float32, component::UInt32)::Float32

One component of a filtered sample from the 2D texture at `binding`, at
normalised coordinates `(u, v)`.

A component at a time, and a `UInt32` binding rather than a texture object,
because that is the shape both backends' bound-texture access has: the shader
names a slot, not a handle.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override sample_texture_2d(::UInt32, ::Float32, ::Float32, ::UInt32)::Float32
    ```
"""
function sample_texture_2d end

# ── Geometry stage configuration ─────────────────────────────────────────────

"""
    GeometryConfig(; input = TriangleList(), output = TriangleStrip(),
                     max_vertices = 3, invocations = 1)

What a geometry stage consumes and may produce.

`input` fixes how many vertices one invocation sees — three for a
`TriangleList`, four for either adjacency form — and `output` is what the
emitted vertices are grouped into. `max_vertices` bounds one invocation's
output, and `invocations` is how many times the stage runs per input primitive.

Here, and not in a runtime, for the reason [`MeshConfig`](@ref) is: a compiler
reads every field to emit the stage's execution modes, and a portable pipeline
description has to hold one without depending on a SPIR-V compiler.
"""
struct GeometryConfig{I<:Topology, O<:Topology}
    input_topology::I
    output_topology::O
    max_vertices::Int
    invocations::Int
end

function GeometryConfig(; input::Topology = TriangleList(),
                          output::Topology = TriangleStrip(),
                          max_vertices::Integer = 3, invocations::Integer = 1)
    max_vertices > 0 || throw(ArgumentError("GeometryConfig: max_vertices must be positive"))
    invocations > 0 || throw(ArgumentError("GeometryConfig: invocations must be positive"))
    GeometryConfig(input, output, Int(max_vertices), Int(invocations))
end

"""
    inputvertices(topology) -> Int

How many vertices one primitive of `topology` hands a geometry stage.

Both adjacency forms give four: the segment plus a neighbour on each side. They
differ in how the index buffer is walked, not in what one invocation sees.
"""
inputvertices(::PointList)          = 1
inputvertices(::LineList)           = 2
inputvertices(::LineStrip)          = 2
inputvertices(::LineListAdjacency)  = 4
inputvertices(::LineStripAdjacency) = 4
inputvertices(::TriangleList)       = 3
inputvertices(::TriangleStrip)      = 3

# ── Geometry stage ───────────────────────────────────────────────────────────
#
# Declared, and not every backend has them: Metal has no geometry stage. A
# backend says so through `caps` and a pipeline naming one is refused there,
# which is a question a caller can ask, unlike a `MethodError` from inside a
# shader compile.
#
# These are what `emit!(::NativeEmitter, …)` lowers to, and not what a body
# calls. A body that names them directly has picked one of the two lowerings by
# hand and will not run where that stage does not exist; `mesh.jl` has the
# emitter that makes the choice instead.

"""
    emit_vertex!()

Emit the vertex the geometry stage has been writing.

!!! note
    Backend implementations with a geometry stage **must** implement:
    ```
    @device_override emit_vertex!()
    ```
"""
function emit_vertex! end

"""
    end_primitive!()

Finish the primitive the geometry stage has been emitting vertices for.

!!! note
    Backend implementations with a geometry stage **must** implement:
    ```
    @device_override end_primitive!()
    ```
"""
function end_primitive! end

"""
    primitive_id_in()::UInt32

Index of the primitive the geometry stage is being run for.

!!! note
    Backend implementations with a geometry stage **must** implement:
    ```
    @device_override primitive_id_in()::UInt32
    ```
"""
function primitive_id_in end

# ── Clip space ───────────────────────────────────────────────────────────────

"""
    clip_y(y::Float32)::Float32

`y` in this backend's clip space, given a `y` in the portable one.

**The portable clip space has +y pointing DOWN the screen**, which is Vulkan's.
A backend whose own points up overrides this with a negation, and the vertex
stage applies it to every `position` it writes.

It is NOT one of the builtins above and its default is the identity rather than
an error, for three reasons: it takes an argument, it is meaningful on the host,
and a backend that agrees with the portable convention should not have to say
so.

A shader that does the viewport transform BY HAND — a shadow lookup turning
`sunvp * world` into a texel row — has to apply the same mirror, which is what
this is exported for. It is its own inverse.

Getting the vertex stage's half right and this half wrong is not a visible
mistake: a mirrored scene still looks like a scene, its shadow map is mirrored
with it, and two renders from the same backend agree perfectly. What gave it
away in September 2026 was a person looking at a window.
"""
clip_y(y::Float32) = y

# The host answers, which exist to SAY something rather than to work.
for f in (:vertex_index, :instance_index, :frag_coord, :frag_coord_x, :frag_coord_y,
          :frag_coord_z, :frag_coord_w, :frag_coord_xy, :dFdx, :dFdy, :discard,
          :set_point_size!, :sample_texture_2d, :emit_vertex!, :end_primitive!,
          :primitive_id_in)
    @eval $f(args...) = error($(string(f)) * _GFX_OUTSIDE)
end
