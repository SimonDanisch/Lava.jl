# The mesh pipeline: the stage vocabulary, and the emitter that lets ONE shader
# body run on it or on a native geometry stage.
#
# Part of the device vocabulary for the reason `graphics.jl` gives — a
# device-side name that a COMPILER lowers belongs under everything that compiles
# or runs a shader — and `MeshConfig` is here for the reason `Topology` is: its
# fields are COMPILE-TIME constants. A backend cannot emit the mesh entry point
# without knowing `max_vertices` and `max_primitives`; they are part of the
# output object's type in MSL and of the execution mode in SPIR-V. A description
# a compiler must read is not a runtime's to own.
#
# ── Why a mesh pipeline is portable vocabulary and not a Metal workaround ─────
#
# Metal has no geometry stage, so the question that reached this file was "how
# do we emulate one". That was the wrong question. Khronos and Apple converged
# on the same replacement: `VK_EXT_mesh_shader` gives Task + Mesh, Metal gives
# Object + Mesh, and they are the same two stages under different names. A
# geometry shader is expressible on a mesh pipeline; a mesh pipeline is NOT
# expressible on a geometry shader. So the mesh pipeline is the more general
# primitive and BOTH backends get it, while the geometry stage becomes the
# legacy path that lowers onto it where it does not exist natively.
#
# That is also what the no-vendor-conditional rule demands. The difference is
# fixed in the EMITTER, for everyone: the same lowering runs on Vulkan, where
# the native geometry stage stands beside it as the reference to diff against.
#
# ── Why the emitter is a VALUE and not a set of global intrinsics ─────────────
#
# A geometry-stage vocabulary of `emit_vertex!()` and `end_primitive!()` —
# argument-free globals writing to implicit output variables — cannot be lowered
# two ways, because there is nothing to
# dispatch on: a body calling `emit_vertex!()` names one lowering and no other.
#
# So the body takes an emitter:
#
#     function scatter_geometry(gs, prim)
#         for c in 1:4
#             emit!(gs, (position = v, uv = Vec2f(ux, uy), color = col))
#         end
#         endprimitive!(gs)
#     end
#
# and `typeof(gs)` selects the lowering. `NativeEmitter` writes the output
# variables and emits; `MeshEmitter` writes indexed slots in the mesh object.
# One source, two targets, no branch on a vendor anywhere.
#
# The vertex is a NamedTuple of `position` plus the pipeline's declared
# varyings — the same shape a vertex stage returns, so a body that was a vertex
# stage and a body that emits read alike.
#
# ── Why the output object is an explicit argument ────────────────────────────
#
# `set_mesh_vertex!(slot, vertex)`, matching the argument-free shape of the
# other builtins, cannot be lowered to Metal at all: in MSL the mesh object is a
# PARAMETER of the entry function — the stage
# is handed somewhere to write, and an intrinsic taking only a slot has no way
# to name it. SPIR-V is the odd one here, not Metal: its output arrays are
# implicit, so it can ignore an argument the other needs, while the reverse is
# impossible. Taking it is also what makes the emitter testable on the host,
# with `HostMeshOutput` below standing in for the driver's.
#
# Indices are ONE-BASED, like every index in this package.

# ── Flat ─────────────────────────────────────────────────────────────────────

"""
    Flat{T}

A stage output of type `T` that belongs to the PRIMITIVE rather than the vertex.

Written in an interface list beside ordinary types, and nothing wraps the value:

    outputs = (uv = Vec2f, halfwidth = Flat{Float32}, color = Flat{Vec4f})

Both APIs have two ways to deliver such a value and they are not the same thing.
A vertex stage has no primitive to attach one to, so there it is an
interpolation qualifier: the value is written per vertex, not interpolated, and
the provoking vertex is the one that counts. A mesh stage does have primitives
and writes each one once, through `air.set_primitive_data_mesh` on Metal and a
`perprimitiveEXT` output on Vulkan.

Declaring the PLACE rather than the qualifier is what lets one declaration serve
both. "This belongs to the primitive" implies "do not interpolate it"; the
reverse does not, which is why `flat` alone would leave the mesh lowering
guessing which vertex defined the primitive.
"""
struct Flat{T} end

"""
    isflat(T) -> Bool

Whether an entry in an interface list is a [`Flat`](@ref) one.
"""
isflat(::Type{<:Flat}) = true
isflat(::Type) = false

"""
    unflat(T) -> Type

The value type of an interface entry, with any [`Flat`](@ref) removed. The type
a stage actually produces, which is the same either way.
"""
unflat(::Type{Flat{T}}) where {T} = T
unflat(::Type{T}) where {T} = T

"""
    flatnames(interface::NamedTuple) -> Tuple{Vararg{Symbol}}
    smoothnames(interface::NamedTuple) -> Tuple{Vararg{Symbol}}

The names on each side of the split, in declaration order.

A geometry or mesh body emits the smooth ones per vertex and the flat ones per
primitive, so this partition is what says which call each field belongs in.
"""
flatnames(nt::NamedTuple) = filter(n -> isflat(nt[n]), keys(nt))
@doc (@doc flatnames) smoothnames(nt::NamedTuple) = filter(n -> !isflat(nt[n]), keys(nt))

"""
    valuetypes(interface::NamedTuple) -> NamedTuple

The same list with every [`Flat`](@ref) removed, which is what a backend builds
its output struct from: flatness decides WHERE a value goes, never what it is.
"""
valuetypes(nt::NamedTuple) = NamedTuple{keys(nt)}(map(unflat, values(nt)))

# ── Configuration a compiler reads ───────────────────────────────────────────

"""
    MeshConfig(; max_vertices, max_primitives, topology = TriangleList(), threads = 1)

What a mesh stage may produce, and how wide its threadgroup is.

Every field is a compile-time constant: `max_vertices` and `max_primitives`
bound the output object a backend allocates per threadgroup, and `topology` is
what the emitted primitives ARE. A stage that writes past either bound is
undefined on both APIs, which is why [`MeshEmitter`](@ref) is given the bound
rather than trusting the body.

`threads` is how many invocations cooperate on one output object. They share it,
so the per-invocation budget is `max_vertices ÷ threads` — see [`MeshEmitter`](@ref)
for the slot arithmetic that follows from that.

Sized for what fits: the output object competes for threadgroup memory, so
`max_vertices * sizeof(varyings)` is the number to keep an eye on. A body
expanding a point into a quad with nine varyings is about 144 bytes per
primitive and a wide threadgroup is comfortable; one with `max_vertices = 128`
per invocation forces `threads = 1` and most of the group idles.
"""
struct MeshConfig{O<:Topology}
    max_vertices::Int
    max_primitives::Int
    topology::O
    threads::Int
end

# A mesh output object's vertex indices are EIGHT BITS: the intrinsic that
# writes one is `air.set_index_mesh(ptr addrspace(7), i32, i8)`, read out of
# `particle_gaussian_mesh` in VFX.framework's shipping metallib, whose own
# declaration is `mesh<particle_vertex_io, particle_primitive_io, 96, 32,
# triangle>`. So this is a hard ceiling and not a style choice, and a config
# that exceeds it produces primitives pointing at wrapped-around vertices —
# geometry that is wrong rather than absent, which is the worse failure.
const MAX_MESH_VERTICES = 256

function MeshConfig(; max_vertices::Integer, max_primitives::Integer,
                      topology::Topology = TriangleList(), threads::Integer = 1)
    max_vertices > 0 || throw(ArgumentError("MeshConfig: max_vertices must be positive"))
    max_vertices <= MAX_MESH_VERTICES || throw(ArgumentError(
        "MeshConfig: max_vertices is $max_vertices, and a mesh output object " *
        "addresses its vertices with 8 bits, so at most $MAX_MESH_VERTICES fit. " *
        "Split the work across more threadgroups instead of widening one."))
    max_primitives > 0 || throw(ArgumentError("MeshConfig: max_primitives must be positive"))
    threads > 0 || throw(ArgumentError("MeshConfig: threads must be positive"))
    MeshConfig(Int(max_vertices), Int(max_primitives), topology, Int(threads))
end

"""
    ObjectConfig(; threads = 1)

The object stage's threadgroup width.

The object stage decides how many mesh threadgroups to dispatch — GPU-side
amplification, which is the half of the mesh pipeline a geometry shader never
had. It is optional on both APIs: a pipeline with no object stage dispatches
mesh threadgroups directly, which is what the geometry lowering does.

There is no payload field yet, deliberately. A payload is how an object stage
hands data down to the mesh stage, and its layout rules differ enough between
the two APIs that guessing them here would cement the wrong shape. Nothing in
the tree amplifies yet; when something does, the AIR and SPIR-V ground truth
comes first and the field follows it.
"""
struct ObjectConfig
    threads::Int
end
ObjectConfig(; threads::Integer = 1) = ObjectConfig(Int(threads))

# ── The mesh stage's device intrinsics ───────────────────────────────────────

"""
    set_mesh_vertex!(out, slot::Integer, vertex::NamedTuple)

Write `vertex` into the output object `out` at `slot`, counting from one.

`vertex` is `position` plus the pipeline's declared varyings — the same
NamedTuple a vertex stage returns.

Slots are the threadgroup's, not the invocation's: cooperating invocations
write disjoint ranges of one object, which is what [`MeshEmitter`](@ref)'s
`vbase` is for.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override set_mesh_vertex!(out, ::Integer, ::NamedTuple)
    ```
    dispatching on their own output object type.
"""
function set_mesh_vertex! end

"""
    set_mesh_triangle!(out, slot::Integer, i0::Integer, i1::Integer, i2::Integer)
    set_mesh_line!(out, slot::Integer, i0::Integer, i1::Integer)
    set_mesh_point!(out, slot::Integer, i0::Integer)

Write the primitive at `slot` as indices into the vertices written by
[`set_mesh_vertex!`](@ref). Both `slot` and the indices count from one.

One function per topology rather than a vector of indices, because the
primitive's arity is fixed by the pipeline's `MeshConfig` and a stage that
emits triangles never emits anything else.

!!! note
    Backend implementations **must** implement the ones whose topologies they
    accept:
    ```
    @device_override set_mesh_triangle!(out, ::Integer, ::Integer, ::Integer, ::Integer)
    @device_override set_mesh_line!(out, ::Integer, ::Integer, ::Integer)
    @device_override set_mesh_point!(out, ::Integer, ::Integer)
    ```
"""
function set_mesh_triangle! end
@doc (@doc set_mesh_triangle!) function set_mesh_line! end
@doc (@doc set_mesh_triangle!) function set_mesh_point! end

"""
    set_mesh_primitive_data!(out, slot::Integer, data::NamedTuple)

Write the PER-PRIMITIVE values of the primitive at `slot`, counting from one.

Separate from [`set_mesh_vertex!`](@ref) because the two are separate planes in
both APIs: `air.set_primitive_data_mesh` on Metal, a `perprimitiveEXT` output on
Vulkan. `data` is the `Flat`-declared subset of the stage's outputs — see
[`Flat`](@ref) for why the declaration names the plane and not the interpolation
qualifier.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override set_mesh_primitive_data!(out, ::Integer, ::NamedTuple)
    ```
"""
function set_mesh_primitive_data! end

"""
    set_mesh_outputs!(out, nvertices::Integer, nprimitives::Integer)

Declare how much of the output object this threadgroup actually filled.

Called ONCE per threadgroup, not once per invocation, which is why it is not a
method on [`MeshEmitter`](@ref): an emitter is one invocation's.

Both counts are taken because SPIR-V's `OpSetMeshOutputsEXT` needs both; Metal
bounds the vertices by the object's type and reads only the primitive count.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override set_mesh_outputs!(out, ::Integer, ::Integer)
    ```
"""
function set_mesh_outputs! end

"""
    set_mesh_groups!(grid, n::Integer)

From an object stage: dispatch `n` mesh threadgroups.

`grid` is the object stage's grid handle, an explicit parameter for the same
reason the mesh output object is one.

!!! note
    Backend implementations with an object stage **must** implement:
    ```
    @device_override set_mesh_groups!(grid, ::Integer)
    ```
"""
function set_mesh_groups! end

"""
    mesh_thread_index()::Int32

Which invocation within the mesh threadgroup this is, counting from one.

Argument-free, unlike the writers above, because it genuinely is a builtin on
both APIs rather than an operation on something the stage was handed.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override mesh_thread_index()::Int32
    ```
"""
function mesh_thread_index end

"""
    mesh_group_index()::Int32

Which mesh threadgroup this is, counting from one.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override mesh_group_index()::Int32
    ```
"""
function mesh_group_index end

const _MESH_OUTSIDE = " is a mesh-stage builtin, and only a compiled mesh " *
                      "stage has one. Calling it on the host cannot mean anything."

for f in (:mesh_thread_index, :mesh_group_index)
    @eval $f(args...) = error($(string(f)) * _MESH_OUTSIDE)
end

# ── The emitter ──────────────────────────────────────────────────────────────

"""
    PrimitiveEmitter

What a geometry body emits through. `emit!` and `endprimitive!` dispatch on it,
which is what lets one body run on a native geometry stage and on a mesh stage.

See [`NativeEmitter`](@ref) and [`MeshEmitter`](@ref).
"""
abstract type PrimitiveEmitter end

"""
    emit!(e::PrimitiveEmitter, vertex::NamedTuple)

Emit one vertex. `vertex` is `position` plus the pipeline's declared varyings.

    endprimitive!(e::PrimitiveEmitter)

Finish the primitive being emitted, so the next `emit!` starts a new one. For a
strip topology this is the restart; for a list topology it only ends the run.
"""
function emit! end
@doc (@doc emit!) function endprimitive! end

"""
    NativeEmitter{Out, FN}()

Emits through a backend's own geometry stage.

Zero-sized, like the stage it emits into: a geometry stage's output VARIABLES
are implicit, so there is nothing to carry at run time. `Out` is the declared
output type (`position` plus the varyings, `Flat` stripped) and `FN` the names
among them that belong to the primitive rather than the vertex — the same pair
`MeshEmitter{O,FN}` carries, and type parameters for the same reason: a
varying's location is its position in the declaration and its flatness picks
the write, so both decide instructions and neither can be a field.

The backend's compiler turns `emit!` into "write the output variables, then
emit", which is one instruction pair and not something assembled from smaller
pieces, so this is where `emit_vertex!` and `end_primitive!` are used and a
shader body has no reason to call those directly.

A backend without a geometry stage never constructs one; `Mantle` picks the
emitter from what the backend reports.

!!! note
    Backend implementations with a geometry stage **must** implement:
    ```
    @device_override emit!(::NativeEmitter, ::NamedTuple)
    @device_override endprimitive!(::NativeEmitter)
    ```
"""
struct NativeEmitter{Out, FN} <: PrimitiveEmitter end

"""No flat outputs, which is the common case and what every smooth-only stage is."""
NativeEmitter{Out}() where {Out} = NativeEmitter{Out, ()}()

# The host answers. A bare MethodError here reads as "you passed the wrong
# emitter", which is the one thing that did not happen.
emit!(::NativeEmitter, ::NamedTuple) = error("emit!(::NativeEmitter, …)" * _MESH_OUTSIDE)
endprimitive!(::NativeEmitter) = error("endprimitive!(::NativeEmitter)" * _MESH_OUTSIDE)

"""
    MeshEmitter{O}(out, vbase, pbase, maxprimitives)

Emits into a mesh stage's output object, at slots this invocation owns.

`O` is the output [`Topology`](@ref), and it is a type parameter because `emit!`
dispatches on it: a strip closes a primitive on every vertex after the second
and alternates winding, a list closes one every `n`-th vertex, and neither is a
runtime branch.

`vbase` and `pbase` are the first vertex and primitive slot of this invocation's
range. Reserving a fixed range per invocation rather than allocating from a
shared cursor is what preserves the geometry stage's ORDER guarantee — input
primitive order is invocation order is slot order — and it costs nothing,
because the ranges are computed from `mesh_thread_index()`.

Unlike a geometry stage, a mesh threadgroup declares its output count once for
all invocations, so an invocation that emits fewer primitives than it reserved
leaves slots behind. [`finish!`](@ref) fills them.
"""
mutable struct MeshEmitter{O<:Topology, FN, T} <: PrimitiveEmitter
    const out::T
    const vbase::Int32
    const pbase::Int32
    const maxprimitives::Int32
    nv::Int32     # vertices this invocation has written
    np::Int32     # primitives this invocation has written
    run::Int32    # vertices since the last endprimitive!
end

function MeshEmitter{O,FN}(out::T, vbase::Integer, pbase::Integer,
                           maxprimitives::Integer) where {O<:Topology, FN, T}
    MeshEmitter{O,FN,T}(out, Int32(vbase), Int32(pbase), Int32(maxprimitives),
                        Int32(0), Int32(0), Int32(0))
end

"""No flat outputs, which is the common case and what every smooth-only stage is."""
MeshEmitter{O}(out, vbase::Integer, pbase::Integer, maxprimitives::Integer) where {O<:Topology} =
    MeshEmitter{O,()}(out, vbase, pbase, maxprimitives)

"""Vertices this emitter has written."""
nvertices(e::MeshEmitter) = e.nv
"""Primitives this emitter has written."""
nprimitives(e::MeshEmitter) = e.np

# The slot the next vertex goes in, and the slot the next primitive goes in.
@inline _vslot(e::MeshEmitter) = e.vbase + e.nv
@inline _pslot(e::MeshEmitter) = e.pbase + e.np

@inline function _wrote!(e::MeshEmitter)
    e.nv += Int32(1)
    e.run += Int32(1)
    return e
end

# Where a vertex's fields go. `FN` is the `Flat`-declared subset of the stage's
# outputs, so the split is a compile-time `structdiff` and the two planes of the
# mesh output object are reached by name rather than by position.
#
# `position` is never in `FN` — it is not a declared output at all — so it stays
# on the vertex side, which is where it belongs.
@inline function _writevertex!(e::MeshEmitter{O,FN}, slot, v::NamedTuple) where {O,FN}
    set_mesh_vertex!(e.out, slot, Base.structdiff(v, NamedTuple{FN}))
    return nothing
end

# A primitive just closed at `slot`, so its per-primitive values are written ONCE
# — which is the whole point of declaring them as the primitive's. `v` is the
# vertex that closed it; for a `Flat` field every vertex of the primitive carries
# the same value, exactly as a provoking-vertex convention would give.
@inline function _closed!(e::MeshEmitter{O,FN}, slot, v::NamedTuple) where {O,FN}
    isempty(FN) || set_mesh_primitive_data!(e.out, slot, NamedTuple{FN}(v))
    e.np += Int32(1)
    return nothing
end

@inline function emit!(e::MeshEmitter{TriangleStrip}, v::NamedTuple)
    i = _vslot(e)
    _writevertex!(e, i, v)
    if e.run >= Int32(2)
        a = i - Int32(2)
        b = i - Int32(1)
        # A strip alternates winding so that every triangle in it faces the same
        # way. Emitting (a,b,c) for all of them would make every second triangle
        # back-facing and `CullBack` would drop half the strip — which looks
        # like a shader bug and is not one.
        if iseven(e.run)
            set_mesh_triangle!(e.out, _pslot(e), a, b, i)
        else
            set_mesh_triangle!(e.out, _pslot(e), b, a, i)
        end
        _closed!(e, _pslot(e), v)
    end
    return _wrote!(e)
end

@inline function emit!(e::MeshEmitter{TriangleList}, v::NamedTuple)
    i = _vslot(e)
    _writevertex!(e, i, v)
    if e.run % Int32(3) == Int32(2)
        set_mesh_triangle!(e.out, _pslot(e), i - Int32(2), i - Int32(1), i)
        _closed!(e, _pslot(e), v)
    end
    return _wrote!(e)
end

@inline function emit!(e::MeshEmitter{LineStrip}, v::NamedTuple)
    i = _vslot(e)
    _writevertex!(e, i, v)
    if e.run >= Int32(1)
        set_mesh_line!(e.out, _pslot(e), i - Int32(1), i)
        _closed!(e, _pslot(e), v)
    end
    return _wrote!(e)
end

@inline function emit!(e::MeshEmitter{LineList}, v::NamedTuple)
    i = _vslot(e)
    _writevertex!(e, i, v)
    if isodd(e.run)
        set_mesh_line!(e.out, _pslot(e), i - Int32(1), i)
        _closed!(e, _pslot(e), v)
    end
    return _wrote!(e)
end

@inline function emit!(e::MeshEmitter{PointList}, v::NamedTuple)
    i = _vslot(e)
    _writevertex!(e, i, v)
    set_mesh_point!(e.out, _pslot(e), i)
    _closed!(e, _pslot(e), v)
    return _wrote!(e)
end

@inline function endprimitive!(e::MeshEmitter)
    e.run = Int32(0)
    return e
end

"""
    finish!(e::MeshEmitter, cullvertex::NamedTuple)

Fill this invocation's unused primitive slots so they draw nothing.

A mesh threadgroup declares one output count for all of its invocations, so the
slots an invocation did not use are still IN the count and still rasterised.
Compacting them away would need a threadgroup-wide prefix sum; pointing them at
a vertex that cannot be seen costs one vertex write and no synchronisation, and
is worth revisiting only when the average fill is low enough to pay for the sum.

`cullvertex` is that vertex: the pipeline's varyings with a `position` outside
the clip volume. Degenerate indices alone would do for triangles and lines,
which have zero area whatever their vertex holds, but a point list has no
degenerate form — one index is one point and it WILL be drawn — so the position
has to do the work for every topology rather than only the one that needs it.

Writing it lazily is why this takes it rather than the constructor: an
invocation that filled its budget never writes one. That a slot is free when one
is needed follows from the topologies here, in all of which fewer primitives
implies fewer vertices.

Called once per invocation, after the body has emitted everything.
"""
@inline function finish!(e::MeshEmitter{O}, cullvertex::NamedTuple) where {O<:Topology}
    e.np < e.maxprimitives || return e
    d = _vslot(e)
    set_mesh_vertex!(e.out, d, cullvertex)
    while e.np < e.maxprimitives
        _degenerate!(e.out, O, _pslot(e), d)
        e.np += Int32(1)
    end
    return e
end

@inline _degenerate!(o, ::Type{TriangleStrip}, p, d) = set_mesh_triangle!(o, p, d, d, d)
@inline _degenerate!(o, ::Type{TriangleList},  p, d) = set_mesh_triangle!(o, p, d, d, d)
@inline _degenerate!(o, ::Type{LineStrip},     p, d) = set_mesh_line!(o, p, d, d)
@inline _degenerate!(o, ::Type{LineList},      p, d) = set_mesh_line!(o, p, d, d)
@inline _degenerate!(o, ::Type{PointList},     p, d) = set_mesh_point!(o, p, d)

# ── The host's output object ─────────────────────────────────────────────────
#
# Not a mock. The emitter's slot arithmetic — the strip's winding alternation,
# where a run closes a primitive, which slots `finish!` fills — is portable code
# with no driver in it, and this is what lets it be RUN and diffed without one.
# When the geometry lowering lands, the same body over `HostMeshOutput` is the
# reference its GPU result is checked against, which is the whole reason the
# lowering is worth more than a per-backend rewrite.

"""
    HostMeshOutput()

A mesh output object that records what was written to it, on the host.

Slots are sparse because invocations own disjoint ranges and an invocation may
leave part of its own range unwritten, so both maps are keyed by slot rather
than filled in order.
"""
struct HostMeshOutput
    vertices::Dict{Int32,Any}
    primitives::Dict{Int32,Tuple}
    primitivedata::Dict{Int32,Any}
    declared::Base.RefValue{Tuple{Int32,Int32}}
end

HostMeshOutput() = HostMeshOutput(Dict{Int32,Any}(), Dict{Int32,Tuple}(),
                                  Dict{Int32,Any}(), Ref((Int32(0), Int32(0))))

function set_mesh_vertex!(o::HostMeshOutput, slot::Integer, v::NamedTuple)
    o.vertices[Int32(slot)] = v
    return nothing
end

function set_mesh_triangle!(o::HostMeshOutput, slot::Integer,
                            i0::Integer, i1::Integer, i2::Integer)
    o.primitives[Int32(slot)] = (Int32(i0), Int32(i1), Int32(i2))
    return nothing
end

function set_mesh_line!(o::HostMeshOutput, slot::Integer, i0::Integer, i1::Integer)
    o.primitives[Int32(slot)] = (Int32(i0), Int32(i1))
    return nothing
end

function set_mesh_point!(o::HostMeshOutput, slot::Integer, i0::Integer)
    o.primitives[Int32(slot)] = (Int32(i0),)
    return nothing
end

function set_mesh_primitive_data!(o::HostMeshOutput, slot::Integer, data::NamedTuple)
    o.primitivedata[Int32(slot)] = data
    return nothing
end

function set_mesh_outputs!(o::HostMeshOutput, nv::Integer, np::Integer)
    o.declared[] = (Int32(nv), Int32(np))
    return nothing
end

"""
    primitives(o::HostMeshOutput) -> Vector{Tuple}

The primitives in slot order, which is the order they will be rasterised in.
Each is the index tuple its topology has: three for a triangle, two for a line,
one for a point.
"""
primitives(o::HostMeshOutput) = [o.primitives[k] for k in sort!(collect(keys(o.primitives)))]

"""The vertices in slot order, as the NamedTuples that were written."""
vertices(o::HostMeshOutput) = [o.vertices[k] for k in sort!(collect(keys(o.vertices)))]

"""The per-primitive data in slot order, one entry per primitive that got any."""
primitivedata(o::HostMeshOutput) =
    [o.primitivedata[k] for k in sort!(collect(keys(o.primitivedata)))]
