"""
Primitive topology: how a vertex stream is grouped into primitives.

Two sides need this vocabulary:

  * a **compiler** dispatches on it to emit the geometry stage's input
    execution mode — `TriangleList` → three vertices per invocation,
    `LineListAdjacency` → four;
  * a **backend** dispatches on it to create a pipeline —
    `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST`, `MTLPrimitiveTypeTriangle`.

One declaration for both: two declarations of one type can only be bridged by a
positional copy, which drifts.

Nothing here is any one API's. The names are the ones every graphics API uses.
"""
abstract type Topology end

"Three vertices per triangle, disjoint groups."
struct TriangleList <: Topology end

"A sliding three-wide window: `n` vertices give `n-2` triangles."
struct TriangleStrip <: Topology end

"Two vertices per line, disjoint groups."
struct LineList <: Topology end

"A sliding two-wide window: `n` vertices give `n-1` segments."
struct LineStrip <: Topology end

"One vertex per point."
struct PointList <: Topology end

"Patches for the tessellation stages."
struct PatchList <: Topology end

# Both feed a geometry shader four vertices per primitive. They differ in how the
# index buffer is walked: the LIST form consumes a disjoint group of 4 per
# primitive, the STRIP form slides a 4-wide window one index at a time. An
# adjacency index list built for a strip (Makie's polylines: `0 0 1 2 3 3`)
# yields ⌊n/4⌋ primitives under the list form instead of n-3 — a polyline drawn
# as scattered dashes.
"Lines with adjacency, four vertices per primitive in disjoint groups."
struct LineListAdjacency <: Topology end
"Lines with adjacency, a four-wide window sliding one vertex per primitive."
struct LineStripAdjacency <: Topology end

"""
    primitivevertices(t::Topology) -> Int

How many vertices one primitive of `t` has.

The number a geometry stage is handed per invocation, and the number a mesh
lowering writes per primitive. It is a property of the TOPOLOGY and nothing
else, which is why it is here: it was written out twice — once in Lava's SPIR-V
emitter to pick the geometry stage's input execution mode, once in Mantle's
Vulkan backend to size the arrayed inputs its geometry wrapper reads — and two
tables of one fact drift the moment a topology is added to one of them.

A strip's answer is the size of its sliding WINDOW, not the length of the strip:
a `TriangleStrip` delivers three vertices per primitive like a `TriangleList`,
and the two differ in how the index stream is walked, not in what a primitive
is.
"""
function primitivevertices end

primitivevertices(::PointList)          = 1
primitivevertices(::LineList)           = 2
primitivevertices(::LineStrip)          = 2
primitivevertices(::TriangleList)       = 3
primitivevertices(::TriangleStrip)      = 3
primitivevertices(::LineListAdjacency)  = 4
primitivevertices(::LineStripAdjacency) = 4

# No answer, rather than a wrong one: a patch's vertex count is the tessellation
# configuration's (`TessConfig.patch_size`), not the topology's, and a caller
# that reaches here has asked the wrong thing.
primitivevertices(::PatchList) = throw(ArgumentError(
    "a patch's vertex count is the tessellation configuration's, not the topology's"))

"""
    primitivestride(t::Topology) -> Int

How far the input assembler advances between one primitive and the next, in
vertices.

This is the whole difference between a LIST and a STRIP, and it is not
[`primitivevertices`](@ref): both `LineListAdjacency` and `LineStripAdjacency`
hand a geometry stage four vertices, and they differ only here — the list
consumes a disjoint group of four, the strip slides a four-wide window one index
at a time. An adjacency list built for a strip read as a list yields `n / 4`
primitives instead of `n - 3`, which draws a polyline as scattered dashes.

A list's stride is its arity; a strip's is one.
"""
function primitivestride end

primitivestride(t::Union{PointList,LineList,TriangleList,LineListAdjacency}) =
    primitivevertices(t)
primitivestride(::Union{LineStrip,TriangleStrip,LineStripAdjacency}) = 1

primitivestride(::PatchList) = throw(ArgumentError(
    "a patch's stride is the tessellation configuration's patch size, not the " *
    "topology's"))

"""
    primitivecount(t::Topology, nvertices) -> Int

How many primitives a stream of `nvertices` assembles under `t`.

`(nvertices - (verts - stride)) / stride`: a list divides by its arity, a strip
subtracts the window it has to fill before the first primitive and then advances
one at a time. Both fall out of [`primitivevertices`](@ref) and
[`primitivestride`](@ref) rather than being a third table to keep in step.

**Fewer vertices than one primitive needs is a draw of NOTHING, not an error.** A
`lines` plot of a single point assembles no segment, and a pipeline asked for it
must submit zero work rather than throw at the caller: the count is clamped at
zero.
"""
function primitivecount end

function primitivecount(t::Topology, nvertices::Integer)
    stride = primitivestride(t)
    lead = primitivevertices(t) - stride      # the window a strip must fill
    return max(0, (Int(nvertices) - lead) ÷ stride)
end

"""
    firstinputvertex(t::Topology, primitive) -> same type as `primitive`

The index of `primitive`'s first vertex in the input stream.

The inverse of [`primitivecount`](@ref)'s walk, and [`primitivestride`](@ref) is
what makes it one line: primitive `p` starts at `(p - 1) * stride + 1`. Both
indices are ONE-based, as the vertex stream is everywhere else here.

Returns the index in the type it was given, so a mesh stage handed an `Int32`
primitive id does no 64-bit arithmetic to find its vertices.
"""
function firstinputvertex end

firstinputvertex(t::Topology, primitive::Integer) =
    (primitive - one(primitive)) * oftype(primitive, primitivestride(t)) +
    one(primitive)
