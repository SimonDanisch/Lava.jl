# Graphics pipeline configuration types for Lava.jl
#
# Pure Julia types — no Vulkan dependency, used for dispatch everywhere.
# Pipeline state encoded in type parameters for zero-cost dispatch.

# ── Shader Stage Types ──

abstract type ShaderStage end
struct VertexStage      <: ShaderStage end
struct FragmentStage    <: ShaderStage end
struct GeometryStage    <: ShaderStage end
struct TessControlStage <: ShaderStage end
struct TessEvalStage    <: ShaderStage end

# ── Blend Modes ──

abstract type BlendMode end
struct Opaque        <: BlendMode end   # no blending
struct AlphaBlend    <: BlendMode end   # src.a * src + (1-src.a) * dst
struct Additive      <: BlendMode end   # src + dst
struct Premultiplied <: BlendMode end   # premultiplied alpha

# ── Culling ──

abstract type CullFace end
struct NoCull    <: CullFace end
struct CullBack  <: CullFace end
struct CullFront <: CullFace end

# ── Input Topology ──

abstract type Topology end
struct TriangleList  <: Topology end
struct TriangleStrip <: Topology end
struct LineList      <: Topology end
struct LineStrip     <: Topology end
struct PointList     <: Topology end
struct PatchList          <: Topology end  # for tessellation
# Both feed a geometry shader four vertices per primitive. They differ in how the
# index buffer is walked: the LIST form consumes a disjoint group of 4 per
# primitive, the STRIP form slides a 4-wide window one index at a time. An
# adjacency index list built for a strip (Makie's polylines: `0 0 1 2 3 3`)
# yields ⌊n/4⌋ primitives under the list form instead of n-3 — a polyline drawn
# as scattered dashes.
struct LineListAdjacency  <: Topology end
struct LineStripAdjacency <: Topology end

# ── Depth Test ──

abstract type DepthMode end
struct DepthLess     <: DepthMode end   # default: closer wins
struct DepthLessEq   <: DepthMode end
struct DepthGreater  <: DepthMode end
struct DepthAlways   <: DepthMode end
struct DepthOff      <: DepthMode end   # no depth test/write

# ── Geometry Shader Config ──

struct GeometryConfig
    input_topology::Topology
    output_topology::Topology
    max_vertices::Int
    invocations::Int
end

function GeometryConfig(; input::Topology=TriangleList(), output::Topology=TriangleStrip(),
                          max_vertices::Integer=3, invocations::Integer=1)
    GeometryConfig(input, output, max_vertices, invocations)
end

# ── Tessellation Config ──

abstract type TessSpacing end
struct EqualSpacing          <: TessSpacing end
struct FractionalEvenSpacing <: TessSpacing end
struct FractionalOddSpacing  <: TessSpacing end

abstract type TessWinding end
struct WindingCW  <: TessWinding end
struct WindingCCW <: TessWinding end

abstract type TessDomain end
struct TessTriangles <: TessDomain end
struct TessQuads     <: TessDomain end
struct TessIsolines  <: TessDomain end

struct TessConfig
    patch_vertices::Int
    spacing::TessSpacing
    winding::TessWinding
    domain::TessDomain
end

function TessConfig(; vertices::Integer=3, spacing::TessSpacing=EqualSpacing(),
                      winding::TessWinding=WindingCCW(), domain::TessDomain=TessTriangles())
    TessConfig(vertices, spacing, winding, domain)
end

# ── Render Target ──

abstract type RenderTarget end
