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

# ── Pipeline state: not here any more ──
#
# `BlendMode`, `CullFace` and `DepthMode` are Mantle's, in
# `Mantle/src/graphics/state.jl`. They are fixed-function rasterizer state, and
# this compiler never dispatched on any of them — the only use outside this file
# was the `export` line.
#
# `Topology` IS dispatched on here, by `geometry_input_vertex_count` and
# `geometry_input_mode` in `compiler/spirv/graphics.jl`, and a backend dispatches
# on it too when it creates a pipeline. Two owners means neither, so it is
# `KernelInterface`'s now, beside `MatrixShape` and `DeviceCaps` — a Metal
# backend can name `TriangleList` without importing a SPIR-V compiler.
using KernelInterface: Topology, TriangleList, TriangleStrip, LineList,
                       LineStrip, PointList, PatchList, LineListAdjacency,
                       LineStripAdjacency

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
#
# `RenderTarget` is Mantle's, in `graphics/state.jl`. It had no use anywhere in
# this package — what a pipeline renders INTO is the runtime's concern, and the
# concrete targets (a window's swapchain image, an offscreen framebuffer) are
# each a backend's.
