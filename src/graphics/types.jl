# Tessellation stage configuration. `Topology` and `GeometryConfig` are the
# vocabulary's (`vocabulary/topology.jl`, `vocabulary/graphics.jl`); blend, cull,
# depth and render-target state are the runtime's.

"""How a tessellation evaluation stage spaces the generated vertices along an edge."""
abstract type TessSpacing end
"""Equal spacing: the tessellation level is rounded up to an integer."""
struct EqualSpacing          <: TessSpacing end
"""Fractional spacing with an even number of segments."""
struct FractionalEvenSpacing <: TessSpacing end
"""Fractional spacing with an odd number of segments."""
struct FractionalOddSpacing  <: TessSpacing end

"""The winding order of the triangles a tessellation evaluation stage generates."""
abstract type TessWinding end
"""Clockwise winding."""
struct WindingCW  <: TessWinding end
"""Counter-clockwise winding."""
struct WindingCCW <: TessWinding end

"""The domain a tessellation evaluation stage subdivides."""
abstract type TessDomain end
"""Triangle domain: `tess_coord_uvw` is barycentric."""
struct TessTriangles <: TessDomain end
"""Quad domain: `tess_coord` is a point in the unit square."""
struct TessQuads     <: TessDomain end
"""Isoline domain: sets of lines across the unit square."""
struct TessIsolines  <: TessDomain end

"""
    TessConfig(; vertices = 3, spacing = EqualSpacing(), winding = WindingCCW(),
                 domain = TessTriangles())

The execution modes of a tessellation stage pair: `vertices` per input patch and how
the evaluation stage spaces, winds and parameterizes what it generates. Passed as
`config` to [`lava_compile`](@ref) for `:tess_control` and `:tess_eval`.
"""
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
