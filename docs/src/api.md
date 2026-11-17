# API reference

Exported names are reachable after `using Lava`. The device vocabulary has its own
[reference page](vocabulary_api.md).

## Compiling

```@docs
lava_compile
CompilationResult
LavaGfxShader
LavaDeviceArray
@lava_printf
```

## Stage configuration

```@docs
Topology
TriangleList
TriangleStrip
LineList
LineStrip
PointList
PatchList
LineListAdjacency
LineStripAdjacency
GeometryConfig
TessConfig
TessSpacing
EqualSpacing
FractionalEvenSpacing
FractionalOddSpacing
TessWinding
WindingCW
WindingCCW
TessDomain
TessTriangles
TessQuads
TessIsolines
```

## Stage intrinsics

```@docs
set_position!
gfx_output
gfx_output_flat
gfx_input
gfx_input_flat
front_facing
invocation_id
geom_input
geom_input_position
tess_coord
tess_coord_uvw
set_tess_level_outer!
set_tess_level_inner!
GfxTexture2D
VertexWrapper
FragmentWrapper
GeometryWrapper
```
