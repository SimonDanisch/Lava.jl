# Graphics stages

Lava compiles Julia functions to vertex, fragment, geometry, tessellation control,
tessellation evaluation, task and mesh stages. Mantle builds pipelines from them;
this page is about what a stage body can do and how it is compiled.

## Stage bodies

A stage body is a plain function. It reads builtins from the
[device vocabulary](vocabulary.md) and writes its results through Lava's stage
intrinsics:

```julia
using Lava, GeometryBasics

function triangle()
    i = Lava.vertex_index()                     # one-based
    x = i == Int32(1) ? -1f0 : i == Int32(2) ? 1f0 : 0f0
    y = i == Int32(3) ? 1f0 : -1f0
    set_position!(Vec4f(x, y, 0f0, 1f0))
    return nothing
end

r = lava_compile(triangle, Tuple{}; stage = :vertex)
```

Mantle's pipelines take bodies that return a `NamedTuple`, `(position = ..., color =
...)`, and read the previous stage's outputs as one argument. The stage wrappers
[`VertexWrapper`](@ref), [`FragmentWrapper`](@ref), [`GeometryWrapper`](@ref) and
`MeshWrapper` translate between the two forms: each field becomes an interface
variable at a fixed location, and a field declared `Flat` is passed without
interpolation.

## Interface and builtins

| | |
|---|---|
| position | [`set_position!`](@ref) |
| outputs and inputs by location | [`gfx_output`](@ref), [`gfx_output_flat`](@ref), [`gfx_input`](@ref), [`gfx_input_flat`](@ref) |
| geometry stage inputs | [`geom_input`](@ref), [`geom_input_position`](@ref), [`invocation_id`](@ref) |
| tessellation | [`tess_coord`](@ref), [`tess_coord_uvw`](@ref), [`set_tess_level_outer!`](@ref), [`set_tess_level_inner!`](@ref) |
| fragment | [`front_facing`](@ref), and the vocabulary's `frag_coord`, `dFdx`, `dFdy`, `discard`, `sample_texture_2d` |
| vertex | the vocabulary's `vertex_index`, `instance_index`, `set_point_size!` |

Indices a shader reads from the vocabulary (`vertex_index`, `instance_index`) count
from one, like every index in Julia. The location and vertex arguments of the stage
intrinsics count from zero, as SPIR-V does.

## Configuration

A stage whose execution modes are part of the module takes a `config`:

* geometry: `GeometryConfig(; input, output, max_vertices, invocations)`, with the
  input and output [`Topology`](@ref);
* tessellation: [`TessConfig`](@ref)`(; vertices, spacing, winding, domain)`;
* mesh: `MeshConfig`, the vertex and primitive limits and the output topology.

```julia
cfg = GeometryConfig(; input = TriangleList(), output = LineStrip(), max_vertices = 4)
lava_compile(edges, Tuple{}; stage = :geometry, config = cfg)
```

## Mesh stages

A mesh stage body emits vertices and primitives through a `MeshEmitter`
(`emit!`, `set_mesh_vertex!`, `set_mesh_triangle!`, `set_mesh_outputs!`). The same
body runs on the host through `Lava.HostMeshOutput`, which records what it emitted,
so a mesh stage can be tested without a device. Mantle lowers geometry stages onto
mesh stages on backends without a geometry stage.
