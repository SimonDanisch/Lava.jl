# The device vocabulary

`src/vocabulary/` declares the device functions and types that kernels and shaders are
written against and that KernelInterface does not define. It has no device methods of
its own. Each backend supplies them as overlays in its method table:

* Lava, for Vulkan, with `@lava_device_override`;
* Mantle's Metal backend, in `Metal.method_table`.

A kernel written against the vocabulary compiles on every backend that implements the
parts it uses. The vocabulary loads without a Vulkan loader, so Metal-only machines can
use it.

The names are public and not exported. Mantle re-exports the ones a shader author
writes, so `using Mantle` brings them in and `using Mantle, Lava` binds each name once.

## Contents

| file | what |
|---|---|
| `matrix.jl` | `MatrixShape`, the operand positions `MatrixA`, `MatrixB`, `Accumulator`, the scopes `SubgroupScope`, `WorkgroupScope`, `supports`, `bestshape` |
| `caps.jl` | `DeviceCaps` and `caps(backend)`: cooperative-matrix shapes, subgroup width, workgroup limits, shared-memory budget, core counts |
| `coopmat.jl` | `CoopMatrix{T,M,N,Use,Scope}`, `AcceleratedMatrix`, `WorkgroupMatrix`, the `coopmat_*` operations, `coopmat_perelement`, `coopmat_reduce` |
| `tensor.jl` | `TensorLayout`, `TensorView`, `tensor_load`, `tensor_store`, clamping modes |
| `topology.jl` | `Topology` and its primitive types |
| `graphics.jl` | stage builtins: `vertex_index`, `instance_index`, `frag_coord`, `dFdx`, `dFdy`, `discard`, `set_point_size!`, `sample_texture_2d`, `clip_y`, geometry-stage `emit_vertex!`, `end_primitive!`, `primitive_id_in`, `GeometryConfig` |
| `mesh.jl` | the mesh pipeline: `MeshConfig`, `ObjectConfig`, `MeshEmitter`, `NativeEmitter`, `PrimitiveEmitter`, `emit!`, `set_mesh_*`, `Flat` outputs, and `HostMeshOutput`, which runs a mesh stage on the host |
| `vocabulary.jl` | `dot4x8`, the list of public names |

## Host queries

The capability functions dispatch on a `KernelInterface.Backend`. Mantle defines them
for its devices:

```julia
c = Lava.caps(device)              # DeviceCaps
c.shapes                           # cooperative-matrix shapes the driver lists
c.subgroup                         # subgroup width
Lava.bestshape(device, Float16, Float32)
Lava.supports_coopmat_perelement(device)
Lava.supports_tensor_addressing(device)
```

A shape being listed means the driver declares it legal. It need not be fast, and a
software rasterizer lists shapes it emulates.

## Implementing it for a backend

A backend adds device methods with its own overlay macro and host methods for its
backend type:

```julia
# device: in the backend's method table
@device_override Lava.vertex_index() = my_vertex_index_intrinsic()

# host: on the backend type
Lava.caps(b::MyBackend) = Lava.DeviceCaps(...)
Lava.supports_coopmat_perelement(::MyBackend) = false
```

Functions without a device method in a backend's table fail to compile there, with
the call chain that reached them.
