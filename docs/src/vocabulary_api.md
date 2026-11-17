# Vocabulary reference

The [device vocabulary](vocabulary.md) is public and not exported: write `Lava.name`,
or use Mantle, which re-exports what kernels and shaders call.

## Stage builtins

```@docs
Lava.vertex_index
Lava.instance_index
Lava.VertexIndex
Lava.wantsvertexindex
Lava.frag_coord
Lava.frag_coord_x
Lava.frag_coord_y
Lava.frag_coord_z
Lava.frag_coord_w
Lava.frag_coord_xy
Lava.dFdx
Lava.dFdy
Lava.discard
Lava.set_point_size!
Lava.sample_texture_2d
Lava.clip_y
Lava.emit_vertex!
Lava.end_primitive!
Lava.primitive_id_in
```

## Mesh pipeline

```@docs
Lava.MeshConfig
Lava.ObjectConfig
Lava.MeshEmitter
Lava.NativeEmitter
Lava.PrimitiveEmitter
Lava.emit!
Lava.endprimitive!
Lava.finish!
Lava.set_mesh_vertex!
Lava.set_mesh_triangle!
Lava.set_mesh_line!
Lava.set_mesh_point!
Lava.set_mesh_primitive_data!
Lava.set_mesh_outputs!
Lava.set_mesh_groups!
Lava.mesh_thread_index
Lava.mesh_group_index
Lava.nvertices
Lava.nprimitives
Lava.inputvertices
Lava.Flat
Lava.isflat
Lava.unflat
Lava.flatnames
Lava.smoothnames
Lava.valuetypes
```

## Cooperative matrices

```@docs
Lava.MatrixUse
MatrixA
MatrixB
Accumulator
Lava.MatrixScope
Lava.SubgroupScope
Lava.WorkgroupScope
Lava.MatrixShape
Lava.CoopMatrix
Lava.AcceleratedMatrix
Lava.WorkgroupMatrix
Lava.matrixuse
Lava.matrixscope
Lava.coopmat_load
Lava.coopmat_store
Lava.coopmat_muladd
Lava.coopmat_mul
Lava.coopmat_add
Lava.coopmat_zero
Lava.coopmat_undef
Lava.coopmat_convert
Lava.coopmat_length
Lava.coopmat_getcomp
Lava.coopmat_setcomp
Lava.coopmat_perelement
Lava.coopmat_reduce
Lava.CoopMatReduce
Lava.dot4x8
```

## Tensor addressing

```@docs
Lava.TensorLayout
Lava.TensorView
Lava.TENSOR_CLAMP_UNDEFINED
Lava.TENSOR_CLAMP_CONSTANT
Lava.TENSOR_CLAMP_TO_EDGE
Lava.tensor_layout
Lava.tensor_setdim
Lava.tensor_setstride
Lava.tensor_setclampvalue
Lava.tensor_clampbits
Lava.tensor_slice
Lava.tensor_view
Lava.tensor_load
Lava.tensor_store
```

## Device capabilities

```@docs
Lava.DeviceCaps
Lava.caps
Lava.matrix_shapes
Lava.wggranularity
Lava.supports
Lava.bestshape
Lava.supports_coopmat_perelement
Lava.supports_coopmat_reduce
Lava.supports_flexible_coopmat_shapes
Lava.supports_tensor_addressing
```
