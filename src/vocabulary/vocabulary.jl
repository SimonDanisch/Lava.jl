# The device vocabulary: what a kernel names to reach matrix hardware, tensor
# addressing, the graphics and mesh stages and the device's capabilities, beyond what
# KernelInterface defines. Each function is declared here without a method a kernel
# could reach; a backend implements it with an overlay in its method table: Lava's
# compiler with `@lava_device_override`, Mantle's Metal backend in `Metal.method_table`.
#
# Outside the `HAS_LOADER` block in `Lava.jl`: Mantle names this vocabulary on every
# platform, also where there is no Vulkan loader and Lava builds no compiler.

include("matrix.jl")
include("caps.jl")      # after `matrix.jl`: `DeviceCaps` holds a `Vector{MatrixShape}`
include("coopmat.jl")   # the cooperative-matrix type and the eleven operations
include("tensor.jl")    # loading and storing a matrix through a description of the array
include("topology.jl")
include("graphics.jl")  # the stage builtins a shader body calls
include("mesh.jl")      # after `graphics.jl`: its emitter reaches the geometry builtins

"""
    dot4x8(a::UInt32, b::UInt32)::Int32

The dot product of the four signed 8-bit integers packed in `a` with the four packed in
`b`, byte `i` of one with byte `i` of the other, exact in 32 bits.

GPUs have an instruction for it (SPIR-V's `OpSDot` with the packed 4x8 format, AMD's
`v_dot4`, NVIDIA's `dp4a`): four multiply-adds in one, which is what an int8
matrix-vector product spends its time on.

The definition here is the reference and runs anywhere. A backend with the instruction
overrides it.
"""
@inline function dot4x8(a::UInt32, b::UInt32)
    byte(x, i) = Int32((x >> (8 * i)) % Int8)
    return byte(a, 0) * byte(b, 0) + byte(a, 1) * byte(b, 1) +
           byte(a, 2) * byte(b, 2) + byte(a, 3) * byte(b, 3)
end

public dot4x8,
    # matrices and capabilities
    MatrixUse, MatrixScope, SubgroupScope, WorkgroupScope, MatrixShape, matrix_shapes,
    supports, bestshape, DeviceCaps, caps, wggranularity,
    CoopMatrix, AcceleratedMatrix, WorkgroupMatrix, matrixuse, matrixscope,
    coopmat_load, coopmat_store, coopmat_muladd, coopmat_mul, coopmat_add, coopmat_zero,
    coopmat_undef, coopmat_convert, coopmat_length, coopmat_getcomp, coopmat_setcomp,
    coopmat_perelement, coopmat_reduce, CoopMatReduce,
    supports_coopmat_perelement, supports_coopmat_reduce, supports_flexible_coopmat_shapes,
    # tensor addressing
    TensorLayout, TensorView, TENSOR_CLAMP_UNDEFINED, TENSOR_CLAMP_CONSTANT,
    TENSOR_CLAMP_TO_EDGE, tensor_layout, tensor_setdim, tensor_setstride,
    tensor_setclampvalue, tensor_clampbits, tensor_slice, tensor_view, tensor_load,
    tensor_store, supports_tensor_addressing,
    # graphics stages
    vertex_index, VertexIndex, wantsvertexindex, instance_index, frag_coord, frag_coord_x,
    frag_coord_y, frag_coord_z, frag_coord_w, frag_coord_xy, dFdx, dFdy, discard,
    set_point_size!, sample_texture_2d, clip_y, emit_vertex!, end_primitive!,
    primitive_id_in,
    # the mesh pipeline
    MeshConfig, ObjectConfig, PrimitiveEmitter, NativeEmitter, MeshEmitter, emit!,
    endprimitive!, finish!, set_mesh_vertex!, set_mesh_triangle!, set_mesh_line!, set_mesh_point!,
    set_mesh_primitive_data!, set_mesh_outputs!, set_mesh_groups!, mesh_thread_index,
    mesh_group_index, nvertices, nprimitives, Flat, isflat, unflat, flatnames, smoothnames,
    valuetypes, inputvertices
