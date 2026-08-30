# `CoopMatrix`, `AcceleratedMatrix`, `WorkgroupMatrix` and their `Base`
# operations are `KernelInterface`'s, in `lib/KernelInterface/src/coopmat.jl`.
#
# They were here, and the comment that stood at the top of this file already
# argued for the move without taking it: `MatrixUse` and `MatrixScope` had gone
# to KI because "every backend has to name the same concepts", and the type is
# the same case. A Metal backend reaches matrix hardware through
# `simdgroup_matrix` and must not import a SPIR-V compiler to name the tile.
#
# What stayed is `coopmat_intrinsics.jl`: 766 lines of `llvmcall` that lower
# KI's nine `coopmat_*` operations to `OpCooperativeMatrix*`. That is this
# compiler's half, and it is the only half that was ever Vulkan's.
using KernelInterface: CoopMatrix, AcceleratedMatrix, WorkgroupMatrix,
                       matrixuse, matrixscope
import KernelInterface: coopmat_load, coopmat_store, coopmat_muladd,
                        coopmat_zero, coopmat_undef, coopmat_convert,
                        coopmat_length, coopmat_getcomp, coopmat_setcomp
