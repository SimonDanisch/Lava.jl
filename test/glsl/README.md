# GLSL references

Hand-written GLSL compiled with `glslangValidator`, not with Lava. Each file is the
reference for an operation Lava emits: when glslang's module computes the right
result on a driver and Lava's does not, the difference is in Lava's module; when
both are wrong on one driver and right on the others, it is the driver.

The compute shaders take Lava's argument buffer as their only push constant (one
64-bit address) and read the same offsets as the Julia kernel they mirror, so a
runtime can install either module for the same dispatch.

| file | reference for |
|---|---|
| `narrow_index.comp` | a rank-3 broadcast indexed with `Int32` Cartesian indices |
| `narrow_index_probe.comp` | the same, computing the index arithmetic in `Int64` and `Int32` side by side |
| `coopmat2_gemm_regs.comp`, `coopmat2_gemm_regs_unrolled.comp` | the register use of a `VK_NV_cooperative_matrix2` GEMM, against Lava's module of the same algorithm |
| `workgroup_scope_coopmat.comp` | workgroup-scope cooperative matrix types and operations |
| `tensor_addressing_opcodes.comp` | the opcodes and operands of `SPV_NV_tensor_addressing` |

Build one with:

    glslangValidator --target-env vulkan1.3 -S comp narrow_index.comp -o narrow_index.spv
