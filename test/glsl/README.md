# GLSL differentials

Hand-written GLSL whose SPIR-V is produced by `glslangValidator`, **not** by
Lava's emitter, and run through Lava's own dispatch by installing the module
under a kernel's frozen-cache key. This is Rule 0's strongest instrument
(`../../spirv-intrinsics.md`): if glslang's module is correct and ours is not,
the bug is ours; if both are wrong on one driver and both are right on another,
it is not ours.

Each shader takes Lava's argument buffer as its only push constant (one 64-bit
address) and reads the same struct offsets the Julia-derived module reads, so
it is a drop-in replacement for the kernel it mirrors.

  narrow_index.comp        the rank-3 `cis[I % Int32]` broadcast kernel of
                           `../test_int32_cartesian_miscompile.jl`
  narrow_index_probe.comp  the same, reading back `q1`, `q2` and `c2` computed
                           twice — once in Int64, once in Int32 — in a single
                           dispatch, which is what located the fault

Build:

    glslangValidator --target-env vulkan1.3 -S comp narrow_index.comp -o /tmp/x.spv

Result on 2026-08-02 — RTX 4000 Ada 595.336 WRONG, lavapipe/LLVM 22.1.8 exact,
for both the Lava module and the glslang module. See
`dev/JuliaVision/plans/projects/lava-core/REPORT.md`.
