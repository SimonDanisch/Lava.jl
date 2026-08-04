# `mul_mm.comp`: the GEMM Lava's staged kernels are ported from

Upstream: <https://github.com/ggml-org/llama.cpp>, `ggml/src/ggml-vulkan/vulkan-shaders/`.
Fetched 2026-08-03 at commit `0eb874d37445fb25cc268ad0b2f2cb07ce561b66`
(the latest commit touching `mul_mm.comp`; repo HEAD was
`ee0445c99cffbe8d920b05cad28cb055d7049c0a`).

    mul_mm.comp               the kernel: staging, both compute branches, the store
    mul_mm_funcs.glsl         `load_a_to_shmem` / `load_b_to_shmem` per source dtype
    dot_product_funcs.glsl    the `fma` chain the scalar branch accumulates with

llama.cpp is MIT, as is Lava, so this is vendored rather than merely cited.

## Why it is in the tree

`src/array/gemm.jl` was ported from this file and says so, but the file itself was
never kept, so "stay close to the reference" had nothing to be checked against.
Two constants in Lava are this shader's (`GEMM_BK = 32`, `GEMM_PAD = 4`), the
`GemmTiling` 6-tuple is its block/warp/tile decomposition, and the comment above
`GEMM_BK` records a porting bug found by diffing behaviour against it: the first
attempt held `ST` A-fragments **and** `ST` B-fragments live across a 2-way
k-unroll, where this shader keeps one of each and reloads B in the innermost
loop. That was 64 registers against a 255 cap. Losing the reference is how that
class of bug gets re-introduced.

Refresh by re-fetching at a named commit and recording it above. Do not edit
these files; they are a reference copy, not a fork.

## What has been ported

The kernel has two compute branches behind `#ifdef COOPMAT` that **share their
staging**, and both are now ported:

  * the `COOPMAT` branch, as Lava's staged cooperative-matrix GEMM;
  * the `#else` scalar branch, as `scalar_gemm_staged_kernel!`, which is `mul!`'s
    fp32 path.

The scalar one replaced `strided_gemm_kernel!`, which declared no `@localmem` at
all: one thread per output element with the K loop in global memory. Measured on
a Radeon 8060S, TFLOP/s:

    N^3      strided_gemm_kernel!    this port    fp16 coopmat
    1024                    0.533        3.067           9.654
    2048                    0.447        5.432          14.577

There was no fp32 shortcut on that hardware to reach for instead: the driver
reports 14 cooperative-matrix shapes and `FLOAT32` appears in none of them as an
A or B type, only as an accumulator, which matches AMD's documented RDNA3 WMMA
input set of f16, bf16, iu8, iu4. fp32 matrix cores exist and are in the Vulkan
spec, but on CDNA (`V_MFMA_F32_16X16X4_F32`), not on any RDNA part.

`PROTOTYPE_gemm_fp32.jl` is kept beside the shaders: the first working
transcription, before bounds handling, strides, `alpha`/`beta` and the dispatch
gate, and still on this file's own `2/4/2` tiling rather than the `2/2/2` the
sweep chose. It is the cleanest statement of the algorithm alone and useful to
diff against when the shipped kernel grows a case.

Two measurements that shaped the shipped version, both of which contradicted the
obvious guess:

  * the reference's `is_aligned && is_in_bounds` fast path is **not** a
    micro-optimisation. Carrying bounds guards and stride multiplies through the
    general path cost 4.301 -> 2.988 at 2048^3, so it is hoisted to a `Val{FAST}`
    type parameter and both arms are compiled.
  * the dispatch gate belongs on **tile count**, not on the extents. Gating on
    `M` and `N` separately sent a 64 x 1370 attention plane to the per-element
    kernel because of its 64 rows, and the staged kernel wins that shape 1.63x.
  * the shipped **tiling is not this file's default**. `WMITER/TM/TN` are spec
    constants here because llama.cpp tunes them per device; swept over nine legal
    configurations on a Radeon 8060S, `2/2/2` beats the shipped `2/4/2` by ~9% at
    a 0.4% spread. Same 16 accumulators and same shared footprint, but 2 `fma`s
    per shared load instead of 1.6. Re-sweep on new hardware rather than assuming
    either value.

## Details the scalar branch does not share with the coopmat one

Worth stating because reusing the ported constants is the obvious mistake:

    SHMEM_STRIDE   COOPMAT: BK/2 + 4      scalar: BK/2 + 1
    BK_STEP        COOPMAT: 4             scalar: 2
    BK (default)   COOPMAT: 32            scalar: 16

So Lava's `GEMM_PAD = 4` is the *cooperative-matrix* padding and is not
automatically the right value here.

The scalar branch's shape, with its shipped `BM = BN = 64`, `WM = WN = 32`,
`WMITER = 2`, `TM = 4`, `TN = 2` and a 64-wide subgroup:

    WNITER = (WM * WN) / (WARP * TM * TN * WMITER) = 1

so each invocation holds `WMITER * TM * WNITER * TN` = **16 accumulators**, reads
`WMITER * TM` = 8 A-vectors and `WNITER * TN` = 2 B-vectors of four k-values each
per step, and issues 64 `fma`s against 20 shared loads. That register block is
the entire difference from a textbook tiled GEMM, which keeps one accumulator per
thread and issues one `fma` per two shared loads. Measured on this device, a
textbook tile (NextLA.jl, pure KernelAbstractions, ran unmodified on Lava)
reaches 1.46 TFLOP/s: real against 0.44, and still an eighth of the fp16 path.

`[[unroll]]` is load-bearing and does not survive a naive port. Nothing in Lava's
pipeline runs a loop-unroll pass and `LoopUnrollPass` declines without a GPU
`TargetTransformInfo`, so every one of these loops has to become
`Base.Cartesian.@nexprs` over compile-time constants, exactly as the ported
coopmat kernel already does.
