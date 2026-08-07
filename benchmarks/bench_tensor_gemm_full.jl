# A real tiled GEMM through tensor addressing, raced against the staged kernel.
#
# Everything it needs was measured first, so none of this is guesswork:
#   * layout dims go in REVERSED for a column-major array (last dim is fastest)
#   * slice offsets are in the same order as `setdim`'s dims
#   * `muladd(load(P), load(Q)) == P' * Q'`, so loading B and A in that order
#     gives `(A*B)'` — the kernel therefore computes Ct and stores it as (N, M)
#   * a clamping layout bounds-checks, so nothing has to divide the tile
#
# Tiling: 64x64 of Ct per workgroup, 4 subgroups in a 2x2 grid, each owning a
# 2x2 block of 16x16 sub-tiles. Per k-step a subgroup loads 2 B-tiles and 2
# A-tiles and issues 4 muladds — one muladd per load, which is the ratio that
# makes the per-block slice amortise (measured 1.064x at 16, 3.24x at 1).
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

const TT = 16          # coopmat tile
const SUB_M = 2        # sub-tiles per subgroup, M direction
const SUB_N = 2
const NSUBG = 4        # subgroups per workgroup, as 2x2
const WTILE = 64       # 2 subgroups * 2 sub-tiles * 16

"""Ct = (A*B)' via tensor addressing. A is (M,K), B is (K,N), Ct is (N,M)."""
@kernel cpu = false unsafe_indices = true function tgemm!(Ct, @Const(A), @Const(B),
                                                          M::Int32, N::Int32, K::Int32)
    tid = @index(Local, Linear) - 1
    grp = @index(Group, Linear) - 1
    sg = tid ÷ 32                       # subgroup within the workgroup
    nblocks = (N + Int32(WTILE) - Int32(1)) ÷ Int32(WTILE)
    bn = Int32(grp) % nblocks           # block along N
    bm = Int32(grp) ÷ nblocks           # block along M
    sn = Int32(sg) % Int32(2)
    sm = Int32(sg) ÷ Int32(2)
    p0 = bn * Int32(WTILE) + sn * Int32(TT * SUB_N)   # origin in N
    q0 = bm * Int32(WTILE) + sm * Int32(TT * SUB_M)   # origin in M

    # Layouts are loop-INVARIANT; only the slices move. Building them once is the
    # thing the primitive benchmark said to do.
    lb = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                            (N, K))                    # B is (K,N) -> dims (N,K)
    la = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
                            (K, M))                    # A is (M,K) -> dims (K,M)
    zb = Lava.coopmat_zero(AMg{Float16,TT,TT,Lava.MatrixA})
    za = Lava.coopmat_zero(AMg{Float16,TT,TT,Lava.MatrixB})

    acc00 = Lava.coopmat_zero(AMg{Float32,TT,TT,Lava.Accumulator})
    acc01 = Lava.coopmat_zero(AMg{Float32,TT,TT,Lava.Accumulator})
    acc10 = Lava.coopmat_zero(AMg{Float32,TT,TT,Lava.Accumulator})
    acc11 = Lava.coopmat_zero(AMg{Float32,TT,TT,Lava.Accumulator})

    nk = (K + Int32(TT) - Int32(1)) ÷ Int32(TT)
    for kk in Int32(0):(nk - Int32(1))
        ko = kk * Int32(TT)
        b0 = Lava.tensor_load(zb, UInt64(pointer(B)),
                 Lava.tensor_slice(lb, (p0, ko), (Int32(TT), Int32(TT))))
        b1 = Lava.tensor_load(zb, UInt64(pointer(B)),
                 Lava.tensor_slice(lb, (p0 + Int32(TT), ko), (Int32(TT), Int32(TT))))
        a0 = Lava.tensor_load(za, UInt64(pointer(A)),
                 Lava.tensor_slice(la, (ko, q0), (Int32(TT), Int32(TT))))
        a1 = Lava.tensor_load(za, UInt64(pointer(A)),
                 Lava.tensor_slice(la, (ko, q0 + Int32(TT)), (Int32(TT), Int32(TT))))
        acc00 = Lava.coopmat_muladd(b0, a0, acc00)
        acc01 = Lava.coopmat_muladd(b0, a1, acc01)
        acc10 = Lava.coopmat_muladd(b1, a0, acc10)
        acc11 = Lava.coopmat_muladd(b1, a1, acc11)
    end

    # Ct is (N, M) column-major: element (p, q) lives at 1 + p + q*N.
    base = pointer(Ct)
    Lava.copyto!(base, 1 + p0 + q0 * N, N, acc00)
    Lava.copyto!(base, 1 + p0 + (q0 + Int32(TT)) * N, N, acc01)
    Lava.copyto!(base, 1 + (p0 + Int32(TT)) + q0 * N, N, acc10)
    Lava.copyto!(base, 1 + (p0 + Int32(TT)) + (q0 + Int32(TT)) * N, N, acc11)
end

"""Run the tensor GEMM for A (M,K), B (K,N); returns Ct as an (N,M) device array."""
function tensor_gemm(back, A, B, M, N, K)
    Ct = KA.allocate(back, Float32, N, M)
    fill!(Ct, 0f0)
    nb = (cld(N, WTILE), cld(M, WTILE))
    groups = nb[1] * nb[2]
    tgemm!(back, (NSUBG * 32,))(Ct, A, B, Int32(M), Int32(N), Int32(K);
                                ndrange = (groups * NSUBG * 32,))
    Ct
end

# ── MEASURED, Whisper's qkv/out shape M=1280 N=1536 K=1280, interleaved, many
# dispatches per sync so the card stays clocked (a per-call sync measures the
# ramp: the same staged kernel reads 3.3 TF/s cold and 40.5 warm).
#
#     staged (Lava, tuned)        0.124 ms/call   40.5 TF/s   38% of ceiling
#     tensor, slice in k-loop     0.406           12.4        12%
#     tensor, k-offset in ptr     0.280           18.0        17%
#
# THE PORT WORKS AND IS CORRECT (rel err 3e-7 against CPU) AND IS 2.25x SLOWER.
#
# The one real optimisation found so far: the k offset does not belong in the
# SLICE. B is (K,N) column-major so k is its fast axis, and A is (M,K) so k is a
# plain M-stride — both are expressible by advancing the base POINTER, which
# makes the slice loop-invariant. Worth 1.45x.
#
# A caution about the ablation that suggested it. Hoisting the slices while
# loading the SAME tile every iteration measured 58.9 TF/s and implied tensor
# addressing would beat staged by 3x. It would not: that arm was L1-resident, and
# the honest version — same hoisting, real addresses — is 18.0. A "what if this
# cost nothing" ablation prices the instruction AND silently deletes the memory
# traffic; it bounds the win, it does not predict it.
#
# WHAT IS LEFT, in the order the numbers point at:
#   * no reuse across subgroups. Each loads its own tiles from global where the
#     staged kernel stages once into LDS for 8 warps. The earlier staging
#     benchmark said dropping LDS wins at 16x16 granularity with 4 subgroups;
#     this kernel is where that stops being true, and it is worth re-measuring
#     rather than assuming either way.
#   * arithmetic intensity is 8 muladds / 6 loads = 1.33. Going 2x2 -> 4x2 moved
#     it 1.0 -> 1.33 and bought 12%, so this axis is real but shallow.
#   * coopmat2 FLEXIBLE DIMENSIONS, unused here. Every tile is the KHR 16x16, so
#     a 64x64 block costs 16 tiles and 6 slices; llama.cpp's `mul_mm_cm2.comp`
#     uses larger cooperative matrices and issues far fewer of both. This is the
#     largest untried lever and it is what task #44 was originally about.
#
# ── WHY IT IS SLOWER. Diagnosed, not guessed — three hypotheses killed with
# numbers before the fourth was accepted.
#
#   register pressure   mine 72 regs, staged 124        RULED OUT (mine lower)
#   occupancy           mine 7 wg/SM, staged 2          RULED OUT (mine higher)
#   instruction count   mine 296 insts, staged 575      RULED OUT (mine simpler)
#   GLOBAL TRAFFIC      mine 2.67x staged, per output   MATCHES the 2.25x time
#
# Registers and occupancy came from `VK_KHR_pipeline_executable_properties` via
# `Lava.enable_pipeline_executable_properties!()` before device creation; the
# workgroup sizes (128 vs 256) identify the two pipelines. Occupancy is
# 65536 regs/SM ÷ (threads × regs): 7 workgroups for mine, 2 for staged.
#
# The traffic figure is global elements read per output element per unit of K:
#
#     staged  (bm*bk + bk*bn) / (bm*bn) / bk  with 64x128x32  =  0.0234
#     mine    4 subgroups * 4 tiles * 256 / 64^2 / 16         =  0.0625
#
# Each of my subgroups loads its own A and B tiles from global. The staged kernel
# stages A(64x32) and B(32x128) into LDS ONCE and eight warps read them from
# there. That is the whole gap.
#
# THIS OVERTURNS THE SCOPE OF `bench_tensor_staging.jl`, which measured "dropping
# the staging wins ~1.8x" and is not wrong so much as narrow: four subgroups
# sharing ONE 16x16 block is a reuse factor of 4 on one operand with an
# L1-resident working set. A real GEMM tile has reuse 8-16 and a working set that
# does not fit. **A micro-benchmark measures its own regime.**
#
# So tensor addressing and LDS staging are NOT alternatives. The tensor load is a
# better way to MOVE a tile; the reuse still has to come from somewhere — either
# shared memory, or a tile large enough that one subgroup's own registers supply
# it. The latter is what coopmat2 FLEXIBLE DIMENSIONS buys, and that is now
# motivated by this measurement rather than by analogy to llama.cpp.
