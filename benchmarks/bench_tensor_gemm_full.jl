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
# ── WHY IT IS SLOWER. Five explanations were tested; the first four FAILED a
# predictive test and only the fifth survived a negative control.
#
#   register pressure   2x2 uses 72 regs, staged 124     FAILED (mine lower)
#   occupancy           2x2 gets 7 wg/SM, staged 2       FAILED (mine higher)
#   instruction count   2x2 296 insts, staged 575        FAILED (mine simpler)
#   global traffic      predicted 4x4 at 0.50x           FAILED (measured 1.53x SLOWER)
#   ADDRESSING REGISTERS  ~20 per live slice, measured   SURVIVED its control
#
# The traffic explanation is the one an earlier revision of this file asserted as
# the answer. It is wrong and this is the retraction. Traffic per output falls
# monotonically across the variants while time does not:
#
#     kernel                 ms     TF/s   traffic/out   regs   wg/SM
#     staged (256 thr)    0.132     38.1        0.0234    124     2.1
#     tensor 2x2          0.369     13.6        0.0625     72     7.1
#     tensor 4x2          0.310     16.3        0.0469    255     2.0
#     tensor 4x4          0.790      6.4        0.0312    255     1.0
#
# 0.0625 -> 0.0469 -> 0.0312 against 0.369 -> 0.310 -> 0.790. A quantity that
# decreases while the time it supposedly explains goes up is not the cause.
#
# WHAT IS ACTUALLY GOING ON: a live tensor slice is enormously expensive in
# registers, so the tile cannot grow to where the reuse would be.
#
#     hoisted slices  1     2     4     6     8      (1 accumulator, 32 threads)
#     registers      81    80   133   168   255      -> ~20 per live slice
#
# and the NEGATIVE CONTROL that makes this an attribution rather than a
# correlation — same number of loads per iteration, but all through ONE slice
# with the offset in the pointer, so only "loads in flight" varies:
#
#     loads/iter      1     2     4     6     8
#     registers      81    82    84    85    87      -> ~0.75 per load
#
# Slices cost 27x what loads cost. For scale, an entire 16x16 fp16 coopmat
# operand is 4 registers per lane; one slice descriptor costs five of them. The
# fixed layout machinery is a further ~60-80 registers before any tile exists.
#
# That is a squeeze with no way through at 16x16:
#   * beating the staged kernel needs operand reuse, which needs a big tile
#   * a big tile out of KHR 16x16 tiles needs many live slices
#   * ~20 registers each saturates the 255-register cap at 6-8 slices, which is
#     reached BEFORE the tile is large enough to supply the reuse
#   * recomputing slices in the loop instead costs 168 regs but 1.197x the time
#     (measured), so neither side of the trade wins
#   * 4x2 at 8 subgroups is 1.157x SLOWER than at 4 (measured): at 255 regs,
#     256 threads gets 1 workgroup/SM
#
# So tensor addressing and LDS staging are NOT alternatives. The tensor load is a
# better way to MOVE a tile (measured 0.993x of a plain coopmat load, i.e. free);
# the reuse still has to come from somewhere — shared memory, or a tile big
# enough that one subgroup's registers supply it.
#
# THIS ALSO NARROWS `bench_tensor_staging.jl`, which measured "dropping the
# staging wins ~1.8x": four subgroups sharing ONE 16x16 block is a reuse factor
# of 4 with an L1-resident working set. A real GEMM tile has reuse 8-16 and does
# not fit. A micro-benchmark measures its own regime.
#
# THE PREDICTION THIS MAKES, and the reason #44 is now the next step on evidence
# rather than on analogy to llama.cpp: coopmat2 FLEXIBLE DIMENSIONS lets one
# cooperative matrix cover 64x16, so the same tile area needs ONE slice where
# this kernel needs four. If ~20 registers per slice is really the binding
# constraint, that quarters the addressing cost at constant tile size and is the
# only lever measured here that moves it. `bench_tensor_registers.jl` holds the
# two sweeps above so the claim can be re-checked when that lands.
