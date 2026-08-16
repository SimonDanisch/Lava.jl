# A GEMM for shapes that divide NOTHING — clamped on both ends, 1.7-2.6x faster
# than what runs today.
#
# THIS IS THE POINT OF THE WHOLE TENSOR-ADDRESSING PORT, and it took five rounds
# of measuring the wrong thing to see it. `bench_tensor_gemm_full.jl` compares a
# tensor GEMM against the tuned staged kernel on 16-divisible shapes, loses 2.3x,
# and records five failed explanations for the gap. The comparison was invalid:
# the staged coopmat path is GATED on divisibility (`gemm.jl:1379` requires
# M, N and K all `% GEMM_TILE == 0`), and any other shape falls to a scalar
# kernel that never touches the tensor cores.
#
#     1280 x 1536 x 1280  (all %16)      0.134 ms   37.64 TF/s
#     1288 x 1544 x 1288  (none %16)     0.875 ms    5.86 TF/s     6.43x
#
# A kernel is only slow relative to what would otherwise run.
#
# ── MEASURED, interleaved arms at the SAME ragged shape. Comparing a ragged
# tensor GEMM against an ALIGNED `mul!` would be the same wrong-baseline mistake
# inverted, so both arms always get identical extents.
#
#     1288x1544x1288   mul! 1.037 ms  4.94 TF/s | tensor 0.499 ms 10.26 TF/s | 2.08x
#     1288x1536x1280   mul! 0.760 ms  6.66 TF/s | tensor 0.441 ms 11.49 TF/s | 1.72x
#     1000x1000x1000   mul! 0.437 ms  4.57 TF/s | tensor 0.169 ms 11.85 TF/s | 2.59x
#
# Correctness: 40/40 random shapes with each extent in 17..400, worst relative
# error 1.71e-06; edge cases 1x1x1, 1x257x3, 17x1x19, 16x16x15, 15x16x16 and
# 3x5x7 all correct.
#
# ── BUT `mul!` IS NOT THE ONLY BASELINE, and the second one is much closer.
# Where padding is possible Lava already pads (`gemm_padn`, `GEMM_BLOCK`), so
# "pad to 16 and use the fast kernel" is the baseline that decides whether this
# kernel is worth routing. Padding cost included in the timing:
#
#     both operands padded per call    ragged is 1.30x / 1.30x / 1.09x faster
#     WEIGHTS padded once at load      ragged is 1.19x / 1.19x faster at the two
#                                      large shapes and 0.85x — i.e. LOSES — at
#                                      200x216x184
#
# AND AT THE SHAPES OUR MODELS ACTUALLY HAVE, PADDING WINS. Whisper's N = 1500
# is padded to 1504 today — a 0.27% pad, so the fast kernel does almost no extra
# work at 37 TF/s against this kernel's ~11:
#
#     1280x1500x1280   pad+fast 0.394 ms | ragged 0.625 ms   ragged 1.59x SLOWER
#     1280x1500x 512   pad+fast 0.167    | ragged 0.188      ragged 1.13x SLOWER
#      512x1500x 512   pad+fast 0.072    | ragged 0.140      ragged 1.94x SLOWER
#
# So the earlier "1.7-2.6x win" is real only against the SCALAR fallback, which
# runs exactly where nobody bothered to pad. Against padding it wins only when
# the pad is expensive relative to the compute — all three extents ragged AND
# both operands re-padded every call. That is not the common case, and it is not
# Whisper's.
#
# **This is the third baseline in one sitting and the third different verdict**
# (2.3x slower than staged-on-aligned; 2.1x faster than the scalar fallback;
# 1.6x slower than pad-and-pad-cheaply). None of them is wrong. The kernel is
# fixed and the question keeps changing, which is the whole lesson: name the
# baseline before quoting a ratio, and prefer the one the real input would
# actually take.
#
# WHAT THIS KERNEL IS THEREFORE FOR, on the evidence:
#   * shapes where padding is impractical or the pad fraction is large
#   * removing the padding machinery's VRAM and its copies, not its FLOPs
#   * NOT a general replacement for pad-then-fast, and not a Whisper win
#
# CAVEAT: everything above was measured with a test suite loading the same GPU.
# Interleaved ratios survive that, but the padded arm is several dispatches
# (fill, two copies, GEMM, slice) against this kernel's one, and multi-dispatch
# arms need not degrade identically under contention. Re-measure on an idle
# device before turning any of this into a routing threshold.
#
# ── TWO BUGS THIS KERNEL EXISTS BECAUSE OF.
#
# **The k offset must NOT ride in the base pointer once clamping matters.**
# `bench_tensor_gemm_full.jl` records "the k offset does not belong in the SLICE
# … worth 1.45x" as the port's one real optimisation. That is true only while
# every extent divides the tile. A layout describes memory *relative to the base
# pointer*, so advancing the base by `k0` makes its declared K extent a lie: the
# final partial k-tile then reads past the end instead of clamping to zero.
#
#     shape          k in pointer   k in slice
#     200x216x184      2.5e-01        8.7e-07
#      37x 53x 71      2.8e-01        3.6e-07
#     128x256x184      9.7e-07        7.6e-07     <- passes anyway
#
# 128x256x184 passing is why this is dangerous rather than merely wrong: the bug
# is shape-dependent and can look correct. A dirty-the-pool control did NOT
# reproduce a mechanism for the pass, so *why* that shape survives is unexplained
# — do not assume it is benign anywhere else.
#
# **The tensor store TRANSPOSES relative to `copyto!`.** Which is convenient: the
# destination becomes an ordinary `(M, N)` array described by layout dims
# `(N, M)`, and the transposed `Ct` the copyto! path forced disappears — this
# kernel computes `C = A*B` directly. Pinning that needed a NON-SQUARE shape. At
# 256x256x256 a wrong orientation measured 1.24e-06 "correct", because a square
# buffer cannot distinguish `(N,M)` from `(M,N)`; the same trap the load's
# orientation test was written to avoid.
using Lava, KernelAbstractions, Printf, Statistics, LinearAlgebra
const KA = KernelAbstractions
const AMr = Lava.AcceleratedMatrix
const TR = 16
const WN_R = 128       # workgroup tile along N
const WM_R = 64        # workgroup tile along M

"""
    tgemm_ragged!(C, A, B, M, N, K)

`C = A*B` for ANY extents. A is `(M,K)`, B is `(K,N)`, C is `(M,N)`, fp16 in,
fp32 accumulate. Nothing is padded and nothing has to divide: both operands are
read through a `TENSOR_CLAMP_CONSTANT` layout, which returns exact zeros out of
range (a zero contributes nothing to a dot product), and the result goes back
through `tensor_store`, which clamps the write the same way.

Four subgroups, each owning a 4x2 block of 16x16 tiles. The k offset is in the
SLICE, deliberately — see the header.
"""
@kernel cpu = false unsafe_indices = true function tgemm_ragged!(C, @Const(A), @Const(B),
                                                                 M::Int32, N::Int32, K::Int32)
    tid = @index(Local, Linear) - 1
    grp = @index(Group, Linear) - 1
    sg = Int32(tid ÷ 32)
    nbn = (N + Int32(WN_R) - Int32(1)) ÷ Int32(WN_R)
    bn = Int32(grp) % nbn
    bm = Int32(grp) ÷ nbn
    sn = sg % Int32(2)
    sm = sg ÷ Int32(2)
    p0 = bn * Int32(WN_R) + sn * Int32(4 * TR)      # origin in N
    q0 = bm * Int32(WM_R) + sm * Int32(2 * TR)      # origin in M

    # Dims are REVERSED against the Julia array shape — the tensor's last
    # dimension is the fastest-varying one, and these are column-major.
    lb = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)), (N, K))
    la = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)), (K, M))
    lc = Lava.tensor_setdim(Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)), (N, M))
    zb = Lava.coopmat_zero(AMr{Float16,TR,TR,Lava.MatrixA})
    za = Lava.coopmat_zero(AMr{Float16,TR,TR,Lava.MatrixB})
    Z() = Lava.coopmat_zero(AMr{Float32,TR,TR,Lava.Accumulator})
    Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 2 j -> (acc_i_j = Z())

    pb = UInt64(pointer(B))
    pa = UInt64(pointer(A))
    nk = (K + Int32(TR) - Int32(1)) ÷ Int32(TR)
    for kk in Int32(0):(nk - Int32(1))
        ko = kk * Int32(TR)
        Base.Cartesian.@nexprs 4 i -> (b_i = Lava.tensor_load(zb, pb,
            Lava.tensor_slice(lb, (p0 + Int32((i - 1) * TR), ko), (Int32(TR), Int32(TR)))))
        Base.Cartesian.@nexprs 2 j -> (a_j = Lava.tensor_load(za, pa,
            Lava.tensor_slice(la, (ko, q0 + Int32((j - 1) * TR)), (Int32(TR), Int32(TR)))))
        Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 2 j ->
            (acc_i_j = Lava.coopmat_muladd(b_i, a_j, acc_i_j))
    end

    pc = UInt64(pointer(C))
    Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 2 j -> Lava.tensor_store(
        acc_i_j, pc,
        Lava.tensor_slice(lc, (p0 + Int32((i - 1) * TR), q0 + Int32((j - 1) * TR)),
                          (Int32(TR), Int32(TR))))
end

"""`A*B` through the ragged kernel; returns a fresh `(M, N)` fp32 array."""
function tensor_gemm_ragged(back, A, B, M, N, K)
    C = KA.allocate(back, Float32, M, N)
    fill!(C, 0f0)
    groups = cld(N, WN_R) * cld(M, WM_R)
    tgemm_ragged!(back, (4 * 32,))(C, A, B, Int32(M), Int32(N), Int32(K);
                                   ndrange = (groups * 4 * 32,))
    C
end

"""Correctness over random ragged shapes, against a CPU reference."""
function verify_ragged(back; trials = 40, lo = 17, hi = 400)
    worst = 0.0
    for _ in 1:trials
        M, N, K = rand(lo:hi), rand(lo:hi), rand(lo:hi)
        a = Float16.(randn(Float32, M, K) .* 0.05f0)
        b = Float16.(randn(Float32, K, N) .* 0.05f0)
        Ad = KA.allocate(back, Float16, M, K); copyto!(Ad, a)
        Bd = KA.allocate(back, Float16, K, N); copyto!(Bd, b)
        o = Array(tensor_gemm_ragged(back, Ad, Bd, M, N, K))
        KA.synchronize(back)
        p = Float32.(a) * Float32.(b)
        worst = max(worst, maximum(abs.(o .- p)) / maximum(abs.(p)))
    end
    worst
end
