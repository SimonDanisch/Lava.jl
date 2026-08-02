# `mul!` for LavaArray.
#
# How a matrix multiply is performed on this device is Lava's business, not its
# caller's. `mul!` picks between the cooperative-matrix kernel and a scalar one
# from the operand types and the device's reported capabilities; nothing above
# this layer needs to know that tensor cores exist.
#
# Measured on an RTX 4000 Ada, 2048³, fp16 in / fp32 accumulate:
#   scalar kernel                            ~0.7 TFLOP/s
#   coopmat, one 16×16 tile per subgroup      0.79
#   coopmat, 4×4 register block              ~6.0
#   coopmat, 4×4 block + 2 subgroups/group   ~22
# Both jumps are about feeding the units, not the units themselves: the 4×4
# block makes four A-loads and four B-loads serve sixteen `muladd`s, and a
# second subgroup per workgroup gives the scheduler something to overlap the
# remaining global loads against. 4×4 is a real optimum — 4×8 and 8×4 fall to
# ~3 TFLOP/s and 8×8 to ~1.75, which is register spilling.

const GEMM_TILE = 16       # the cooperative-matrix tile this device implements
const GEMM_BLOCK = 4       # register block: 4×4 accumulators per subgroup
const GEMM_WORKGROUP = 64  # 2 subgroups; 32 loses latency hiding, 256 spills
const GEMM_MAXBLOCK = 4    # largest register block `@nexprs` is unrolled for

# ── the staged kernel ────────────────────────────────────────────────────────
#
# The kernel above loads every cooperative matrix straight from global memory,
# so with a 4×4 register block each A tile is read four times and each B tile
# four times per k-step, and both subgroups repeat the same B loads. Every
# high-performance implementation instead stages a block of A and a block of B
# into workgroup memory once and loads the matrices from there — llama.cpp's
# Vulkan backend (`ggml-vulkan/vulkan-shaders/mul_mm.comp`, MIT) does exactly
# that, and so does cuBLAS, which reaches 44.6 TFLOP/s on these shapes.
#
# **The first port of it was a wash, and the reason was measurable.** With
# `VK_KHR_pipeline_executable_properties` the driver reports, per pipeline,
# registers and workgroup memory — and workgroup memory is where NVIDIA puts a
# cooperative matrix it cannot keep in registers:
#
#                       registers   shared/WG   of which ours
#   direct 4×4 block      255 (cap)     25344         0
#   staged 2×2             96            8960      8960
#   staged 4×4            255 (cap)     48896     17664
#
# The direct kernel spills its entire 4×4 accumulator block — it declares no
# `@localmem` at all and the driver still gives it 25 KB — and the staged 4×4,
# which is exactly `mul_mm.comp`'s shipped configuration, spills 31 KB on top of
# its blocks and lands at 4.6 TFLOP/s. Neither is a tiling problem. Both are one
# structural difference from the reference: the first port held `ST` A-fragments
# **and** `ST` B-fragments live across a 2-way k-unroll, sixteen fragments at
# ST=4, where `mul_mm.comp` keeps exactly one of each and reloads B inside the
# innermost loop. Sixteen 16×16 fp16 fragments is 64 registers on top of the
# accumulator block's 128, and 255 is the cap.
const GEMM_BK = 32
const GEMM_PAD = 4                  # bank-conflict padding, in elements

"""
A staged-GEMM tiling: `(STM, STN, WM, WN, BK, PAD)`.

The three levels of `mul_mm.comp`'s block -> warp -> tile, as parameters rather
than as constants, because which one wins is a property of the device and not of
the algorithm — and because the previous version hardcoded its `@nexprs` counts
to the value it shipped with, so the "4 measured worse" recorded beside the
constant had never actually run a 4×4 warp tile.

  * `WM × WN` subgroups per workgroup, arranged with `WM` down the M axis;
  * each subgroup owns `STM × STN` cooperative-matrix tiles;
  * so the workgroup covers `BM = 16*STM*WM` by `BN = 16*STN*WN` of C, walking K
    in steps of `BK`, and its shared blocks carry `PAD` elements of
    leading-dimension padding against bank conflicts.

Arithmetic intensity is what the warp tile buys: per k-step a workgroup stages
`(BM + BN) * BK` elements and does `BM * BN * BK * 2` flops, so widening the
*warp* tile raises the ratio without touching the staging cost at all — provided
the accumulators still fit in registers, which is the whole difficulty.
"""
const GemmTiling = NTuple{6,Int}

"""
    splitidx(idx, ::Val{N}) -> (idx % N, idx ÷ N)

Decompose a flat staging index, **without ever emitting `OpUDiv`**.

That is a correctness requirement, not a micro-optimisation. A staging loop whose
shared-store address goes through a real division by a non-power-of-two constant
*loses stores* on this driver once the loop is unrollable and a cooperative-matrix
`muladd` is in scope: at `BM = 96`, 2184 of 3072 slots end up holding another
row's value and 392 are never written at all. The two emitted modules for
`BM = 112` (lossy) and `BM = 128` (exact) have **identical opcode sequences except
that one contains `OpUDiv` and the other `OpShiftRightLogical` + `OpBitwiseAnd`**,
which is what identified the division. Replacing it with the magic-number form
below makes every geometry exact at every K.

`FastDiv32` is the same `init_fastdiv_values` port the broadcast path uses, so
this costs nothing — a high multiply and a shift are cheaper than a divide, which
is why it was ported in the first place. A power-of-two `N` keeps the mask and
shift it would have had anyway.
"""
@generated function splitidx(idx::Integer, ::Val{N}) where {N}
    ispow2(N) && return :((Int(idx) & $(N - 1), Int(idx) >> $(trailing_zeros(N))))
    fd = FastDiv32(N)
    quote
        u = UInt32(idx)
        q = Int((UInt32((UInt64(u) * $(UInt64(fd.mp))) >> 32) + u) >> $(fd.L))
        (Int(u) - q * $N, q)
    end
end

@inline gemm_bm(c::GemmTiling) = GEMM_TILE * c[1] * c[3]
@inline gemm_bn(c::GemmTiling) = GEMM_TILE * c[2] * c[4]
@inline gemm_wg(c::GemmTiling) = c[3] * c[4] * COOPMAT_SUBGROUP
@inline gemm_bk(c::GemmTiling) = c[5]
@inline gemm_lda(c::GemmTiling) = gemm_bm(c) + c[6]
@inline gemm_ldb(c::GemmTiling) = gemm_bk(c) + c[6]

"""
    GEMM_TILINGS

The tilings a kernel is generated for, **fastest first** — `gemm_tiling` takes
the first whose block divides the shape, so the order is the preference and the
last entries exist for coverage rather than for speed.

One compiled `@kernel` per entry, so the list is short on purpose. Measured on an
RTX 4000 Ada over SAM 2's six `addmm` shapes, weighted by their share of the
encoder's GEMM arithmetic, against the register-blocked kernel's 20.5:

     64 x 128, 8 warps   28.3      <- shipped default
     64 x  64, 4 warps   26.1
    128 x  64, 8 warps   22.3
     32 x 128, 8 warps   19.8      but 20.7 against 13.9 on the one shape that
                                      only it divides

**Widening the warp grid wins; widening the warp tile loses.** A 2x4 warp *tile*
(same block, 4 warps) runs at 7.3 and `mul_mm.comp`'s own 4x4 at 5.9, because a
16x16 fp32 accumulator is 8 registers per lane and sixteen of them do not fit —
the driver spills them into workgroup memory, 31 KB of it. This kernel wants
warps in flight, not work per warp, which is the opposite of what the reference
is tuned for and is worth re-measuring on any other device.

Every entry is checked by `test_gemm_staged.jl` at ten values of K, and that is
not a formality: 96 x 128 silently lost 4 of every 32 k-terms per row until the
staging index stopped going through a real division. See `splitidx` and
`test_shared_index_division.jl`.

**The 28.3-vs-27.x ordering above was an artefact of averaging.** 96 x 128 is the
faster kernel on four of SAM 2's six shapes, by 8.6% to 21.0%; it loses only on
`576 x 4096 x 2304`, and it loses there by 18.3%, which is enough to take the
weighted mean with it because that shape is 24.4% of the arithmetic. Swept over K
with everything else fixed, the loss is not a trend but a wall — see
`gemm_aliasing`. Per-shape, the mean was hiding the opposite conclusion.
"""
const GEMM_TILINGS = GemmTiling[
    (3, 2, 2, 4, 32, 8),    #  96 x 128, 8 warps — faster wherever it is allowed
    (2, 2, 2, 4, 32, 8),    #  64 x 128, 8 warps
    (2, 2, 2, 2, 32, 8),    #  64 x  64, 4 warps
    (1, 2, 2, 4, 32, 8),    #  32 x 128, 8 warps
    (1, 1, 2, 4, 32, 8),    #  32 x  64, 8 warps — one tile per warp, widest reach
]

"""
    GEMM_TILING[]

Force one tiling instead of letting `gemm_tiling` choose, or `nothing` for the
shape-driven choice. A `Ref` so a comparison runs interleaved in one session on
one set of buffers, which is the only form of it that means anything on a card
whose clock drifts during the run.
"""
const GEMM_TILING = Ref{Union{Nothing,GemmTiling}}(nothing)

"""
    GEMM_STAGED[]

Use the workgroup-staged kernel where the shape allows it.
"""
const GEMM_STAGED = Ref(true)

"""
    gemm_tiling(M, N, K) -> GemmTiling | nothing

The fastest tiling whose block divides `M x N x K`, or `nothing` if none does.

The kernel masks nothing — a cooperative-matrix load is one instruction with no
bounds check to hang a predicate on — so a block that does not divide exactly is
not a slow case, it is an out-of-range read. Falling through to the
register-blocked kernel is the answer for those.
"""
@inline gemm_divides(c::GemmTiling, M::Int, N::Int, K::Int) =
    M % gemm_bm(c) == 0 && N % gemm_bn(c) == 0 && K % gemm_bk(c) == 0

"""
    gemm_aliasing(c, K) -> Bool

Whether this tiling hits the stride pathology at this `K`, and must be passed
over even though its block divides the shape.

The 96-row block collapses when `K` is a multiple of 256, and only then. Swept at
M = 576, N = 4096 with everything but `K` fixed, it runs 14-22% *faster* than the
64-row block at K = 576, 1152, 1728, 2016, 2112, 2208, 2400, 2496, 2880 and 3456
— and 10-18% slower at 1024, 1280, 1536, 2048, 2304, 2560 and 3072. Every value
in the second list is a multiple of 256 and no value in the first is. It is not a
crossover: at the bad K its throughput pins at 30-33 TFLOP/s no matter how large
K grows, while the 64-row block scales normally past 40.

`A` is `M x K`, so its row stride is `2K` bytes and a `K` divisible by 256 makes
that a multiple of 512 — the classic power-of-two-stride aliasing shape, where
the rows of a staged tile collide in a few cache sets. That is the suspected
mechanism and it is *not* confirmed: both blocks are register-limited to two
workgroups per SM (120 and 128 registers, neither spilling, measured through
`VK_KHR_pipeline_executable_properties`), so it is not occupancy, and the 96-row
block reads less global memory than the 64-row one, so it is not traffic either.
Pinning it further needs Nsight.

Keyed on the block not being a power of two rather than on the literal 96,
because that is the property the suspected mechanism turns on. The generalisation
is untested for other non-power-of-two blocks — but it can only ever *decline* a
tiling in favour of one that was measured, so its failure mode is a missed win.
"""
@inline gemm_aliasing(c::GemmTiling, K::Int) = !ispow2(gemm_bm(c)) && K % 256 == 0

@inline function gemm_tiling(M::Int, N::Int, K::Int)
    forced = GEMM_TILING[]
    # A forced tiling still has to divide the shape. Skipping that check makes a
    # benchmark quietly lie: a 288-row product forced onto a 64-row block computes
    # 256 rows, reads past the operands for the rest, and reports a number for
    # work it did not do — which is exactly what the first run of this comparison
    # did.
    forced === nothing || return gemm_divides(forced, M, N, K) ? forced : nothing
    for c in GEMM_TILINGS
        gemm_divides(c, M, N, K) && !gemm_aliasing(c, K) && return c
    end
    return nothing
end

"""Which tiling this call may use, or `nothing` for the register-blocked kernel."""
@inline function staged_gemm_tiling(M::Int, N::Int, K::Int, nbatch::Int, splitk::Int)
    GEMM_STAGED[] && nbatch == 1 || return nothing
    # **The staged kernel does not split K** — it walks the whole of it and writes
    # one plane. If the caller's plan says otherwise it has allocated `splitk`
    # partial planes and will sum them, so the other planes' stale scratch lands
    # in the result. Latent while `GEMM_STAGED` was off, and invisible on the
    # shapes it was measured on because they all choose `splitk == 1`; at
    # 64x64x64 the plan picks 4 and the answer comes back 0.83 relative error.
    splitk == 1 || return nothing
    c = gemm_tiling(M, N, K)
    c === nothing && return nothing
    return haskey(GEMM_STAGED_KERNELS, c) ? c : nothing
end

"""Does this device implement the tile `mul!` wants?"""
# The block kernel derives its subgroup index as `lane ÷ 32` and GEMM_WORKGROUP
# is sized as "2 subgroups" on the same assumption, so the whole thing is only
# correct where a subgroup is 32 lanes wide. Same constant the pipeline layer
# uses when it pins a coopmat pipeline's subgroup size.
const GEMM_SUBGROUP = COOPMAT_SUBGROUP

# Cooperative-matrix operations are subgroup-scoped, so a device whose subgroup
# is not 32 lanes gets a different number of subgroups per workgroup than this
# kernel assumes, and `lane ÷ 32` stops naming a real subgroup. The arithmetic
# still comes out right for the lanes that run — on a wave64 device exactly half
# the output tile is written, bit-exact, and the other half stays zero, which is
# a silently wrong answer rather than a failure.
#
# Two ways to have a 32-lane subgroup: the device is natively wave32, or it lets
# the pipeline pin its subgroup size, which `get_compute_pipeline` does for any
# coopmat module. Failing both, `mul!` falls through to `gemmlaunch!`, which is
# correct at any width — slower, but not wrong. Making the kernel itself
# wave-size agnostic would mean retuning GEMM_WORKGROUP and the block factors
# together, and those were measured at 32.
function coopmat_gemm_available(ctx::VkContext = vk_context())
    coopmat_shape(ctx, Float16, GEMM_TILE, GEMM_TILE, GEMM_TILE) &&
        (device_subgroup_size(ctx) == GEMM_SUBGROUP ||
         can_require_subgroup_size(ctx, GEMM_SUBGROUP))
end

# Tile index comes from the global lane index, so a workgroup may hold several
# subgroups. Nothing is masked: the caller guarantees M and N are multiples of
# `GEMM_TILE * BLK` and K a multiple of `GEMM_TILE`, because a cooperative-matrix
# load is a single instruction with no bounds check to hang a predicate on.
#
# `BLK` and `SPLITK` are what make one kernel cover both regimes:
#
#   * a square 2048³ wants BLK=4 — sixteen `muladd`s per eight loads — and no
#     split, because output tiling alone already yields 512 subgroups;
#   * this model's dominant convolution is M=128, N=256, K=2304 after im2col,
#     which at BLK=4 is *eight* subgroups for the whole device. It runs at
#     0.16 TFLOP/s that way, worse than the scalar implicit-GEMM it replaces.
#     Shrinking the block and splitting the reduction is the only way to fill
#     the device on a shape that skinny.
#
# With SPLITK > 1 each split writes its own `C[:, :, s]` plane and a separate
# pass sums them. Accumulating in place would need atomics on a cooperative
# matrix, and SPIR-V has no per-element access to one.
# One kernel per register block, generated with the block size as a *literal*.
# Writing a single kernel and gating the tiles on `i <= BLK` looks equivalent —
# `BLK` is a `Val` parameter, so the branches fold — but it is not: a
# cooperative matrix defined inside a conditional is no longer a plain SSA value
# to the emitter, and every shape collapsed to ~0.15 ms, including the ones that
# had been running at 4 TFLOP/s. The generated bodies have no branches at all.
"""
Start an accumulator tile from `bias` rather than zero, so the bias add is free.

A stride of 0 makes every column of the tile read the same `GEMM_TILE` elements —
i.e. a broadcast of `bias[row]` across the tile's columns, which is exactly what
`addmm` wants. Verified on the device (`test_coopmat_epilogue.jl`); the Vulkan
specification does not say what stride 0 does, so it is measured, not assumed.

The `Nothing` method is the old behaviour and is what attention and the
convolution take — they have no bias and accumulate into fp32 partials.
"""
@inline accinit(::Nothing, ptr, off) = zero(AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator})
@inline function accinit(bias, ptr, off)
    # Loaded in the bias's OWN element type and converted, not read as fp32.
    # Under autocast the model's biases are fp16, and reading those bytes as
    # fp32 is not a rounding error — it is a different number. It survived an
    # fp32-bias unit test and took SAM 2's masks to IoU 0.0.
    b = AcceleratedMatrix{eltype(bias),GEMM_TILE,GEMM_TILE,Accumulator}(ptr, off, 0)
    return convert(AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator}, b)
end

"""
Store an fp32 accumulator into `C`, converting to `C`'s element type in registers.

For an fp16 destination this is the whole of `mm_epilogue_kernel!` — which reads
`M x N` fp32 back out of a scratch, adds the bias and writes fp16, for 23% of
matmul time on shapes where `splitk == 1` and there is nothing to reduce. For an
fp32 destination `convert` is the identity and this is what it always was.
"""
@inline accstore!(C, off, ld, c::AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator}) =
    accstore!(C, off, ld, c, identity)

"""
Store an accumulator tile, applying `f` to every component on the way out.

`f` is an ordinary unary function passed as a kernel argument — a singleton, so
it inlines and costs nothing when it is `identity`, and the `F === typeof(identity)`
test below folds away with it. Keeping the activation on the caller's side is
what stops a GEMM in the runtime from having to know what a `gelu` is; the
component access it needs is the general capability, see `coopmat_getcomp`.

**Applied after the conversion, not before.** The accumulator is fp32 and the
destination is usually fp16, and an activation on the fp32 value is a *different
function* from the same activation on the rounded one — more accurate, and not
what an exported graph computed when it rounded between the matmul and the
activation. Matching the graph is the point; being closer to the real number is
not.
"""
@inline function accstore!(C, off, ld,
                           c::AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator},
                           f::F) where {F}
    m = convert(AcceleratedMatrix{eltype(C),GEMM_TILE,GEMM_TILE,Accumulator}, c)
    if F !== typeof(identity)
        n = coopmat_length(AcceleratedMatrix{eltype(C),GEMM_TILE,GEMM_TILE,Accumulator})
        for i in Int32(0):(n - Int32(1))
            m = coopmat_setcomp(m, i, f(coopmat_getcomp(m, i)))
        end
    end
    copyto!(pointer(C), off, ld, m)
end

const GEMM_BLOCK_KERNELS = Dict{Int,Any}()

for BLK in (1, 2, 4)
    kname = Symbol("coopmat_gemm_kernel_", BLK, "!")
    @eval begin
        @kernel cpu=false function $kname(C, @Const(A), @Const(B), bias, epi,
                                ::Val{M}, ::Val{N}, ::Val{K},
                                ::Val{KPER}) where {M,N,K,KPER}
            lane = @index(Global, Linear) - 1
            g = lane ÷ 32
            tiles_m = M ÷ (GEMM_TILE * $BLK)
            ntiles = tiles_m * (N ÷ (GEMM_TILE * $BLK))
            # Batched by decomposing the subgroup index, not by a second kernel.
            # `splitk` and `ntiles` come from `Val` parameters, so `per` is a
            # compile-time constant and this is two folded integer ops; with
            # `nbatch == 1` it is `bat = 0, r = g` and the arithmetic below is
            # identical to the unbatched form it replaces. The batch is
            # OUTERMOST so consecutive subgroups still walk one matrix's tiles.
            splitk = (K ÷ GEMM_TILE) ÷ KPER
            per = ntiles * splitk
            bat = g ÷ per
            r = g % per
            t = r % ntiles
            sk = r ÷ ntiles
            tm = (t % tiles_m) * (GEMM_TILE * $BLK)
            tn = (t ÷ tiles_m) * (GEMM_TILE * $BLK)
            aoff = bat * M * K
            boff = bat * K * N
            coff = bat * M * N * splitk

            # `KPER` — how many 16-wide k-tiles this split covers — is a static
            # parameter and every split gets exactly that many, so the reduction
            # loop has a compile-time trip count. Writing it with runtime bounds
            # (`while k0 < min(K, klo + kstep)`) also cost the unrolling, so
            # `SPLITK` is chosen to divide `K ÷ GEMM_TILE` exactly.
            klo = sk * KPER * GEMM_TILE

            # Zero, or the bias broadcast down the tile's rows — `bias` is
            # indexed by the output ROW, so the offset follows `tm` and `i` and
            # is independent of `j`.
            bp = bias === nothing ? bias : pointer(bias)
            Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                c_i_j = accinit(bias, bp, 1 + tm + (i - 1) * GEMM_TILE)

            # Unrolled by two, by hand. Nothing in Lava's pipeline unrolls (see
            # `unroll_loops!`), and on a loop whose body is a handful of loads
            # and MMAs the per-iteration compare-and-branch is a real fraction of
            # the work. Two k-tiles per iteration also gives the scheduler two
            # independent load groups to overlap. The tail handles an odd `KPER`.
            kt = 0
            while kt + 2 <= KPER
                Base.Cartesian.@nexprs 2 u -> begin
                    k0_u = klo + (kt + (u - 1)) * GEMM_TILE
                    Base.Cartesian.@nexprs $BLK i -> a_i_u =
                        AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                            pointer(A), 1 + aoff + tm + (i - 1) * GEMM_TILE + k0_u * M, M)
                    Base.Cartesian.@nexprs $BLK j -> b_j_u =
                        AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                            pointer(B), 1 + boff + k0_u + (tn + (j - 1) * GEMM_TILE) * K, K)
                    Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                        c_i_j = muladd(a_i_u, b_j_u, c_i_j)
                end
                kt += 2
            end
            while kt < KPER
                k0 = klo + kt * GEMM_TILE
                Base.Cartesian.@nexprs $BLK i -> a_i =
                    AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                        pointer(A), 1 + aoff + tm + (i - 1) * GEMM_TILE + k0 * M, M)
                Base.Cartesian.@nexprs $BLK j -> b_j =
                    AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                        pointer(B), 1 + boff + k0 + (tn + (j - 1) * GEMM_TILE) * K, K)
                Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                    c_i_j = muladd(a_i, b_j, c_i_j)
                kt += 1
            end

            Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                accstore!(C, 1 + coff + tm + (i - 1) * GEMM_TILE +
                             (tn + (j - 1) * GEMM_TILE) * M + sk * M * N,
                          M, c_i_j, epi)
        end
        GEMM_BLOCK_KERNELS[$BLK] = $kname
    end
end

# One workgroup computes a `BM x BN` block of C, staging A and B through
# workgroup memory: fill shared, barrier, multiply out of shared, barrier.
#
# **The innermost loop is the reference's, and its shape is the point.**
# `mul_mm.comp` walks (k-tile, tile row, tile column) with a single `cache_a` and
# a single `cache_b`, reloading B from shared for every `coopMatMulAdd`:
#
#     for (i = 0; i < BK; i += TK)
#       for (cm_row ...)  { coopMatLoad(cache_a, ...);
#         for (cm_col ...) { coopMatLoad(cache_b, ...);
#                            sums[..] = coopMatMulAdd(cache_a, cache_b, sums[..]); } }
#
# That reads as redundant work and is the opposite: a 16×16 fp16 fragment is four
# registers per lane, so hoisting `STM` A-fragments and `STN` B-fragments out of
# the loop — which is what the first port did, across a 2-way k-unroll on top —
# costs 64 registers at 4×4 against an accumulator block that already wants 128.
# The driver then spills the accumulators into workgroup memory (31 KB of it,
# reported by `VK_KHR_pipeline_executable_properties`) and the kernel runs at a
# quarter speed. Shared memory is fast; reloading from it is the cheap side of
# that trade, and it is why the reference can afford a 4×4 warp tile at all.
#
# Both staging loops are written so that consecutive lanes read consecutive
# *global* addresses — `i` fastest for A, `k` fastest for B — because that is the
# side where coalescing matters. The shared side is strided and does not care.
#
# A comment rather than a docstring because the kernels are generated in the
# `@eval` loop below and there is no single definition for it to attach to.

# ── wide staging loads: tried, measured, removed ─────────────────────────────
#
# `mul_mm.comp` stages its A and B blocks with wide loads (`LOAD_VEC_A_EFF`) —
# the last structural difference between its inner loop and ours. Both of our
# blocks are contiguous on the axis a lane walks and every offset is a multiple
# of 4 by construction, so it ports directly: `unsafe_load` through a
# `Ptr{NTuple{4,Float16}}` works on this backend and is bit-exact.
#
# **It loses, at every width.** Against the register-blocked kernel on the four
# dominant shapes:
#
#   scalar   1.04 - 1.09x
#   2-wide   0.80 - 0.92x
#   4-wide   0.69 - 0.81x
#
# The global side is not the problem — it is coalesced either way. The shared
# side is: `@localmem` here is `Float16`, so a lane that loaded `V` elements
# writes them with `V` scalar stores whose addresses stride by `V` across the
# warp. At V=2 that is a 2-way bank conflict on every store and at V=4 a 4-way,
# against none at all for the stride-1 scalar loop. The monotonic 1 > 2 > 4
# ordering is the conflict count, not noise.
#
# The reference does not hit this because it widens the *shared array* at the
# same time — `FLOAT_TYPEV2 buf_a[]`, with `SHMEM_STRIDE` counted in `vec2`.
#
# **That capability now exists**; this kernel simply does not use it yet. Two
# pieces were missing and both are in, with a device test in
# `test_coopmat_shared.jl`:
#
#   * `@localmem NTuple{2,VecElement{Float16}}` — `Op.OpCompositeInsert` was used
#     by the emitter and never declared, so building a vector value element by
#     element died with `UndefVarError` rather than compiling;
#   * `loadw2`, a cooperative-matrix load whose access chain addresses the vector
#     rather than the scalar component. `OpCooperativeMatrixLoadKHR` permits a
#     vector pointee whose component type matches the matrix and counts `Stride`
#     in those vectors.
#
# What remains is the integration, and it is mechanical: declare `sA`/`sB` as
# vec2 arrays, halve `LDA`/`LDB` and the coopmat strides, and have each lane
# store one packed pair instead of two halves — `AREPS`/`BREPS` halve with it.
# A is packed along `m` and B along `k`, which is the contiguous axis on each, so
# both stay coalesced on the global side and neither needs a transpose. The
# `test_gemm_staged.jl` K-sweep is what should catch it if the index maths slips.

const GEMM_STAGED_KERNELS = Dict{GemmTiling,Any}()

"""
`vec2`-typed staging buffers, the shape `mul_mm.comp` uses.

Every shared access becomes 32 bits instead of 16: the staging store writes one
packed pair per lane, and the cooperative-matrix load addresses the vector with
its stride counted in vectors. A packs along `m` and B along `k` — the contiguous
axis on each — so both stay coalesced globally and neither needs a transpose.

**On, and worth 1.09x.** Measured interleaved over SAM 2's six `addmm` shapes,
weighted by their share of the encoder's GEMM arithmetic:

    register-blocked   21.0 TFLOP/s
    scalar staging     32.3
    vec2 staging       35.3      <- 1.09x over scalar, 1.68x over register-blocked

which is **79% of cuBLAS's 44.6**, from 70%. The gain is largest exactly where
staging is largest relative to arithmetic: the 288-row shape, which needs the
32 x 128 tiling, goes 18.7 -> 28.3, a 1.51x.

A `Ref` because the only measurement worth having is the scalar and vec2 kernels
interleaved in one session on one set of buffers. Both are generated; this picks
which runs. A tiling with no vec2 twin falls back to its scalar kernel.
"""
const GEMM_VEC2 = Ref(true)

"""`vec2`-staged twins of `GEMM_STAGED_KERNELS`, keyed the same way."""
const GEMM_STAGED_V2_KERNELS = Dict{GemmTiling,Any}()

"""
Whether every linear index the narrow kernel forms fits in an `Int32`: `M*K` for
the A staging, `K*N` for B, `M*N` for the store.

`widemul`, because the products are the thing being range-checked and computing
them in `Int` to test them is how a range check overflows itself. Strictly `<`
rather than `<=`: the indices are 1-based, so the largest the A staging forms is
`M*K + 1`, not `M*K`.
"""
@inline gemm_fits32(M::Integer, N::Integer, K::Integer) =
    widemul(Int(M), Int(K)) < typemax(Int32) && widemul(Int(K), Int(N)) < typemax(Int32) &&
    widemul(Int(M), Int(N)) < typemax(Int32)

"""Narrow-index twins of `GEMM_STAGED_V2_KERNELS`. See [`GEMM_NARROW`](@ref)."""
const GEMM_STAGED_V2N_KERNELS = Dict{GemmTiling,Any}()

"""
    GEMM_NARROW[] :: Bool

Compute the staging addresses in `Int32` instead of `Int`.

Julia hands out `Int64` indices and Lava emits them as-is, but NVIDIA has no
64-bit integer unit: adds and multiplies are emulated. The same narrowing was
worth **1.56x** in `im2col_kernel!`, which is why it was tried here.

It does **not** work the way that suggests. The register count goes *up*
(118 -> 124) and the occupancy is unchanged at 2 workgroups an SM, so whatever it
buys is instruction count in the staging loop, not residency. Measured
interleaved in one session, both orders, over SAM 2's eight `addmm` shapes
weighted by call count: **-1.2%**, from -4.9% to +0.9%, results bit-identical.

The gap between that and the 3.5-9% a *cross-session* comparison first showed is
the reason the rule exists — the earlier baseline was a different process on a
differently-warmed card, and it inflated the effect by 4x.

Kept anyway: it is free and never wrong. Both kernels exist because the narrow
one is only *legal* while `M*K`, `K*N` and `M*N` fit in an `Int32`
(`gemm_fits32`), and the wide one is the fallback for anything larger, so the
duplication is a correctness requirement and not only a measurement convenience.
"""
const GEMM_NARROW = Ref(true)

"A 2-wide fp16 vector, i.e. `f16vec2`; see `Lava.coopmat_vec2`."
const GemmV2 = NTuple{2,VecElement{Float16}}

for (ci, cfg) in enumerate(GEMM_TILINGS)
    STM, STN, WM, WN, BK, PAD = cfg
    BM, BN = GEMM_TILE * STM * WM, GEMM_TILE * STN * WN
    WG = WM * WN * COOPMAT_SUBGROUP
    LDA, LDB = BM + PAD, BK + PAD
    NKT = BK ÷ GEMM_TILE
    (BM * BK) % WG == 0 && (BK * BN) % WG == 0 ||
        error("gemm tiling $cfg: staging does not divide the workgroup")
    # There is deliberately no "BM must divide WG" constraint here. There was one,
    # because every geometry whose `BM` divided the workgroup was exact and four
    # that did not lost staged data — but that was the symptom, not the cause. The
    # dividing geometries were the ones whose `idx % BM` / `idx ÷ BM` folded to a
    # mask and a shift; the rest emitted `OpUDiv`, and that is what loses stores.
    # `splitidx` removes the division, so any `BM` is safe again. See its
    # docstring and `test_shared_index_division.jl`.
    AREPS, BREPS = (BM * BK) ÷ WG, (BK * BN) ÷ WG
    kname = Symbol("coopmat_gemm_staged_kernel_", ci, "!")
    @eval begin
        # `unsafe_indices=true` drops KA's `__validindex` guard. It is dead here —
        # `coopmat_gemm!` launches an exact multiple of the workgroup size, and the
        # kernel derives every index from the group and local ids rather than from
        # a global one — and it is not free: with the k-loop written as a counted
        # `for`, the structurizer failed to flatten the guard and left the
        # `muladd` under an `OpSelectionMerge`, which cost **3x** (8.8 against 26.1
        # TFLOP/s weighted). The guard is why the two loop forms structurize
        # differently at all.
        @kernel cpu=false unsafe_indices=true function $kname(
                                          C, @Const(A), @Const(B), bias, epi,
                                          ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
            sA = @localmem Float16 ($LDA * $BK,)
            sB = @localmem Float16 ($LDB * $BN,)

            tid = @index(Local, Linear) - 1
            blk = @index(Group, Linear) - 1
            nblk_m = M ÷ $BM
            tm = (blk % nblk_m) * $BM
            tn = (blk ÷ nblk_m) * $BN

            s = tid ÷ 32                # subgroup within the workgroup
            sm = (s % $WM) * $STM       # its first tile row, in tiles
            sn = (s ÷ $WM) * $STN       # its first tile column, in tiles

            bp = bias === nothing ? bias : pointer(bias)
            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

            # A counted `for` rather than `while k0 < K`. This looked for a while
            # like it mattered for correctness — it does not, and the note in
            # `test_shared_index_division.jl` records the measurement that showed
            # both forms lose stores and both are exact once `splitidx` removes
            # the division. Kept because the trip count is known here anyway.
            for kb in 0:(K ÷ $BK - 1)
                k0 = kb * $BK
                # One element per lane per step, and it stays that way — see the
                # note on wide staging loads above.
                #
                # A block: rows [tm, tm+BM) × cols [k0, k0+BK), column-major both sides.
                @inbounds for r in 0:($AREPS - 1)
                    idx = tid + r * $WG
                    i, kk = splitidx(idx, Val($BM))
                    sA[1 + i + kk * $LDA] = A[1 + (tm + i) + (k0 + kk) * M]
                end
                # B block: rows [k0, k0+BK) × cols [tn, tn+BN).
                @inbounds for r in 0:($BREPS - 1)
                    idx = tid + r * $WG
                    kk, j = splitidx(idx, Val($BK))
                    sB[1 + kk + j * $LDB] = B[1 + (k0 + kk) + (tn + j) * K]
                end
                @synchronize

                # One A fragment and one B fragment live, both reassigned — the
                # reference's ordering. See the note above on why reloading beats
                # hoisting here.
                Base.Cartesian.@nexprs $NKT u -> begin
                    kt = (u - 1) * GEMM_TILE
                    Base.Cartesian.@nexprs $STM mt -> begin
                        a = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                                sA, 1 + (sm + mt - 1) * GEMM_TILE + kt * $LDA, $LDA)
                        Base.Cartesian.@nexprs $STN nt -> begin
                            b = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                                    sB, 1 + kt + (sn + nt - 1) * GEMM_TILE * $LDB, $LDB)
                            c_mt_nt = muladd(a, b, c_mt_nt)
                        end
                    end
                end
                @synchronize   # nothing may refill shared until every subgroup is done
            end

            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                accstore!(C,
                          1 + (tm + (sm + mt - 1) * GEMM_TILE) +
                              (tn + (sn + nt - 1) * GEMM_TILE) * M,
                          M, c_mt_nt, epi)
        end
        GEMM_STAGED_KERNELS[$cfg] = $kname
    end

    # ── the same kernel with `vec2` staging buffers ──────────────────────────
    #
    # Identical arithmetic; only the staging width changes. Each lane stores one
    # packed pair rather than two halves, so the trip counts halve, and the
    # cooperative-matrix loads address the vector with their stride counted in
    # vectors — `LDA2`/`LDB2` and the tile offsets are all the scalar values
    # divided by two, which is exact because every one of them is even by
    # construction (`GEMM_TILE` is 16, `BK` and `PAD` are even).
    if iseven(BM) && iseven(BK) && iseven(LDA) && iseven(LDB) &&
       ((BM * BK) ÷ 2) % WG == 0 && ((BK * BN) ÷ 2) % WG == 0
        LDA2, LDB2 = LDA ÷ 2, LDB ÷ 2
        AREPS2, BREPS2 = (BM * BK) ÷ 2 ÷ WG, (BK * BN) ÷ 2 ÷ WG
        BM2, BK2 = BM ÷ 2, BK ÷ 2
        kv2 = Symbol("coopmat_gemm_staged_kernel_", ci, "_v2!")
        @eval begin
            @kernel cpu=false unsafe_indices=true function $kv2(
                                              C, @Const(A), @Const(B), bias, epi,
                                              ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
                sA = @localmem GemmV2 ($LDA2 * $BK,)
                sB = @localmem GemmV2 ($LDB2 * $BN,)

                tid = @index(Local, Linear) - 1
                blk = @index(Group, Linear) - 1
                nblk_m = M ÷ $BM
                tm = (blk % nblk_m) * $BM
                tn = (blk ÷ nblk_m) * $BN

                s = tid ÷ 32
                sm = (s % $WM) * $STM
                sn = (s ÷ $WM) * $STN

                bp = bias === nothing ? bias : pointer(bias)
                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

                for kb in 0:(K ÷ $BK - 1)
                    k0 = kb * $BK
                    # A: pairs along `m`, the contiguous axis, so consecutive
                    # lanes still read consecutive global addresses.
                    @inbounds for r in 0:($AREPS2 - 1)
                        idx = tid + r * $WG
                        p, kk = splitidx(idx, Val($BM2))
                        g = 1 + (tm + 2p) + (k0 + kk) * M
                        sA[1 + p + kk * $LDA2] = (VecElement(A[g]), VecElement(A[g + 1]))
                    end
                    # B: pairs along `k`, contiguous here for the same reason.
                    @inbounds for r in 0:($BREPS2 - 1)
                        idx = tid + r * $WG
                        q, j = splitidx(idx, Val($BK2))
                        g = 1 + (k0 + 2q) + (tn + j) * K
                        sB[1 + q + j * $LDB2] = (VecElement(B[g]), VecElement(B[g + 1]))
                    end
                    @synchronize

                    Base.Cartesian.@nexprs $NKT u -> begin
                        kt = (u - 1) * GEMM_TILE
                        Base.Cartesian.@nexprs $STM mt -> begin
                            a = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                                    sA, 1 + (sm + mt - 1) * ($(GEMM_TILE ÷ 2)) +
                                        kt * $LDA2, $LDA2)
                            Base.Cartesian.@nexprs $STN nt -> begin
                                b = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                                        sB, 1 + (kt ÷ 2) +
                                            (sn + nt - 1) * GEMM_TILE * $LDB2, $LDB2)
                                c_mt_nt = muladd(a, b, c_mt_nt)
                            end
                        end
                    end
                    @synchronize
                end

                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    accstore!(C,
                              1 + (tm + (sm + mt - 1) * GEMM_TILE) +
                                  (tn + (sn + nt - 1) * GEMM_TILE) * M,
                              M, c_mt_nt, epi)
            end
            GEMM_STAGED_V2_KERNELS[$cfg] = $kv2
        end

        kv2n = Symbol("coopmat_gemm_staged_kernel_", ci, "_v2n!")
        @eval begin
            @kernel cpu=false unsafe_indices=true function $kv2n(
                                              C, @Const(A), @Const(B), bias, epi,
                                              ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
                sA = @localmem GemmV2 ($LDA2 * $BK,)
                sB = @localmem GemmV2 ($LDB2 * $BN,)

                tid = Int32(@index(Local, Linear) - 1)
                blk = Int32(@index(Group, Linear) - 1)
                nblk_m = Int32(M ÷ $BM)
                tm = (blk % nblk_m) * Int32($BM)
                tn = (blk ÷ nblk_m) * Int32($BN)

                s = tid ÷ Int32(32)
                sm = (s % Int32($WM)) * Int32($STM)
                sn = (s ÷ Int32($WM)) * Int32($STN)

                bp = bias === nothing ? bias : pointer(bias)
                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

                for kb in Int32(0):Int32(K ÷ $BK - 1)
                    k0 = kb * Int32($BK)
                    # A: pairs along `m`, the contiguous axis, so consecutive
                    # lanes still read consecutive global addresses.
                    @inbounds for r in Int32(0):Int32($AREPS2 - 1)
                        idx = tid + r * Int32($WG)
                        p, kk = splitidx(idx, Val($BM2))
                        g = Int32(1) + (tm + Int32(2) * Int32(p)) +
                            (k0 + Int32(kk)) * Int32(M)
                        sA[1 + Int(p) + Int(kk) * $LDA2] =
                            (VecElement(A[g]), VecElement(A[g + Int32(1)]))
                    end
                    # B: pairs along `k`, contiguous here for the same reason.
                    @inbounds for r in Int32(0):Int32($BREPS2 - 1)
                        idx = tid + r * Int32($WG)
                        q, j = splitidx(idx, Val($BK2))
                        g = Int32(1) + (k0 + Int32(2) * Int32(q)) +
                            (tn + Int32(j)) * Int32(K)
                        sB[1 + Int(q) + Int(j) * $LDB2] =
                            (VecElement(B[g]), VecElement(B[g + Int32(1)]))
                    end
                    @synchronize

                    Base.Cartesian.@nexprs $NKT u -> begin
                        kt = (u - 1) * GEMM_TILE
                        Base.Cartesian.@nexprs $STM mt -> begin
                            a = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                                    sA, 1 + (sm + mt - 1) * ($(GEMM_TILE ÷ 2)) +
                                        kt * $LDA2, $LDA2)
                            Base.Cartesian.@nexprs $STN nt -> begin
                                b = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                                        sB, 1 + (kt ÷ 2) +
                                            (sn + nt - 1) * GEMM_TILE * $LDB2, $LDB2)
                                c_mt_nt = muladd(a, b, c_mt_nt)
                            end
                        end
                    end
                    @synchronize
                end

                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    accstore!(C,
                              1 + Int(tm + (sm + mt - 1) * Int32(GEMM_TILE)) +
                                  (tn + (sn + nt - 1) * GEMM_TILE) * M,
                              M, c_mt_nt, epi)
            end
            GEMM_STAGED_V2N_KERNELS[$cfg] = $kv2n
        end
    end
end

# The scalar kernel takes each operand as a *dense base array plus strides*
# rather than as the wrapper the caller had. A transposed operand is then just a
# pair of swapped strides — no copy, no second kernel, and no wrapper type inside
# the kernel at all. That last part matters: `@Const` runs `Adapt.adapt_structure`
# on the device, and rebuilding a wrapper there drags its constructor's error
# paths in with it (see the PermutedDimsArray quirk).
#
# `K` is passed rather than read from `axes(A, 2)` for the same reason — the host
# knows the extent, so the kernel need not query anything.
@kernel cpu=false function strided_gemm_kernel!(C, @Const(A), @Const(B), ::Val{K},
                                      co, cr, cc, ao, ar, ac, bo, br, bc,
                                      α, β, M, ntot) where {K}
    # Flat launch: an N-D `ndrange` is partitioned into N-D workgroups, so
    # consecutive lanes stop walking consecutive memory. Worth 2.3 ms of a
    # 31.7 ms inference step when applied to the convolution's im2col and
    # epilogue.
    lin = @index(Global, Linear)
    if lin <= ntot
    i = (Int32(lin) - Int32(1)) % Int32(M) + Int32(1)
    j = (Int32(lin) - Int32(1)) ÷ Int32(M) + Int32(1)
    @inbounds begin
        T = eltype(C)
        acc = zero(T)
        ai = ao + (i - 1) * ar
        bj = bo + (j - 1) * bc
        # Unrolled by four, by hand. LLVM unrolls this loop for NVPTX and does
        # not for SPIR-V — nothing in Lava's pipeline runs a loop-unroll pass and
        # `LoopUnrollPass` declines without a GPU `TargetTransformInfo` to ask
        # about partial unrolling (see `unroll_loops!`). On a dependent `muladd`
        # chain the difference is 3905 -> 13699 GFLOP/s, which is most of the gap
        # to CUDA.jl on the same source. `K` is a `Val`, so the trip counts fold.
        k = 0
        while k + 4 <= K
            Base.Cartesian.@nexprs 4 d -> begin
                kd = k + (d - 1)
                acc = muladd(T(A[ai + kd * ac]), T(B[bj + kd * br]), acc)
            end
            k += 4
        end
        while k < K
            acc = muladd(T(A[ai + k * ac]), T(B[bj + k * br]), acc)
            k += 1
        end
        ci = co + (i - 1) * cr + (j - 1) * cc
        C[ci] = iszero(β) ? acc * α : muladd(acc, α, C[ci] * β)
    end
    end
end

"""
    gemmstrides(A) -> (base, offset, rowstride, colstride) | nothing

Describe `A` as a strided window onto a dense `LavaArray`, or `nothing` if it
isn't one. `offset` is a 1-based linear index into `base`, and the strides are in
elements, so `A[i,j] == base[offset + (i-1)*rowstride + (j-1)*colstride]`.

This is what lets a transpose cost nothing: `PermutedDimsArray(A, (2,1))` and
`transpose(A)` return `A`'s own strides, swapped. Wrappers that are *not*
strided — a reshape of a permuted array, say, whose elements genuinely aren't at
a fixed 2-D stride — return `nothing` and get materialised by the caller.
"""
gemmstrides(A::LavaArray{T,2}) where {T} = (A, 1, 1, size(A, 1))
gemmstrides(A::LavaArray{T,1}) where {T} = (A, 1, 1, length(A))

function gemmstrides(A::PermutedDimsArray{T,2,(2, 1)}) where {T}
    p = gemmstrides(parent(A))
    p === nothing ? nothing : (p[1], p[2], p[4], p[3])
end
gemmstrides(A::PermutedDimsArray{T,2,(1, 2)}) where {T} = gemmstrides(parent(A))

function gemmstrides(A::LinearAlgebra.Transpose{T}) where {T}
    p = gemmstrides(parent(A))
    p === nothing ? nothing : (p[1], p[2], p[4], p[3])
end
# Only for real eltypes: an adjoint of a complex array also conjugates, which is
# not something a stride can express.
function gemmstrides(A::LinearAlgebra.Adjoint{T}) where {T<:Real}
    p = gemmstrides(parent(A))
    p === nothing ? nothing : (p[1], p[2], p[4], p[3])
end

# A reshape is free only when the thing underneath is dense; `LavaArray` is the
# only parent for which that holds unconditionally.
gemmstrides(A::Base.ReshapedArray{T,2,<:LavaArray}) where {T} =
    (parent(A), 1, 1, size(A, 1))

gemmstrides(::AbstractArray) = nothing

# Last resort, for an operand that is not strided over a LavaArray at all — a
# reshape of a permuted array, say, whose elements are genuinely not at a fixed
# 2-D stride. Broadcast, not `copyto!`: `copyto!` on a wrapper falls into Base's
# generic elementwise loop, which scalar-indexes the GPU array from the host.
#
# This copy is not free, and a graph that hits it every call should be
# materialising the tensor as an explicit op instead of leaving the wrapper for
# `mul!` to trip over.
densify(a::AbstractArray) = (d = LavaArray{eltype(a)}(undef, size(a)...); d .= a; d)

"""
    mul!(C, A, B, α, β) -> C

`C = A*B*α + C*β` on the device.

Dispatch is on the *destination* rather than on the operands, because an operand
can be wrapped to any depth — an exported ATen graph routinely hands us a reshape
of a permuted array — and no union of wrapper types catches all of those. If `C`
lives on the device then the multiply happens on the device, whatever the
operands look like; `gemmstrides` unwraps them and only genuinely non-strided
ones are copied.

Uses cooperative matrices when the operands are dense fp16 into an fp32 result,
the extents suit the tile, and the device implements it; otherwise the strided
scalar kernel. That choice is an implementation detail — the result is the same
either way.
"""
function LinearAlgebra.mul!(C::LavaArray{T,2}, A::AbstractVecOrMat,
                            B::AbstractVecOrMat, α::Number, β::Number) where {T}
    M, K = size(A, 1), size(A, 2)
    N = size(B, 2)
    size(B, 1) == K && size(C) == (M, N) ||
        throw(DimensionMismatch("mul!: $(size(C)) = $(size(A)) * $(size(B))"))
    backend = LavaBackend()

    if T === Float32 && eltype(A) === Float16 && eltype(B) === Float16 &&
       isone(α) && iszero(β) && A isa LavaArray && B isa LavaArray &&
       K % GEMM_TILE == 0 && M % GEMM_TILE == 0 && N % GEMM_TILE == 0 &&
       coopmat_gemm_available()
        return coopmat_gemm!(C, A, B, M, N, K)
    end

    gemmlaunch!(C, A, B, M, N, K, T(α), T(β))
end

"""
    coopmat_gemm_shape(M, N, K; cores) -> (BLK, SPLITK)

Register block and reduction split for an `M×N×K` product.

The device is full at roughly four subgroups per shader core. Output tiling
alone reaches that for anything square; the skinny products an im2col
convolution produces do not, so the choice is: keep the widest block that still
divides the extents and split the reduction to make up the shortfall, dropping
to a narrower block only when even an 8-way split cannot fill the device. A
narrower block costs arithmetic intensity — BLK=1 does one `muladd` per two
loads where BLK=4 does sixteen per eight — so it is the last resort, not the
first.
"""
function coopmat_gemm_shape(M::Int, N::Int, K::Int; cores::Int = 48, nbatch::Int = 1)
    # A batched call already has `nbatch` independent copies of every output
    # tile, so the device is full at `target ÷ nbatch` tiles per matrix. Ignoring
    # this is not a missed optimisation, it is a large pessimisation: SAM 2's
    # windowed attention (M=N=256, K=80, nbatch=128) picks `splitk = 5` on the
    # unbatched target and runs at 1.2 TFLOP/s, against 11.6 at `splitk = 1` —
    # 1.14 ms versus 0.12. The split writes five partial planes per matrix and
    # then reads them all back to sum, for parallelism the batch already had.
    target = max(1, cld(4cores, nbatch))
    nk = K ÷ GEMM_TILE
    # Only splits that divide the k-tile count, so every split has the same
    # static trip count (see the kernel).
    divisors = [s for s in 1:min(GEMM_MAXSPLIT, nk) if nk % s == 0]
    fallback = (1, 1)
    first = true
    for blk in (GEMM_MAXBLOCK, 2, 1)
        span = GEMM_TILE * blk
        (M % span == 0 && N % span == 0) || continue
        ntiles = (M ÷ span) * (N ÷ span)
        want = cld(target, ntiles)
        splitk = something(findfirst(>=(want), divisors), length(divisors))
        splitk = divisors[splitk]
        first && (fallback = (blk, splitk); first = false)
        ntiles * splitk >= target && return (blk, splitk)
    end
    fallback
end

const GEMM_MAXSPLIT = 8    # past this the partial-sum traffic outweighs the parallelism

"""
Sum the `SPLITK` partial planes `Cp[:, :, s]` into `C`.

`n` is one plane, `M * N`. Batched runs lay the planes out as
`[batch][split][plane]`, so the source offset needs the batch's own stride of
`S * n` — indexing with `i + s * n` alone would read batch 0's later splits for
every batch and silently return the wrong sum for all but the first.
"""
@kernel cpu=false function splitk_reduce_kernel!(C, @Const(Cp), ::Val{S}, n) where {S}
    i = @index(Global, Linear)
    @inbounds begin
        b = (i - 1) ÷ n                 # 0 for an unbatched call
        j = (i - 1) % n + 1 + b * S * n
        acc = Cp[j]
        for s in 1:(S - 1)
            acc += Cp[j + s * n]
        end
        C[i] = acc
    end
end

"""
Scratch for split-K partial sums, grown monotonically and reused.

A fresh allocation per multiply is not merely wasteful here: the array is
dropped as soon as `coopmat_gemm!` returns while the dispatches reading it are
still queued, and once the pool starts reclaiming blocks under that churn the
result is `sync_access!: buffer is not ALIVE`. One buffer that only ever grows
avoids both. Callers that already own scratch (the convolution's `Workspace`)
pass `partials` and never touch this.
"""
const GEMM_SPLIT_SCRATCH = Ref{Any}(nothing)
const GEMM_SPLIT_RETIRED = Any[]

function splitscratch(M::Int, N::Int, splitk::Int)
    n = M * N * splitk
    buf = GEMM_SPLIT_SCRATCH[]
    if buf === nothing || length(buf)::Int < n
        # Retain, don't drop: dispatches already recorded point into the old
        # buffer and its finalizer would pull it out from under them. Growth is
        # geometric and stops once the largest product has been seen.
        buf === nothing || push!(GEMM_SPLIT_RETIRED, buf)
        buf = LavaArray{Float32}(undef, n + n ÷ 2)
        GEMM_SPLIT_SCRATCH[] = buf
    end
    GPUArrays.derive(Float32, buf::LavaArray{Float32,1}, (M, N, splitk), 0)
end

"""
    coopmat_gemm!(C, A, B, M, N, K) -> C

Tensor-core `C = A*B`. Extents must already be multiples of `GEMM_TILE`; the
caller pads (an im2col matrix is built to size, and weights are padded once at
load) because a cooperative-matrix load has no bounds check.
"""
function coopmat_gemm!(C, A, B, M::Int, N::Int, K::Int;
                       nbatch::Int = 1,
                       blk_split = coopmat_gemm_shape(M, N, K; nbatch),
                       partials = nothing, reduce::Bool = true, bias = nothing,
                       epilogue = identity)
    backend = LavaBackend()
    # A bias goes into the accumulator's initial value, so it must land exactly
    # once — with `splitk > 1` every plane would carry its own copy and the
    # reduction would sum them. The caller keeps its own epilogue there.
    bias === nothing || blk_split[2] == 1 ||
        throw(ArgumentError("coopmat_gemm!: bias needs splitk == 1, got $(blk_split[2])"))
    blk, splitk = blk_split
    # The staged kernel masks nothing and does not split K, so it takes only the
    # shapes it divides exactly and only where the plan wants a single plane —
    # which includes all four of SAM 2's dominant `addmm` shapes, i.e. 72.7% of
    # the encoder's arithmetic.
    c = staged_gemm_tiling(M, N, K, nbatch, splitk)
    if c !== nothing
        wg = gemm_wg(c)
        # The narrow kernel addresses in `Int32`, so it is only legal while every
        # linear index it forms stays inside one: `M*K` for the A staging, `K*N`
        # for B, `M*N` for the store. Far outside anything this repo runs — the
        # largest is 18.9M against a 2.1e9 limit — but it is a silent wrong
        # answer rather than an error if it is ever not, so it is checked.
        narrow = GEMM_NARROW[] && gemm_fits32(M, N, K)
        kern = if !GEMM_VEC2[]
            GEMM_STAGED_KERNELS[c]
        elseif narrow
            get(GEMM_STAGED_V2N_KERNELS, c,
                get(GEMM_STAGED_V2_KERNELS, c, GEMM_STAGED_KERNELS[c]))
        else
            get(GEMM_STAGED_V2_KERNELS, c, GEMM_STAGED_KERNELS[c])
        end
        kern(backend, wg)(
            C, A, B, bias, epilogue, Val(M), Val(N), Val(K);
            ndrange = (M ÷ gemm_bm(c)) * (N ÷ gemm_bn(c)) * wg)
        return C
    end
    span = GEMM_TILE * blk
    ntiles = (M ÷ span) * (N ÷ span)
    dst = splitk == 1 ? C :
          (partials === nothing ? splitscratch(M, N, splitk * nbatch) : partials)
    # A split-K plane is a PARTIAL sum, so an activation on it would be applied
    # to a fraction of the dot product and then summed — wrong, and silently so.
    # The epilogue belongs to whoever reduces the planes.
    epilogue === identity || splitk == 1 ||
        throw(ArgumentError("coopmat_gemm!: an epilogue needs splitk == 1, got $splitk"))
    GEMM_BLOCK_KERNELS[blk](backend, GEMM_WORKGROUP)(
        dst, A, B, bias, epilogue, Val(M), Val(N), Val(K),
        Val((K ÷ GEMM_TILE) ÷ splitk); ndrange = ntiles * splitk * 32 * nbatch)
    splitk == 1 && return C
    # `reduce=false` hands the partial planes back untouched, for a caller whose
    # own epilogue can sum them — a convolution already reads every element of
    # the result to add bias and scatter it, so folding the sum in there saves a
    # dispatch and a full write-plus-read of `M*N*splitk` floats per convolution.
    reduce || return dst
    splitk_reduce_kernel!(backend)(C, dst, Val(splitk), M * N; ndrange = M * N * nbatch)
    C
end

# A matrix-vector product is just the N == 1 case. `C` carries strides like the
# operands do, so its rank never reaches the kernel and one kernel serves both.
function LinearAlgebra.mul!(C::LavaArray{T,1}, A::AbstractVecOrMat,
                            B::AbstractVector, α::Number, β::Number) where {T}
    M, K = size(A, 1), size(A, 2)
    length(B) == K && length(C) == M ||
        throw(DimensionMismatch("mul!: $(size(C)) = $(size(A)) * $(size(B))"))
    gemmlaunch!(C, A, B, M, 1, K, T(α), T(β))
end

function gemmlaunch!(C, A, B, M, N, K, α, β)
    a = gemmstrides(A)
    a === nothing && (a = gemmstrides(densify(A)))
    b = gemmstrides(B)
    b === nothing && (b = gemmstrides(densify(B)))
    c = gemmstrides(C)
    strided_gemm_kernel!(LavaBackend())(c[1], a[1], b[1], Val(K),
                                        c[2], c[3], c[4],
                                        a[2], a[3], a[4],
                                        b[2], b[3], b[4],
                                        α, β, M, M * N; ndrange = M * N)
    C
end

# Disambiguation: Diagonal * matrix.
#
# `mul!(C::LavaArray{T,2}, ::AbstractVecOrMat, ::AbstractVecOrMat, α, β)` above and
# GPUArrays' `mul!(::AbstractGPUVecOrMat, ::Diagonal{<:Any,<:AbstractGPUArray}, …)`
# are both applicable to `mul!(::LavaArray{Float32,2}, ::Diagonal{Float32,LavaArray{Float32,1}}, ::LavaArray{Float32,2}, α, β)`
# and neither is more specific, so the call is an ambiguity error rather than a
# dispatch to either. Lava's method became applicable when the GEMM landed.
#
# GPUArrays' is the one that should win: a diagonal operand is a scaling, and
# routing it through the dense GEMM would materialise the zeros and do O(n)
# times the work. This mirrors its implementation rather than `invoke`-ing it,
# because the signature to invoke through is unwieldy and would silently rot if
# GPUArrays retyped it.
function LinearAlgebra.mul!(C::LavaArray{T,2},
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            B::Union{AbstractGPUArray,
                                     Adjoint{S, <:AbstractGPUArray{S}},
                                     Transpose{S, <:AbstractGPUArray{S}}},
                            α::Number, β::Number) where {T, S}
    dd = D.diag
    d = length(dd)
    m, n = size(B, 1), size(B, 2)
    m′, n′ = size(C, 1), size(C, 2)
    m == d || throw(DimensionMismatch("right hand side has $m rows but D is $d by $d"))
    (m, n) == (m′, n′) ||
        throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    @. C = α * dd * B + β * C
    C
end

# Disambiguation: matrix * Diagonal — the mirror of the case above.
#
# Same collision, other operand: GPUArrays has a second method
# `mul!(::AbstractGPUVecOrMat, ::Union{AbstractGPUArray,Adjoint,Transpose}, ::Diagonal{<:Any,<:AbstractGPUArray}, α, β)`
# which is equally ambiguous against Lava's dense GEMM. Fixing only the
# Diagonal-on-the-left case left this one throwing, and the test for it only
# covered the side that had been fixed — GPUArrays' own linalg/diagonal testset
# is what caught it, for Float32 and ComplexF32.
#
# `D` scales COLUMNS here (A[:,j] * d[j]), so the broadcast transposes `dd`.
function LinearAlgebra.mul!(C::LavaArray{T,2},
                            A::Union{AbstractGPUArray,
                                     Adjoint{S, <:AbstractGPUArray{S}},
                                     Transpose{S, <:AbstractGPUArray{S}}},
                            D::Diagonal{<:Any, <:AbstractGPUArray},
                            α::Number, β::Number) where {T, S}
    dd = D.diag
    d = length(dd)
    m, n = size(A, 1), size(A, 2)
    m′, n′ = size(C, 1), size(C, 2)
    n == d || throw(DimensionMismatch("left hand side has $n columns but D is $d by $d"))
    (m, n) == (m′, n′) ||
        throw(DimensionMismatch("expect output to be $m by $n, but got $m′ by $n′"))
    # `ddT` MUST be hoisted out of the `@.`: inside it, `transpose(dd)` becomes
    # `transpose.(dd)`, which broadcasts transpose over each scalar (a no-op) and
    # leaves a length-n column, scaling ROWS instead of columns. GPUArrays hoists
    # it for the same reason.
    ddT = transpose(dd)
    @. C = α * A * ddT + β * C
    C
end
