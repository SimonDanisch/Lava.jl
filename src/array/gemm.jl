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

"""Does this device implement the tile `mul!` wants?"""
function coopmat_gemm_available(ctx::VkContext = vk_context())
    coopmat_shape(ctx, Float16, GEMM_TILE, GEMM_TILE, GEMM_TILE)
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
const GEMM_BLOCK_KERNELS = Dict{Int,Any}()

for BLK in (1, 2, 4)
    kname = Symbol("coopmat_gemm_kernel_", BLK, "!")
    @eval begin
        @kernel function $kname(C, @Const(A), @Const(B),
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

            Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                c_i_j = zero(AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator})

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
                copyto!(pointer(C), 1 + coff + tm + (i - 1) * GEMM_TILE +
                                    (tn + (j - 1) * GEMM_TILE) * M + sk * M * N,
                        M, c_i_j)
        end
        GEMM_BLOCK_KERNELS[$BLK] = $kname
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
@kernel function strided_gemm_kernel!(C, @Const(A), @Const(B), ::Val{K},
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
@kernel function splitk_reduce_kernel!(C, @Const(Cp), ::Val{S}, n) where {S}
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
                       partials = nothing, reduce::Bool = true)
    backend = LavaBackend()
    blk, splitk = blk_split
    span = GEMM_TILE * blk
    ntiles = (M ÷ span) * (N ÷ span)
    dst = splitk == 1 ? C :
          (partials === nothing ? splitscratch(M, N, splitk * nbatch) : partials)
    GEMM_BLOCK_KERNELS[blk](backend, GEMM_WORKGROUP)(
        dst, A, B, Val(M), Val(N), Val(K),
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
