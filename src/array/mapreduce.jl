# GPU mapreducedim! for LavaArray
#
# Fast path: Float32 sum via a single-dispatch Vulkan-native reduce using
# OpGroupNonUniformFAdd (subgroup reduce) + OpAtomicFAddEXT to a BAR-mapped
# scalar. One dispatch, one fence wait, no partial-array temp, no CPU
# readback ping-pong — roughly 4-5× faster than the AK path on big arrays.
#
# Slow path: AcceleratedKernels.jl for everything else (partial-dim reductions,
# non-sum ops, non-Float32). KA-based tree reduction that iterates to 1 via
# repeated partial-reduce kernels, each forcing a sync.

import GPUArrays
import AcceleratedKernels as AK
using KernelAbstractions
const _KA_reduce = KernelAbstractions
using Atomix

# ── Vulkan-native single-dispatch sum ──

@kernel cpu=false function _vk_reduce_fadd_kernel!(out, @Const(src), total_threads::Int32)
    gi = _KA_reduce.@index(Global, Linear)
    n = length(src)
    # Strided: each thread walks the array in steps of `total_threads`. Cuts
    # atomic pressure by `n / total_threads` vs "1 thread per element" — at
    # n=10M with ~16k threads, ~160 elements/thread → ~256 atomics/step
    # instead of 156k.
    acc = 0f0
    i = gi
    @inbounds while i <= n
        acc += src[i]
        i += total_threads
    end
    # Subgroup (wave) reduce — single hardware instruction.
    s = subgroup_add(acc)
    # One atomic per wave, from the first active lane.
    if subgroup_elect()
        Atomix.@atomic out[1] += s
    end
    nothing
end

"""
    reduce_scratch(ctx) -> LavaArray{Float32,1}

The one-cell scratch buffer `vk_reduce_sum` writes its result into, on `ctx`.

Singleton because allocating a `LavaArray` per call was most of the per-call
overhead at small sizes. One cell of BAR-mapped memory; its `mapped_ptr` stays
valid for the lifetime of the context, which is now literally true — it is a
field of the context, so it dies with it and no reset callback has to remember.

It was an `IdDict` keyed by context, and the allocation went to `vk_context()`
rather than to `ctx`, so with two devices live the entry stored under the SECOND
context held a buffer belonging to the first. Keyed right, allocated wrong,
which reads as correct until there are two devices.
"""
@inline function reduce_scratch(ctx::VkContext)
    buf = ctx.caches.reduce_scratch
    buf === nothing || return buf::LavaArray{Float32,1}
    buf = LavaArray{Float32}(undef, (1,); unified=true, bq=ctx.default_bq)
    ctx.caches.reduce_scratch = buf
    return buf
end

"""
    vk_reduce_sum(A::LavaArray{Float32}) -> Float32

Vulkan-native single-pass reduction sum for Float32. Launches one compute
dispatch that does:
  per-thread load → subgroup_add → atomic_fadd into a BAR-mapped scalar.

ONE fence wait, result read directly from mapped memory. Replaces the AK
tree-reduce path's multiple CPU readbacks.
"""
function vk_reduce_sum(A::LavaArray{Float32})
    n = length(A)
    ctx = A.buf[].ctx
    out = reduce_scratch(ctx)
    # Zero the mapped cell directly — skips a fill! dispatch.
    out_ptr = Base.unsafe_convert(Ptr{Float32}, out.buf[].mapped_ptr)
    unsafe_store!(out_ptr, 0f0)
    # Fixed thread count — tuned for RX 7900 XTX-class GPUs. Too few threads
    # underutilizes SMs; too many causes atomic contention on out[1]. With
    # ~16k threads, n=10M → ~625 elems/thread → ~256 atomics → fast.
    # Tuning (RX 7900 XTX, n=10M benchmarked): wgsize=64 keeps one subgroup
    # per workgroup → each workgroup's atomic fires once.  nblocks=2048 saturates
    # the 96 CUs with enough overlap to hide memory latency. At this point we hit
    # ~85% of peak memory bandwidth for Float32 sum; smaller arrays are limited
    # by the fixed dispatch+fence overhead (~45 μs).
    wgsize = 64
    nblocks = min(cld(n, wgsize), 2048)
    total_threads = Int32(nblocks * wgsize)
    ndr = Int(total_threads)
    _vk_reduce_fadd_kernel!(KA.get_backend(A), wgsize)(out, A, total_threads; ndrange=ndr)
    bq = ctx.default_bq
    vk_flush!(bq)
    # out is mapped — read directly.
    return unsafe_load(out_ptr)
end

function GPUArrays.mapreducedim!(f::F, op::OP, R::LavaArray{T}, A::AbstractArray;
                                  init=nothing) where {F, OP, T}
    mapreducedim_ak!(f, op, R, A; init)
    return R
end

function GPUArrays.mapreducedim!(f::F, op::OP, R::LavaArray{T},
                                  A::Base.Broadcast.Broadcasted;
                                  init=nothing) where {F, OP, T}
    # Materialize broadcasted to LavaArray first — AK expects AbstractArray
    A_mat = Base.materialize(A)
    mapreducedim_ak!(f, op, R, A_mat; init)
    return R
end

function mapreducedim_ak!(f::F, op::OP, R::LavaArray{T}, A;
                            init=nothing) where {F, OP, T}
    n = length(A)
    n == 0 && return R

    init_val = if init !== nothing
        convert(T, init)
    else
        GPUArrays.neutral_element(op, T)
    end

    if length(R) == 1
        # Full reduction (dims=:)
        #
        # Fast path: f=identity, op=+, eltype Float32, input is a plain
        # LavaArray{Float32} — one-dispatch Vulkan-native reduce with
        # OpGroupNonUniformFAdd + atomic_fadd. Handles ~85% of peak memory
        # bandwidth; AK's tree-reduce is ~2× slower because of per-level
        # scalar readbacks. Any non-matching case falls through to AK.
        # `Base.add_sum` is `Base.sum`'s default op (=== +, but different fn object).
        if T === Float32 && (op === (+) || op === Base.add_sum) && f === identity && A isa LavaArray{Float32}
            result = vk_reduce_sum(A::LavaArray{Float32}) + init_val
            R_host = Float32[result]
            copyto!(R, 1, R_host, 1, 1)
            return R
        end
        # Fallback: AK.mapreduce to scalar
        result = AK.mapreduce(f, op, A, KA.get_backend(A);
                              init=init_val, neutral=init_val,
                              block_size=64, switch_below=0)
        # Write scalar result into R
        R_host = T[convert(T, result)]
        copyto!(R, 1, R_host, 1, 1)
    else
        # Partial reduction (dims=N) — determine which dim is being reduced.
        # AK.mapreduce needs a dense array, so wrappers (views, PermutedDimsArray,
        # ...) get materialised — on the *device*, via broadcast. This used to be
        # `convert(LavaArray, collect(A))`, i.e. a blocking download to the host
        # followed by a re-upload; `sum(view(x, ...); dims=)` alone was 21% of a
        # DNNKernels inference step.
        A_arr = A isa LavaArray ? A : densify(A)
        rdim = find_reduced_dim(size(A_arr), size(R))
        if rdim !== nothing
            # AK.mapreduce with dims= expects temp to have same ndims as input
            # with the reduced dim collapsed to 1. If R has fewer dims (implicit
            # singleton), reshape it.
            R_temp = if ndims(R) < ndims(A_arr)
                expected_sz = ntuple(i -> i == rdim ? 1 : size(A_arr, i), ndims(A_arr))
                reshape(R, expected_sz)
            else
                R
            end
            AK.mapreduce(f, op, A_arr, KA.get_backend(R_temp);
                         init=init_val, neutral=init_val,
                         dims=rdim, temp=R_temp, block_size=64)
        elseif size(A_arr) == size(R)
            # No dimension reduced: R and A have same shape (e.g., dims=[]).
            if init !== nothing
                fill!(R, init_val)
                R .= op.(R, f.(A_arr))
            else
                R .= f.(A_arr)
            end
        else
            # Multiple dims reduced or ndims mismatch — reduce sequentially
            # along each reduced dimension
            multi_dim_reduce!(f, op, R, A_arr, init_val)
        end
    end
    return R
end

"""Reduce along multiple dimensions by iterating over each reduced dim."""
function multi_dim_reduce!(f::F, op::OP, R::LavaArray{T}, A::LavaArray,
                            init_val) where {F, OP, T}
    # Find all dimensions that need reducing (size went to 1 or was dropped)
    szA = size(A)
    szR = size(R)
    ndA = ndims(A)
    ndR = ndims(R)

    # Pad R's size to match A's dimensionality
    szR_padded = ntuple(i -> i <= ndR ? szR[i] : 1, ndA)

    # Reduce one dimension at a time
    current = A
    first_reduction = true
    for d in ndA:-1:1
        if szA[d] != szR_padded[d] && szR_padded[d] == 1
            # Apply f only on the first reduction step
            map_fn = first_reduction ? f : identity
            first_reduction = false

            # Compute target size for this reduction step
            new_sz = ntuple(ndA) do i
                i == d ? 1 : size(current, i)
            end
            if new_sz == size(R)
                # Last reduction — write directly into R
                AK.mapreduce(map_fn, op, current, KA.get_backend(R);
                             init=init_val, neutral=init_val, dims=d, temp=R, block_size=64)
            else
                temp = LavaArray{T}(undef, new_sz)
                fill!(temp, init_val)
                AK.mapreduce(map_fn, op, current, KA.get_backend(temp);
                             init=init_val, neutral=init_val, dims=d, temp=temp, block_size=64)
                current = temp
            end
        end
    end
    return R
end

"""Find which single dimension was reduced (size went to 1).
Handles implicit singleton dimensions (e.g., (2,2) → (2,) means dim 2 reduced)."""
function find_reduced_dim(szA, szR)
    ndA = length(szA)
    ndR = length(szR)

    # Pad R's size with trailing 1s to match A's dimensionality
    szR_padded = ntuple(i -> i <= ndR ? szR[i] : 1, ndA)

    rdim = nothing
    for i in 1:ndA
        if szA[i] != szR_padded[i]
            if szR_padded[i] == 1 && rdim === nothing
                rdim = i
            else
                return nothing  # Multiple dims or unexpected shape
            end
        end
    end
    return rdim
end

# ── Sort ──
# There is deliberately no `AK.merge_sort_by_key!` override here.
#
# One used to exist, implementing sort-by-key as sortperm + permute, because
# AK's block-level merge kernel reads shared-memory positions it never wrote
# when `len < 2 * block_size`, and Vulkan leaves workgroup memory undefined.
# That override was circular: AK implements `sortperm` *via* `merge_sort_by_key!`
# (see AcceleratedKernels/src/sort/merge_sortperm.jl), so the two called each
# other forever — a StackOverflowError, or 34 GB of pool growth and
# ERROR_OUT_OF_DEVICE_MEMORY when the recursion allocated temporaries first.
#
# The real defect was the uninitialized shared memory, and that is now fixed at
# the source: the Workgroup Block variable is emitted with an OpConstantNull
# initializer (see `emit_workgroup_block!` in compiler/compilation.jl), so every
# kernel starts with zeroed shared memory and AK's own implementation is correct.
