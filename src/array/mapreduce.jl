# GPU mapreducedim! for LavaArray
#
# Delegates to AcceleratedKernels.jl — proven cross-architecture KA-based
# reduction kernels that work on any KernelAbstractions backend.

import GPUArrays
import AcceleratedKernels as AK

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
        # Full reduction (dims=:) → AK.mapreduce to scalar
        result = AK.mapreduce(f, op, A, LavaBackend();
                              init=init_val, neutral=init_val,
                              block_size=64, switch_below=0)
        # Write scalar result into R
        R_host = T[convert(T, result)]
        copyto!(R, 1, R_host, 1, 1)
    else
        # Partial reduction (dims=N) — determine which dim is being reduced
        A_arr = A isa LavaArray ? A : convert(LavaArray, collect(A))
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
            AK.mapreduce(f, op, A_arr, LavaBackend();
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
                AK.mapreduce(map_fn, op, current, LavaBackend();
                             init=init_val, neutral=init_val, dims=d, temp=R, block_size=64)
            else
                temp = LavaArray{T}(undef, new_sz)
                fill!(temp, init_val)
                AK.mapreduce(map_fn, op, current, LavaBackend();
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

# ── Sort override ──
# AK.merge_sort_by_key! uses shared memory that must be zero-initialized on
# Vulkan (workgroup memory is undefined per spec). The block-level merge kernel
# reads uninitialized positions when len < 2*block_size, producing wrong results.
# Workaround: implement via sortperm + permute which uses correct kernels.
function AK.merge_sort_by_key!(
    keys::LavaArray, values::LavaArray, backend::LavaBackend=LavaBackend();
    lt=isless, by=identity, rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward, kwargs...
)
    perm = AK.sortperm(keys, backend; lt, by, rev, order)
    KA.synchronize(backend)
    sorted_keys = keys[perm]
    sorted_vals = values[perm]
    copyto!(keys, sorted_keys)
    copyto!(values, sorted_vals)
    return keys, values
end
