# The array a KERNEL receives, and how a kernel indexes it. Nothing here is
# Vulkan's.
#
# Split out of `array/lavaarray.jl`, which holds the host-side `LavaArray` — the
# buffer, its lifetime, the pool it came from and the transfers. Those are the
# runtime's and are Mantle's now. This is not: a `(pointer, dims)` pair is what
# the compiler emits loads and stores against, `device/atomics.jl` builds
# `Atomix` references over it, and neither needs a device to exist.
#
# The two are still a pair — `LavaDeviceArray(a::LavaArray)` is beside the host
# type, because it is that type that knows its own buffer address.
#
# The indexing below arrived late, and how it was missed is worth recording. It
# had always lived in `array/ka_backend.jl`, whose other 1100 lines are launch,
# dispatch and queue — so the file moved to Mantle whole and took `getindex`,
# `setindex!` and `linear_index` with it. Nothing complained. Lava still
# *compiled* every kernel; the bodies just inferred to `Union{}`, which
# `replace_unreachable` turns into `ret void`, so the emitter produced a valid,
# empty entry point and `spirv-val` passed it. A kernel that does nothing is not
# a compile error anywhere in the pipeline — only Tier 1's `check(d, "OpFAdd")`
# says so.
#
# The rule the miss came from ignoring: a method a KERNEL BODY calls is device
# vocabulary and belongs here, whatever file it was sitting in.

"""
    LavaDeviceArray{T,N}

Device-side isbits array representation for GPU kernels.

The pointer is a Vulkan buffer device address, which is why this is a plain
`Ptr{T}`: BDA is a 64-bit number in the shader's address space, so a kernel
indexes it exactly as it would host memory and the emitter needs no separate
descriptor path.
"""
struct LavaDeviceArray{T,N} <: GPUArrays.AbstractDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Int}
end

Base.size(a::LavaDeviceArray) = a.dims
Base.pointer(a::LavaDeviceArray{T}) where {T} = a.ptr
Base.pointer(a::LavaDeviceArray{T}, i::Integer) where {T} = a.ptr + (i - 1) * sizeof(T)
Base.unsafe_convert(::Type{Ptr{T}}, a::LavaDeviceArray{T}) where {T} = a.ptr

# ── Device-side indexing for GPU arrays ──

# 0D array indexing (used by broadcasting on scalar-wrapped arrays)
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,0}) where T
    unsafe_load(a.ptr)
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,0}, v) where T
    unsafe_store!(a.ptr, convert(T, v))
    return v
end

# 0D with CartesianIndex (broadcast dispatches CartesianIndex{1} on 0D arrays)
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,0}, I::CartesianIndex) where T
    unsafe_load(a.ptr)
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,0}, v, I::CartesianIndex) where T
    unsafe_store!(a.ptr, convert(T, v))
    return v
end

# LavaDeviceArray{T,N} linear indexing via BDA pointer
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T}, i::Integer) where T
    unsafe_load(a.ptr, i)
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T}, v::T, i::Integer) where T
    unsafe_store!(a.ptr, v, i)
    return v
end
@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T}, v, i::Integer) where T
    unsafe_store!(a.ptr, convert(T, v), i)
    return v
end

# CartesianIndex indexing for multi-dimensional broadcast support (any N)
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,N}, I::CartesianIndex{N}) where {T,N}
    @inbounds unsafe_load(a.ptr, linear_index(a.dims, I))
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
    @inbounds unsafe_store!(a.ptr, convert(T, v), linear_index(a.dims, I))
    return v
end

# Multi-index `a[i, j, k, …]` must route through the same Horner `linear_index`
# as the CartesianIndex path above. Without these, Base's generic fallback
# (`_to_linear_index` → `_sub2ind`) builds precisely the naive nested-product
# expansion documented below, and the NVIDIA shader compiler miscompiles it into
# an out-of-bounds PhysicalStorageBuffer offset.
#
# It only misfires once the indices are *computed* rather than literal, so it
# hides from simple tests: `a[1,2,2,1]` constant-folds and is fine, while
# `a[is[1], is[2], 2, 1]` with `is` from a delinearised loop index reads garbage.
# That is how it reached GPUArrays' `vectorized_getindex` (host/indexing.jl:84),
# silently truncating every strided `a[:, :, 2:2, :]`.
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,N},
                                                     i1::Integer, i2::Integer,
                                                     Ir::Vararg{Integer}) where {T,N}
    @inbounds unsafe_load(a.ptr, linear_index(a.dims, CartesianIndex(i1, i2, Ir...)))
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,N}, v,
                                                      i1::Integer, i2::Integer,
                                                      Ir::Vararg{Integer}) where {T,N}
    @inbounds unsafe_store!(a.ptr, convert(T, v),
                            linear_index(a.dims, CartesianIndex(i1, i2, Ir...)))
    return v
end

# Trailing singleton indices. Julia lets you index an array with MORE indices
# than it has dimensions as long as the extras are 1 — `v[i, 1]` is valid for a
# Vector, `A[i, j, 1]` for a Matrix — and generic kernels rely on it rather than
# specialising on ndims. GPUArrays' `gpu_kron_kernel!` does exactly that: it
# indexes `a[i, j]` where `a` is whichever operand it was handed, so
# `kron(vec(x), y)` passes a 1-D array and the kernel fails to compile with a
# method lookup failure inside `linear_index` — for every element type, since
# nothing about it is type-specific.
#
# Only M > N is handled. M < N (fewer indices than dimensions, where the last
# index spans the remaining ones) is a different rule and no caller here needs
# it; the branch is resolved at compile time, so the error never reaches a
# shader.
@inline function linear_index(dims::NTuple{N,Int}, I::CartesianIndex{M}) where {N,M}
    if M > N
        linear_index(dims, CartesianIndex(ntuple(i -> I[i], Val(N))))
    else
        error("linear_index: $M indices for $N dimensions is not supported")
    end
end

# Convert CartesianIndex to linear index for LavaDeviceArray
@inline function linear_index(dims::NTuple{1,Int}, I::CartesianIndex{1})
    I[1]
end
@inline function linear_index(dims::NTuple{2,Int}, I::CartesianIndex{2})
    I[1] + dims[1] * (I[2] - 1)
end
# Horner form. The naive column-major expansion
#   I[1] + dims[1]*(I[2]-1) + dims[1]*dims[2]*(I[3]-1) + …
# forms a standalone `dims[1]*dims[2]` product (and shares `dims[1]` across two
# terms). NVIDIA's shader compiler miscompiles exactly that shape when the result
# feeds a PhysicalStorageBuffer load offset: it drops the I[1] term, so every read
# lands on source row 1. Horner factoring keeps each stride coefficient as a single
# value applied to a running sum, never materialising the nested product, which the
# driver evaluates correctly. Symptom was `repeat(x; inner)` with a 3-D `inner`
# (GPUArrays repeat_inner_dst_kernel! → xs[CartesianIndex(sdx)]). Pinned by
# test_repeat_inner_3d.jl. Same NVIDIA complex-integer family as the unswitch and
# shared-PSB-access-chain miscompiles.
@inline function linear_index(dims::NTuple{3,Int}, I::CartesianIndex{3})
    I[1] + dims[1] * ((I[2] - 1) + dims[2] * (I[3] - 1))
end
@inline function linear_index(dims::NTuple{N,Int}, I::CartesianIndex{N}) where N
    off = I[N] - 1
    @inbounds for d in (N-1):-1:1
        off = off * dims[d] + (I[d] - 1)
    end
    return off + 1
end

# Ptr{T} indexing overrides are intentionally absent — device kernels never
# receive raw Ptr{T} args any more.  All array-like kernel parameters come
# through LavaAdaptor as LavaDeviceArray and use the overrides above.
