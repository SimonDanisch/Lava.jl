# Atomix.jl integration for Lava.jl
#
# Provides atomic references for Lava device arrays. UnsafeAtomics preserves
# the caller's order and scope; the emitter checks the Vulkan contract.
#
# On the GPU, LavaArray becomes Ptr{T} via LavaAdaptor. We need to make
# Atomix.@atomic work with raw Ptr{T}. The Atomix macro expands to:
#   Atomix.modify!(referenceable(arr)[i], +, v, seq_cst)[2]
# Since Ptr{T} isn't an AbstractArray, we provide:
# 1. A PtrRef wrapper as the "referenceable" for Ptr{T}
# 2. A PtrIndexableRef as the indexed reference
# 3. Atomix.modify! overrides that call our atomicrmw llvmcall

import Atomix
import UnsafeAtomics

# ── Ptr-based Atomix references ──

struct PtrRef{T}
    ptr::Ptr{T}
end

struct PtrIndexableRef{T}
    ptr::Ptr{T}
    index::Int
end

@lava_device_override function Atomix.Internal.referenceable(ptr::Ptr{T}) where T
    PtrRef{T}(ptr)
end

# LavaDeviceArray atomics: wrap with LavaAtomicRef to preserve dims for CartesianIndex
struct LavaAtomicRef{T, N}
    ptr::Ptr{T}
    dims::NTuple{N, Int}
end

@lava_device_override function Atomix.Internal.referenceable(a::LavaDeviceArray{T,N}) where {T,N}
    LavaAtomicRef{T,N}(a.ptr, a.dims)
end

@lava_device_override function Base.getindex(r::LavaAtomicRef{T}, i::Integer) where T
    PtrIndexableRef{T}(r.ptr, Int(i))
end

@lava_device_override function Base.getindex(r::LavaAtomicRef{T,N}, I::CartesianIndex{N}) where {T,N}
    li = linear_index(r.dims, I)
    PtrIndexableRef{T}(r.ptr, li)
end

# `Atomix.@atomic arr[i, j, ...]` expands to `referenceable(arr)[i, j, ...]`, so
# we need a getindex that accepts N integer args (one per dim).
@lava_device_override function Base.getindex(r::LavaAtomicRef{T,N}, I::Vararg{Integer,N}) where {T,N}
    li = linear_index(r.dims, CartesianIndex(map(Int, I)))
    PtrIndexableRef{T}(r.ptr, li)
end

# `linear_index` is `device/devicearray.jl`'s, included just before this file.
#
# No second copy here, and in particular not the naive form:
#
#     li = I[1]; stride = 1
#     for d in 2:N;  stride *= dims[d-1];  li += stride * (I[d] - 1);  end
#
# which is exactly the nested-product expansion `linear_index`'s Horner comment
# says NVIDIA's shader compiler miscompiles when the result feeds a
# PhysicalStorageBuffer offset. An `Atomix.@atomic a[i, j, k]` on a 3-D array
# with computed indices would have the same dropped-`I[1]` exposure that
# `repeat(x; inner)` is fixed for.

@lava_device_override function Base.getindex(r::PtrRef{T}, i::Integer) where T
    PtrIndexableRef{T}(r.ptr, Int(i))
end

@lava_device_override function Atomix.pointer(ref::PtrIndexableRef{T}) where T
    ref.ptr + (ref.index - 1) * sizeof(T)
end

@lava_device_override function Atomix.gcroot(ref::PtrIndexableRef{T}) where T
    nothing
end

@lava_device_override function Base.eltype(ref::PtrIndexableRef{T}) where T
    T
end

# UnsafeAtomics emits order- and scope-correct LLVM atomics. Do not replace
# its explicit ordering with a hard-coded LLVM constant. Vulkan's emitter
# rejects sequential consistency rather than silently reducing it to acq_rel.
# Float64 add/sub uses integer CAS: these devices do not enable native
# Float64 atomic add.
@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Float64}, op::Union{typeof(+),typeof(-)}, val::Float64, order, scope;
    volatile::Bool=false, align::Integer=sizeof(Float64)
)
    bits = Ptr{UInt64}(ptr)
    old = UnsafeAtomics.load(bits, UnsafeAtomics.monotonic, scope; volatile, align)
    while true
        new = op(reinterpret(Float64, old), val)
        result = UnsafeAtomics.cas!(bits, old, reinterpret(UInt64, new), order,
            UnsafeAtomics.monotonic, scope; volatile, align)
        result.success && return reinterpret(Float64, old) => new
        old = result.old
    end
end

# Keep the existing device minNum/maxNum policy and native Float32 operations,
# while forwarding the requested ordering/scope instead of hard-coding them.
@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Float32}, op::Union{typeof(min),typeof(max)}, val::Float32, order, scope;
    volatile::Bool=false, align::Integer=sizeof(Float32)
)
    f = op === min ? UnsafeAtomics.fmin : UnsafeAtomics.fmax
    return UnsafeAtomics.modify!(ptr, f, val, order, scope; volatile, align)
end

# ── Base.modifyindex_atomic! override for LavaDeviceArray ──
# Atomix 1.1.2+ @atomic macro expands to Base.modifyindex_atomic! on Julia 1.12+.
# Without this override, the kernel tries to call jl_f_throw_methoderror on GPU.

@lava_device_override @inline function Base.modifyindex_atomic!(
    arr::LavaDeviceArray{T,N}, order::Symbol, op::OP, val, i::Integer
) where {T, N, OP}
    v = convert(T, val)
    ref = Atomix.Internal.referenceable(arr)[i]
    return Atomix.modify!(ref, op, v, order)
end

@lava_device_override @inline function Base.modifyindex_atomic!(
    arr::LavaDeviceArray{T,N}, order::Symbol, op::OP, val, I::CartesianIndex{N}
) where {T, N, OP}
    v = convert(T, val)
    li = linear_index(arr.dims, I)
    ref = Atomix.Internal.referenceable(arr)[li]
    return Atomix.modify!(ref, op, v, order)
end

# N-dimensional indexing: `@atomic arr[i, j, ...]` expands to a call with
# one Integer per dimension. Convert to a linear index and forward.
@lava_device_override @inline function Base.modifyindex_atomic!(
    arr::LavaDeviceArray{T,N}, order::Symbol, op::OP, val, I::Vararg{Integer,N}
) where {T, N, OP}
    v = convert(T, val)
    li = linear_index(arr.dims, CartesianIndex(map(Int, I)))
    ref = Atomix.Internal.referenceable(arr)[li]
    return Atomix.modify!(ref, op, v, order)
end
