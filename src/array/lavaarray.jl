# LavaArray{T,N} — GPU array backed by Vulkan buffer
#
# Full GPUArrays.jl compatible implementation with DataRef, offset, derive.

import GPUArraysCore: AbstractGPUArray, AbstractGPUVector, AbstractGPUMatrix

"""
    LavaArray{T,N} <: AbstractGPUArray{T,N}

GPU array backed by a Vulkan device-local buffer with BDA (Buffer Device Address).
"""
mutable struct LavaArray{T,N} <: AbstractGPUArray{T,N}
    buf::GPUArrays.DataRef{VkManagedBuffer}
    dims::NTuple{N,Int}
    offset::Int  # offset in number of elements (not bytes)

    # No finalizer on LavaArray itself: lifetime is delegated entirely to
    # `GPUArrays.DataRef`'s refcount, which runs the `vk_free!` closure when
    # refcount → 0.  Registering a second finalizer here would mean two
    # independent vk_free! calls could race on the same VkManagedBuffer.
    # `VkManagedBuffer.state` atomic CAS covers that corner, but keeping a
    # single ownership path is simpler and matches AMDGPU.jl's model.
    function LavaArray{T,N}(buf::GPUArrays.DataRef{VkManagedBuffer}, dims::NTuple{N,Int};
                            offset::Integer=0) where {T,N}
        return new{T,N}(buf, dims, offset)
    end
end

function LavaArray{T,N}(::UndefInitializer, dims::NTuple{N,Int};
                        ctx::VkContext=vk_context(),
                        extra_usage::UInt32=UInt32(0)) where {T,N}
    nbytes = prod(dims) * sizeof(T)
    # Non-default usage (index buffer, AS input/storage, scratch, etc.)
    # bypasses the pool since those buffers need specific usage flags at
    # VkBuffer creation time. Default `extra_usage=0` uses the pool.
    managed_buf = extra_usage == UInt32(0) ?
        pool_alloc(ctx, max(nbytes, 16)) :
        vk_alloc(ctx, max(nbytes, 16); extra_usage)
    ref = GPUArrays.DataRef(managed_buf) do buf
        vk_free!(buf)
    end
    LavaArray{T,N}(ref, dims)
end

# Varargs constructor for LavaArray{T,N}(undef, d1, d2, ...)
LavaArray{T,N}(::UndefInitializer, dims::Int...; kw...) where {T,N} = LavaArray{T,N}(undef, dims; kw...)
LavaArray{T,N}(::UndefInitializer, dims::Integer...; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)
LavaArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)

"""Allocate a LavaArray with INDEX_BUFFER_BIT for use as Vulkan index buffer."""
alloc_index_buffer(data::AbstractVector{UInt32}) = begin
    arr = LavaArray{UInt32,1}(undef, (length(data),);
        extra_usage=UInt32(Vulkan.BUFFER_USAGE_INDEX_BUFFER_BIT))
    upload!(arr, data)
    return arr
end

# Empty vector constructor (matches Array{T,1}() behavior)
LavaArray{T,1}() where {T} = LavaArray{T,1}(undef, (0,))

LavaArray{T}(::UndefInitializer, dims::NTuple{N,Int}; kw...) where {T,N} = LavaArray{T,N}(undef, dims; kw...)
LavaArray{T}(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {T,N} = LavaArray{T,N}(undef, Int.(dims); kw...)
LavaArray{T}(::UndefInitializer, dims::Integer...; kw...) where {T} = LavaArray{T}(undef, Int.(dims); kw...)

# Construct from host data
function LavaArray{T,N}(data::AbstractArray{T,N}) where {T,N}
    arr = LavaArray{T,N}(undef, size(data))
    GC.@preserve arr upload!(arr, data)
    return arr
end
LavaArray(data::AbstractArray{T,N}) where {T,N} = LavaArray{T,N}(data)

# Type-converting constructors: LavaArray{T}(array_of_S)
function LavaArray{T}(data::AbstractArray{S,N}) where {T,S,N}
    LavaArray{T,N}(convert(AbstractArray{T}, data))
end
function LavaArray{T,N}(data::AbstractArray{S,N}) where {T,S,N}
    LavaArray{T,N}(convert(AbstractArray{T}, data))
end

# UniformScaling constructor (resolve ambiguity with GPUArrays inner constructor)
import LinearAlgebra: UniformScaling
function (::Type{LavaArray{T,N}})(s::UniformScaling, dims::Tuple{Int,Int}) where {T,N}
    res = similar(LavaArray{T,N}, dims)
    fill!(res, zero(T))
    isempty(res) && return res
    @kernel function identity_kernel!(res, stride, val)
        i = @index(Global, Linear)
        ilin = (stride * (i - 1)) + i
        if ilin <= length(res)
            @inbounds res[ilin] = val
        end
    end
    kernel = identity_kernel!(LavaBackend())
    kernel(res, size(res, 1), T(s.λ); ndrange=minimum(dims))
    return res
end
(::Type{LavaArray{T}})(s::UniformScaling, dims::Tuple{Int,Int}) where {T} =
    LavaArray{T,2}(s, dims)

# ── GPUArrays interface: storage & derive ──

GPUArrays.storage(a::LavaArray) = a.buf

function GPUArrays.derive(::Type{T}, a::LavaArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.buf)
    offset += (a.offset * Base.elsize(a)) ÷ sizeof(T)
    LavaArray{T,N}(ref, dims; offset)
end

# ── copy ──

function Base.copy(a::LavaArray{T,N}) where {T,N}
    b = similar(a)
    copyto!(b, 1, a, 1, length(a))
    return b
end

# ── similar ──

Base.similar(a::LavaArray{T,N}) where {T,N} = LavaArray{T,N}(undef, a.dims)
Base.similar(a::LavaArray{T}, dims::Base.Dims{N}) where {T,N} = LavaArray{T,N}(undef, dims)
Base.similar(a::LavaArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = LavaArray{T,N}(undef, dims)

# ── Base interface ──

Base.size(a::LavaArray) = a.dims
Base.length(a::LavaArray) = prod(a.dims)
Base.sizeof(a::LavaArray{T}) where T = length(a) * sizeof(T)
Base.eltype(::LavaArray{T}) where T = T
Base.ndims(::LavaArray{T,N}) where {T,N} = N
Base.IndexStyle(::Type{<:LavaArray}) = IndexLinear()
Base.elsize(::Type{<:LavaArray{T}}) where T = sizeof(T)

# BDA address for kernel argument passing (includes offset)
function bda_address(a::LavaArray{T}) where T
    a.buf[].address + a.offset * sizeof(T)
end

# pin! forwarder: unwrap LavaArray's DataRef to its VkManagedBuffer leaf
# so the ctx assertion + last_write tracking happen on the actual GPU buffer.
@inline pin!(batch::CommandBatch, a::LavaArray) = pin!(batch, a.buf[])

# LavaAdaptor: converts LavaArray → LavaDeviceArray (Ptr-wrapping) for GPU
# kernel compilation, and pins every visited LavaArray into the current batch.
# Declared here (ahead of ka_backend.jl which uses it in method signatures);
# the `adapt_storage` / `adapt_structure` methods live in gpuarrays.jl.
struct LavaAdaptor
    batch::CommandBatch
end

# ── Transfers ──

"""Upload host data to GPU array (must have matching length)."""
function upload!(dst::LavaArray{T}, data::AbstractArray{T}) where T
    @assert length(data) == length(dst) "Size mismatch: $(length(data)) vs $(length(dst))"
    src = vec(collect(data))
    nbytes = length(src) * sizeof(T)
    bytes = Vector{UInt8}(undef, nbytes)
    GC.@preserve src bytes begin
        unsafe_copyto!(Ptr{UInt8}(pointer(bytes)), Ptr{UInt8}(pointer(src)), nbytes)
    end
    GC.@preserve dst upload!(dst.buf[], bytes; offset=dst.offset * sizeof(T))
end

"""
    update!(dst::LavaArray, data::AbstractArray)

Update a GPU array with new host data. Resizes if needed — frees the old
buffer immediately (no GC pressure) and allocates a new one.
If sizes match, uploads in-place with no allocation.
"""
function update!(dst::LavaArray{T,N}, data::AbstractArray{T,N}) where {T,N}
    if size(dst) == size(data)
        upload!(dst, data)
    else
        # Free old buffer immediately (not deferred)
        old_ref = dst.buf
        GPUArrays.unsafe_free!(old_ref)
        # Allocate new (reuse the old buf's ctx so updates stay on the same device).
        ctx = dst.buf[].ctx::VkContext
        new_nbytes = max(length(data) * sizeof(T), 16)
        new_buf = pool_alloc(ctx, new_nbytes)
        new_ref = GPUArrays.DataRef(new_buf) do buf
            vk_free!(buf)
        end
        dst.buf = new_ref
        dst.dims = size(data)
        dst.offset = 0
        upload!(dst, data)
    end
    return dst
end
update!(dst::LavaArray{T}, data::AbstractArray{S}) where {T,S} = update!(dst, convert(AbstractArray{T}, data))

"""Download GPU array to host."""
function Base.Array(src::LavaArray{T,N}) where {T,N}
    result = Array{T}(undef, src.dims...)
    download_typed!(vec(result), src.buf[]; offset=src.offset * sizeof(T))
    return result
end

"""Convenience: collect downloads to host."""
Base.collect(a::LavaArray) = Array(a)

# ── Memory management ──

function unsafe_free!(a::LavaArray)
    GPUArrays.unsafe_free!(a.buf)
end

# ── Device-side array (isbits, passed to GPU kernels) ──

"""
    LavaDeviceArray{T,N}

Device-side isbits array representation for GPU kernels.
Contains a Ptr{T} (actually a BDA address) and dimensions.
"""
struct LavaDeviceArray{T,N} <: GPUArrays.AbstractDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Int}
end

Base.size(a::LavaDeviceArray) = a.dims
Base.pointer(a::LavaDeviceArray{T}) where {T} = a.ptr
Base.pointer(a::LavaDeviceArray{T}, i::Integer) where {T} = a.ptr + (i - 1) * sizeof(T)
Base.unsafe_convert(::Type{Ptr{T}}, a::LavaDeviceArray{T}) where {T} = a.ptr

# Convert LavaArray → LavaDeviceArray for kernel arguments
function LavaDeviceArray(a::LavaArray{T,N}) where {T,N}
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
end
