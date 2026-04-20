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

    # Register `unsafe_free!` as a GC finalizer so LavaArrays that fall out
    # of scope without an explicit `unsafe_free!` call actually release their
    # GPU memory. `GPUArrays.DataRef` is refcount-based — its `release`
    # callback only fires on explicit `unsafe_free!`, NOT via Julia GC — so
    # WITHOUT this finalizer every scratch LavaArray leaks until the process
    # exits. Matches AMDGPU.ROCArray's pattern (`finalizer(unsafe_free!, xs)`
    # at `AMDGPU/src/array.jl:13,19`). Idempotency is guaranteed by the
    # `DataRef.freed` flag + `VkManagedBuffer.state` atomic CAS, so a second
    # `unsafe_free!` (e.g. explicit user call before GC) is a no-op.
    function LavaArray{T,N}(buf::GPUArrays.DataRef{VkManagedBuffer}, dims::NTuple{N,Int};
                            offset::Integer=0) where {T,N}
        xs = new{T,N}(buf, dims, offset)
        finalizer(unsafe_free!, xs)
        return xs
    end
end

function LavaArray{T,N}(::UndefInitializer, dims::NTuple{N,Int};
                        bq::BatchQueue=vk_context().default_bq,
                        extra_usage::UInt32=UInt32(0),
                        unified::Bool=false) where {T,N}
    ctx = bq.ctx::VkContext
    nbytes = prod(dims) * sizeof(T)

    # Alignment is a function of `extra_usage` alone (see `bda_alignment_for`).
    # We over-allocate by (align - 1) bytes and shift the LavaArray's element
    # offset so `bda_address(arr)` lands on the right boundary.  For T=UInt8
    # the offset is always valid; otherwise we assert divisibility.
    align = bda_alignment_for(ctx, extra_usage)
    slack = align > 1 ? Int(align) - 1 : 0

    # Non-default usage (index buffer, AS input/storage, scratch, etc.) or
    # BAR-mapped (unified) memory bypasses the pool since those buffers need
    # specific flags at VkBuffer/DeviceMemory creation time.
    use_pool = extra_usage == UInt32(0) && !unified
    managed_buf = use_pool ?
        pool_alloc(bq, max(nbytes + slack, 16)) :
        vk_alloc(bq, max(nbytes + slack, 16); extra_usage, unified)

    elem_offset = 0
    if align > UInt64(1)
        base = managed_buf.address
        aligned = cld(base, align) * align
        byte_offset = Int(aligned - base)
        @assert byte_offset % sizeof(T) == 0  "extra_usage alignment ($(align)) is not a multiple of sizeof($T)"
        elem_offset = byte_offset ÷ sizeof(T)
    end

    ref = GPUArrays.DataRef(managed_buf) do buf
        vk_free!(buf)
    end
    LavaArray{T,N}(ref, dims; offset=elem_offset)
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

# Construct from host data.  Forwards alloc kwargs (bq/extra_usage/unified)
# so callers don't need a separate "alloc-then-upload" helper for each usage.
function LavaArray{T,N}(data::AbstractArray{T,N}; kw...) where {T,N}
    arr = LavaArray{T,N}(undef, size(data); kw...)
    GC.@preserve arr upload!(arr, data)
    return arr
end
LavaArray(data::AbstractArray{T,N}; kw...) where {T,N} = LavaArray{T,N}(data; kw...)

LavaArray{T}(data::AbstractArray{S,N}; kw...) where {T,S,N} =
    LavaArray{T,N}(convert(AbstractArray{T}, data); kw...)
LavaArray{T,N}(data::AbstractArray{S,N}; kw...) where {T,S,N} =
    LavaArray{T,N}(convert(AbstractArray{T}, data); kw...)

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

# pin! the LavaArray wrapper itself, NOT just its VkManagedBuffer. The wrapper
# is what Julia's GC traces: pinning the raw VkManagedBuffer wouldn't keep the
# LavaArray alive, so its GC finalizer could fire mid-batch, call `unsafe_free!`
# on the underlying DataRef, and leave the wavefront using a DEFERRED/DEAD
# buffer — which `sync_access!` rightly asserts against.
# `sync_access!(::LavaArray)` below forwards to the underlying VkManagedBuffer
# so cross-queue last_write tracking still runs on the leaf.
@inline pin!(batch::CommandBatch, a::LavaArray) = begin
    a in batch.pinned && return
    push!(batch.pinned, a)
    return nothing
end

@inline sync_access!(batch::CommandBatch, a::LavaArray) = sync_access!(batch, a.buf[])

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

Update a GPU array with new host data.  Grow-only policy:

* If `data` fits the backing VkManagedBuffer's existing capacity, the GPU
  buffer is reused (no Vulkan alloc/free churn); only `dst.dims` is updated
  and the bytes are overwritten.
* If `data` exceeds capacity, allocate a new buffer sized for the new data
  and release the old one through the DataRef refcount path.

This is the right shape for animation loops that update per-frame data
whose size fluctuates but has a bounded peak (streamplot tubes, meshscatter
triangle lists, dye volumes, etc.) — steady state allocates zero.
"""
function update!(dst::LavaArray{T,N}, data::AbstractArray{T,N}) where {T,N}
    new_bytes = length(data) * sizeof(T)
    buf = dst.buf[]
    # Capacity = backing buffer's allocated size minus our element offset.
    # For non-pooled buffers that's buf.size; for pooled chunks it's also
    # buf.size (pool carves exact-size sub-allocations).
    capacity_bytes = buf.size - dst.offset * sizeof(T)

    if new_bytes <= capacity_bytes
        # Reuse buffer, just set new dims and overwrite the bytes.
        dst.dims = size(data)
        upload!(dst, data)
    else
        # Need to grow: release old DataRef, allocate a fresh buffer sized for `data`.
        old_ref = dst.buf
        ctx = buf.ctx::VkContext
        GPUArrays.unsafe_free!(old_ref)
        new_nbytes = max(new_bytes, 16)
        new_buf = pool_alloc(ctx.default_bq, new_nbytes)
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
