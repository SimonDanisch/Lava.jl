# Full GPUArrays.jl interface for LavaArray
#
# Provides broadcasting, scalar indexing, and the Adapt pipeline.
# Device-side representation: LavaDeviceArray{T,N} (isbits, BDA pointer + dims).

import GPUArrays
import Adapt

# ── Adapt.jl integration ──

# LavaAdaptor struct is declared in array/lavaarray.jl (needed earlier by
# ka_backend.jl method signatures).  This file defines the Adapt rules.
#
# The adaptor is the single place LavaArray → LavaDeviceArray (pointer strip)
# happens during kernel-arg conversion, so it is also the single place buffer-
# lifetime pinning fires.  Adapt.jl's recursion handles wrapper structs,
# Broadcasted, NamedTuple — every LavaArray anywhere inside lands here.

function Adapt.adapt_storage(ad::LavaAdaptor, a::LavaArray{T,N}) where {T,N}
    pin!(ad.batch, a)
    return LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
end
Adapt.adapt_storage(::LavaAdaptor, x) = x  # Scalars pass through

# Type-based adapt: Array → LavaArray (used by GPUArrays test suite's compare())
Adapt.adapt_storage(::Type{<:LavaArray}, xs::AT) where {AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(LavaArray, xs)
Adapt.adapt_storage(::Type{<:LavaArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(LavaArray{T}, xs)
Adapt.adapt_storage(::Type{Array}, xs::LavaArray) = convert(Array, xs)

# ── GPU-safe RefValue (isbits replacement for mutable Base.RefValue) ──

"""LavaRefValue: isbits wrapper replacing Base.RefValue for GPU kernels."""
struct LavaRefValue{T} <: Ref{T}
    x::T
end
Base.getindex(r::LavaRefValue) = r.x

# Generic RefValue → LavaRefValue (isbits, GPU-safe)
Adapt.adapt_structure(to::LavaAdaptor, r::Base.RefValue) = LavaRefValue(Adapt.adapt(to, r[]))

# RefValue{Type/DataType} → zero-size type wrapper (types are not values on GPU)
struct LavaRefType{T} <: Ref{DataType} end
Base.getindex(::LavaRefType{T}) where {T} = T
Adapt.adapt_structure(::LavaAdaptor, r::Base.RefValue{<:Union{DataType, Type}}) =
    LavaRefType{r[]}()

# Broadcasted with Type as function (e.g. T.(args...)) → lambda wrapper
Adapt.adapt_structure(to::LavaAdaptor,
                      bc::Base.Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Base.Broadcast.Broadcasted{Style}((x...) -> T(x...), Adapt.adapt(to, bc.args), bc.axes)

# ── Broadcast style ──

struct LavaArrayStyle{N} <: GPUArrays.AbstractGPUArrayStyle{N} end

LavaArrayStyle{M}(::Val{N}) where {M,N} = LavaArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::Type{<:LavaArray{T,N}}) where {T,N} = LavaArrayStyle{N}()

# Promote structured matrix styles (Diagonal, Tridiagonal, etc.) to LavaArrayStyle
# so that broadcasting with Diagonal{T, LavaArray{T,1}} runs on GPU.
Base.Broadcast.BroadcastStyle(::LavaArrayStyle{N}, ::LinearAlgebra.StructuredMatrixStyle{T}) where {N,T} = LavaArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{T}, ::LavaArrayStyle{N}) where {N,T} = LavaArrayStyle{N}()

# Wrapped GPU arrays (Transpose, Adjoint, SubArray) should use LavaArrayStyle
Base.Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Transpose{T, <:LavaArray{T}}}) where T = LavaArrayStyle{2}()
Base.Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{T, <:LavaArray{T}}}) where T = LavaArrayStyle{2}()
Base.Broadcast.BroadcastStyle(::Type{<:SubArray{T, N, <:LavaArray}}) where {T, N} = LavaArrayStyle{N}()

# Allocate broadcast output
function Base.similar(bc::Base.Broadcast.Broadcasted{LavaArrayStyle{N}}, ::Type{T}, dims) where {T,N}
    # dims can be axes (OneTo) or plain integers
    LavaArray{T}(undef, map(Base.to_dim, dims))
end

# Let GPUArrays choose between linear (1D) and cartesian (multi-D) broadcast kernels.
# LavaDeviceArray supports both linear and CartesianIndex indexing.

# ── Scalar indexing ──
# GPUArrays.jl provides Base.getindex(::AbstractGPUArray, ::Int) and
# Base.setindex!(::AbstractGPUArray, v, ::Int) which delegate to copyto!.
# Our copyto! implementations (below) handle the actual data transfer.

# ── copyto! variants for CPU↔GPU transfers ──

function Base.copyto!(dest::LavaArray{T}, doffs::Integer,
                      src::Array{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    bytes = Vector{UInt8}(undef, n * sizeof(T))
    unsafe_copyto!(Ptr{UInt8}(pointer(bytes)), Ptr{UInt8}(pointer(src, soffs)), n * sizeof(T))
    # upload! records into the active batch and flushes; sync_access! takes care
    # of any prior cross-queue writer to `dest.buf[]` via timeline semaphore.
    upload!(dest.buf[], bytes; offset=(dest.offset + Int(doffs) - 1) * sizeof(T))
    return dest
end

function Base.copyto!(dest::Array{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    # download! records a copy into the active batch of whichever queue last
    # wrote `src.buf[]` and flushes — sync_access! inserts any cross-queue wait.
    bytes = Vector{UInt8}(undef, n * sizeof(T))
    download!(bytes, src.buf[]; offset=(src.offset + Int(soffs) - 1) * sizeof(T))
    unsafe_copyto!(Ptr{UInt8}(pointer(dest, doffs)), Ptr{UInt8}(pointer(bytes)), n * sizeof(T))
    return dest
end

function Base.copyto!(dest::LavaArray{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    # Direct GPU→GPU copy via vkCmdCopyBuffer (no CPU staging roundtrip).
    src_offset = src.buf[].pool_offset + (src.offset + Int(soffs) - 1) * sizeof(T)
    dst_offset = dest.buf[].pool_offset + (dest.offset + Int(doffs) - 1) * sizeof(T)
    nbytes = n * sizeof(T)
    bq = (dest.buf[].ctx::VkContext).default_bq
    cmd_copy_buffer!(bq, src.buf[], dest.buf[], nbytes;
                     src_off=src_offset, dst_off=dst_offset)
    flush!(bq, bq.device)
    return dest
end

# ── fill! ──

function Base.fill!(a::LavaArray{T}, val) where T
    length(a) == 0 && return a
    v = convert(T, val)
    @kernel function fill_kernel!(A, v)
        I = @index(Global)
        @inbounds A[I] = v
    end
    k = fill_kernel!(LavaBackend())
    k(a, v; ndrange=length(a))
    return a
end

# ── resize! ──

function Base.resize!(a::LavaArray{T,1}, n::Integer) where T
    n < 0 && throw(ArgumentError("new size must be non-negative"))
    old_len = length(a)
    n == old_len && return a
    ctx = a.buf[].ctx::VkContext
    bq = ctx.default_bq
    new_buf = pool_alloc(ctx, max(Int(n) * sizeof(T), 16))
    if old_len > 0 && n > 0
        copy_len = min(old_len, Int(n)) * sizeof(T)
        src_off = a.buf[].pool_offset + a.offset * sizeof(T)
        cmd_copy_buffer!(bq, a.buf[], new_buf, copy_len;
                         src_off=src_off, dst_off=new_buf.pool_offset)
        flush!(bq, bq.device)
    end
    new_ref = GPUArrays.DataRef(new_buf) do buf
        vk_free!(buf)
    end
    old_ref = a.buf
    a.buf = new_ref
    a.dims = (Int(n),)
    a.offset = 0
    # Free old buffer — DataRef uses explicit refcounting, not GC finalizers,
    # so we must release the old ref to avoid leaking VkDeviceMemory.
    GPUArrays.unsafe_free!(old_ref)
    return a
end

# ── append! ──

function Base.append!(a::LavaArray{T,1}, items::AbstractVector) where T
    items_arr = collect(T, items)
    old_len = length(a)
    new_len = old_len + length(items_arr)
    resize!(a, new_len)
    # Upload the new items to the end
    if !isempty(items_arr)
        bytes = Vector{UInt8}(reinterpret(UInt8, vec(items_arr)))
        upload!(a.buf[], bytes; offset=old_len * sizeof(T))
    end
    return a
end

# ── norm override ──
# GPUArrays' norm uses closures that capture variables as Core.Box (non-isbits),
# causing compilation failures in the rescaling branch. Override with separate
# helper functions to ensure all captured variables are isbits.

import LinearAlgebra

# Float16/Float32: accumulate in Float64 to avoid overflow entirely (no rescaling needed)
function lava_norm_p2_widen(v::LavaArray{T}) where T
    s = sum(x -> Float64(abs(x))^2, v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(sqrt(s))
end

function lava_norm_p1_widen(v::LavaArray{T}) where T
    s = sum(x -> Float64(abs(x)), v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(s)
end

function lava_norm_pp_widen(v::LavaArray{T}, spp::Float64) where T
    # Use exp2(p*log2(x)) instead of x^p to avoid ^ operator dispatch issues on GPU
    s = sum(x -> exp2(spp * log2(Float64(abs(x)))), v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(exp2(inv(spp) * log2(s)))
end

# Float64: rescale by inv_maxabs to avoid overflow
function lava_norm_p2_rescale(v::LavaArray{T}, inv_maxabs::Float64, maxabs::Float64) where T
    s = sum(x -> (abs(x) * inv_maxabs)^2, v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(maxabs * sqrt(s))
end

function lava_norm_pp_rescale(v::LavaArray{T}, inv_maxabs::Float64, maxabs::Float64, spp::Float64) where T
    # Use exp2(p*log2(x)) instead of x^p to avoid ^ operator dispatch issues on GPU
    s = sum(x -> exp2(spp * log2(abs(x) * inv_maxabs)), v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(maxabs * exp2(inv(spp) * log2(s)))
end

function LinearAlgebra.norm(v::LavaArray{T}, p::Real=2) where T
    RT = typeof(float(LinearAlgebra.norm(zero(T))))
    isempty(v) && return zero(RT)
    p == 0 && return convert(RT, count(!iszero, v))
    p == Inf && return convert(RT, maximum(abs, v))
    p == -Inf && return convert(RT, minimum(abs, v))
    # Non-trivial p-norms rely on transcendental math in the reduction path.
    # These are currently not numerically stable across Vulkan drivers (e.g. RDNA3.5),
    # so use CPU fallback for correctness/portability.
    (p != 1 && p != 2) && return convert(RT, LinearAlgebra.norm(Array(v), p))

    # Float16/Float32/ComplexF16/ComplexF32: accumulate in Float64 (no overflow possible)
    if RT === Float32 || RT === Float16
        p == 2 && return lava_norm_p2_widen(v)
        p == 1 && return lava_norm_p1_widen(v)
        return lava_norm_pp_widen(v, Float64(p))
    end

    # Float64/ComplexF64: rescale by max to avoid overflow
    maxabs = convert(Float64, maximum(abs, v))
    (iszero(maxabs) || isinf(maxabs)) && return convert(RT, maxabs)

    # Try without rescaling first (faster, only when no overflow AND no underflow)
    if p == 2 && isfinite(length(v) * maxabs^2) && !iszero(maxabs^2)
        s = sum(abs2, v; init=Float64(0))
        return convert(RT, sqrt(s))
    end

    inv_maxabs = 1.0 / maxabs
    p == 2 && return lava_norm_p2_rescale(v, inv_maxabs, maxabs)
    p == 1 && return convert(RT, sum(abs, v; init=Float64(0)))
    return lava_norm_pp_rescale(v, inv_maxabs, maxabs, Float64(p))
end
