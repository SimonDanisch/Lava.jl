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

"""
    adapt_storage(::LavaAdaptor, ::LavaArray) → LavaDeviceArray

Pure strip: wrap the GPU buffer address in a device-visible struct. **No pin
side effect**. Callers must separately invoke `pin_leaves!(batch, args...)`
before submitting a command buffer that uses the stripped result — otherwise
the underlying `VkManagedBuffer` can be GC'd before the GPU reads from it.

(AMDGPU's adapt is also pure; they get away without an explicit pin pass
because `@roc` uses `GC.@preserve vars...` lexically around a synchronous
dispatch. Lava's dispatches are batched and submitted later, so `@preserve`
isn't enough — hence the explicit `pin_leaves!` step.)
"""
function Adapt.adapt_storage(::LavaAdaptor, a::LavaArray{T,N}) where {T,N}
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

# Wrapped GPU arrays (Transpose, Adjoint, SubArray, PermutedDimsArray) should
# use LavaArrayStyle. Without an entry here the wrapper falls back to
# DefaultArrayStyle, which iterates and therefore hits the scalar-indexing
# guard — so `PermutedDimsArray(a, perm) .+ 1` threw rather than running.
Base.Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Transpose{T, <:LavaArray{T}}}) where T = LavaArrayStyle{2}()
Base.Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{T, <:LavaArray{T}}}) where T = LavaArrayStyle{2}()
Base.Broadcast.BroadcastStyle(::Type{<:SubArray{T, N, <:LavaArray}}) where {T, N} = LavaArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::Type{<:PermutedDimsArray{T, N, <:Any, <:Any, <:LavaArray}}) where {T, N} = LavaArrayStyle{N}()

# Wrappers compose: `reshape(PermutedDimsArray(a, perm), dims)` is a
# ReshapedArray over a PermutedDimsArray over a LavaArray, and matching only on
# a direct LavaArray parent leaves it on DefaultArrayStyle.
const AnyLavaArray{T} = Union{LavaArray{T},
                              SubArray{T, <:Any, <:LavaArray},
                              PermutedDimsArray{T, <:Any, <:Any, <:Any, <:LavaArray},
                              Base.ReshapedArray{T, <:Any, <:LavaArray},
                              LinearAlgebra.Transpose{T, <:LavaArray},
                              LinearAlgebra.Adjoint{T, <:LavaArray}}
Base.Broadcast.BroadcastStyle(::Type{<:Base.ReshapedArray{T, N, <:AnyLavaArray}}) where {T, N} = LavaArrayStyle{N}()

# ── broadcast: always launch over a flat index space ──
#
# GPUArrays' `_copyto!` launches with `ndrange = size(dest)`, so an N-D
# destination gets an N-D index space and KernelAbstractions partitions it into
# N-D workgroups — at which point consecutive lanes no longer walk consecutive
# memory. The cost is not subtle. `D .= A .+ B` over 16.7M Float16:
#
#   dest 1-D          315 GB/s     (CUDA.jl: 325)
#   dest (512,512,64)  54 GB/s     (CUDA.jl: 204)
#
# Same kernel, same data, same card — only the launch geometry differs. CUDA.jl
# never sees this because it overrides broadcast with its own flat grid-stride
# loop rather than going through the KA path.
#
# So: flatten the range, and recover the Cartesian index inside the kernel when
# the broadcast genuinely needs one. That div/mod chain is far cheaper than
# losing coalescing. This is the single largest thing in Lava for anything
# elementwise — `add`, `relu`, `sigmoid`, `_to_copy`, `cat` and every kernel
# epilogue in LavaDNN were all running at a quarter of the achievable bandwidth.
@kernel function lava_broadcast_flat!(dest, bc, n)
    I = @index(Global, Linear)
    if I <= n
        @inbounds dest[I] = bc[I]
    end
end

# Narrowing the linear->Cartesian `divrem` to 32 bits here was tried and
# reverted. It is the same trick that took `im2col_kernel!` (four divisions per
# element) from 35.7 ms to 32.7 on a whole step, but handing these kernels a
# `CartesianIndices` with `Int32` axes silently produced wrong results — a view
# operand off by 252, a permuted operand off by 18 — and widening the recovered
# index back to `Int` immediately afterwards did not fix it, so the fault is in
# the narrowed `CartesianIndices` itself rather than in what `Broadcasted` does
# with the index type. Worth revisiting with a hand-rolled 32-bit decomposition
# instead of `CartesianIndices`, but not without these four cases as a test.
"""
Cartesian index for 0-based linear position `q`, given the extents as `Int32`.

Hand-rolled rather than `CartesianIndices[...]`: the divisions happen in 32-bit,
which NVIDIA has hardware for and 64-bit it does not, but the components are
widened to `Int` as they are built. Handing a `CartesianIndices` with `Int32`
axes to these kernels instead is *miscompiled* by Lava — see
`test/test_int32_cartesian_miscompile.jl` — while this form is correct.

The recursion is over the tuple's type, so it unrolls completely; SPIR-V's ban
on recursive call graphs does not apply.
"""
@inline cart32(q::Int32, ::Tuple{}) = ()
@inline function cart32(q::Int32, sz::Tuple{Int32,Vararg{Int32}})
    s = first(sz)
    (Int(q % s) + 1, cart32(q ÷ s, Base.tail(sz))...)
end

@kernel function lava_broadcast_flat_cartesian!(dest, bc, sz, n)
    I = @index(Global, Linear)
    if I <= n
        J = CartesianIndex(cart32(Int32(I) - Int32(1), sz))
        @inbounds dest[J] = bc[J]
    end
end

"""Destination dense, source not: index `dest` linearly and only `bc` by index."""
@kernel function lava_broadcast_flat_mixed!(dest, bc, sz, n)
    I = @index(Global, Linear)
    if I <= n
        @inbounds dest[I] = bc[CartesianIndex(cart32(Int32(I) - Int32(1), sz))]
    end
end

"""
Can every operand be read with the destination's own linear index?

`IndexStyle(::Broadcasted)` is too conservative for the case that dominates a
DNN — every operand the same dense shape as the destination, no extrusion — and
answering it directly is what gets those onto the linear kernel. Operands are
inspected before `preprocess` wraps them in `Extruded`; the `Extruded` method is
there for when one is passed anyway.
"""
flatok(dest, x) = true                                  # scalars, refs, functions
flatok(dest, x::AbstractArray) =
    size(x) == size(dest) && IndexStyle(x) === IndexLinear()
flatok(dest, x::Broadcast.Extruded) = flatok(dest, x.x)
flatok(dest, x::Broadcast.Broadcasted) = all(a -> flatok(dest, a), x.args)

"""
Rebuild a broadcast tree over 1-D reshapes of its operands.

Launching over a flat range is not enough on its own: `getindex(::Broadcasted,
::Integer)` on an N-D tree is defined as `bc[CartesianIndices(bc)[i]]`, so the
div/mod chain comes back per element even inside a linear kernel. Reshaping the
leaves to vectors makes the tree genuinely 1-D and the index arithmetic
disappears. Only valid when every leaf already has the destination's shape,
which is what `flatok` establishes.
"""
flat1(x) = x
flat1(x::AbstractArray) = reshape(x, length(x))
flat1(bc::Broadcast.Broadcasted) = Broadcast.broadcasted(bc.f, map(flat1, bc.args)...)

function GPUArrays._copyto!(dest::AnyLavaArray, bc::Broadcast.Broadcasted)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    n = length(dest)
    backend = LavaBackend()
    if ndims(dest) > 1 && IndexStyle(dest) === IndexLinear() && flatok(dest, bc)
        flat = Broadcast.instantiate(flat1(bc))
        d1 = reshape(dest, n)
        lava_broadcast_flat!(backend)(d1, Broadcast.preprocess(d1, flat), n; ndrange = n)
        return dest
    end
    linear = ndims(dest) == 1
    bc = Broadcast.preprocess(dest, bc)
    # Extents as `Int32` for `cart32`; anything Lava can address fits.
    sz = map(Int32, size(dest))
    if linear
        lava_broadcast_flat!(backend)(dest, bc, n; ndrange = n)
    elseif IndexStyle(dest) === IndexLinear()
        lava_broadcast_flat_mixed!(backend)(dest, bc, sz, n; ndrange = n)
    else
        lava_broadcast_flat_cartesian!(backend)(dest, bc, sz, n; ndrange = n)
    end
    return dest
end

# GPUArrays implements `repeat` for `AnyGPUArray` (host/base.jl `repeat_inner`
# / `repeat_outer`), but `AnyGPUArray` only recognises a *single* layer of
# wrapping, so a `ReshapedArray{PermutedDimsArray{LavaArray}}` misses it and
# falls back to `Base._RepeatInnerOuter.repeat_outer`, which indexes
# elementwise and trips the scalar-indexing guard. Detaching the wrapper first
# gets it back onto the device kernels; `repeat` allocates its output anyway, so
# the copy costs nothing that was not already being paid.
const NestedLavaWrapper = Union{
    Base.ReshapedArray{<:Any, <:Any, <:Union{SubArray{<:Any, <:Any, <:LavaArray},
                                             PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:LavaArray}}},
    SubArray{<:Any, <:Any, <:Union{Base.ReshapedArray{<:Any, <:Any, <:LavaArray},
                                   PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:LavaArray}}},
    PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:Union{Base.ReshapedArray{<:Any, <:Any, <:LavaArray},
                                                          SubArray{<:Any, <:Any, <:LavaArray}}},
}

function Base.repeat(x::NestedLavaWrapper; inner=nothing, outer=nothing)
    dense = similar(x)
    dense .= x
    repeat(dense; inner, outer)
end

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
    upload!(dest.buf[], bytes; offset=dest.offset + (Int(doffs) - 1) * sizeof(T))
    return dest
end

function Base.copyto!(dest::Array{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    # download! records a copy into the active batch of whichever queue last
    # wrote `src.buf[]` and flushes — sync_access! inserts any cross-queue wait.
    bytes = Vector{UInt8}(undef, n * sizeof(T))
    download!(bytes, src.buf[]; offset=src.offset + (Int(soffs) - 1) * sizeof(T))
    unsafe_copyto!(Ptr{UInt8}(pointer(dest, doffs)), Ptr{UInt8}(pointer(bytes)), n * sizeof(T))
    return dest
end

function Base.copyto!(dest::LavaArray{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    # Direct GPU→GPU copy via vkCmdCopyBuffer (no CPU staging roundtrip).
    src_offset = src.buf[].pool_offset + src.offset + (Int(soffs) - 1) * sizeof(T)
    dst_offset = dest.buf[].pool_offset + dest.offset + (Int(doffs) - 1) * sizeof(T)
    nbytes = n * sizeof(T)
    bq = (dest.buf[].ctx::VkContext).default_bq
    cmd_copy_buffer!(bq, src.buf[], dest.buf[], nbytes;
                     src_off=src_offset, dst_off=dst_offset)
    # No flush: this is device→device, so nothing on the host needs the result.
    # `cmd_copy_buffer!` records the transfer-write→shader-read barrier itself,
    # which is all the ordering a later dispatch in this batch needs. The flush
    # that used to be here drained the whole GPU on every `copy(::LavaArray)`.
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
#
# Capacity-aware resize for LavaArrays, both 1-D (standard Base.resize! shape)
# and N-D (non-standard but the natural extension; textures are 2-D/3-D and
# `copyto_texture!` needs it).  Invariants:
#
# * The backing VkManagedBuffer is pool-allocated with rounded-up capacity.
#   If the new element count still fits, we just update `dims` — no Vulkan
#   alloc, no copy.  Elements beyond old length are left undefined, matching
#   `Base.resize!(::Vector, n)` semantics.
# * On genuine growth, allocate a fresh pool chunk, copy the first min(old,new)
#   elements, and retire the old DataRef via `GPUArrays.unsafe_free!`.  The
#   retirement enters `bq.deferred_frees` gated on the batch timeline — the
#   caller does NOT need a `synchronize` to make it safe w.r.t. in-flight work.
# * Throws on a freed DataRef (surfaces caller bugs; no self-heal).

function Base.resize!(a::LavaArray{T,N}, new_dims::Dims{N}) where {T,N}
    all(>=(0), new_dims) || throw(ArgumentError("new size must be non-negative"))
    a.buf.freed && throw(ArgumentError(
        "resize!(::LavaArray{$T,$N}): the backing DataRef was already freed. " *
        "This array was unsafe_free!'d elsewhere — typically an HW-accel rebuild " *
        "that released state, or a cached slot whose owner was torn down."))
    Tuple(a.dims) == new_dims && return a
    old_len = length(a)
    new_len = prod(new_dims)

    buf = a.buf[]
    capacity_bytes = buf.size - a.offset
    if new_len * sizeof(T) <= capacity_bytes
        a.dims = new_dims
        return a
    end

    ctx = buf.ctx::VkContext
    bq = ctx.default_bq
    new_buf = pool_alloc(bq, max(new_len * sizeof(T), 16))
    if old_len > 0 && new_len > 0
        copy_len = min(old_len, new_len) * sizeof(T)
        src_off = buf.pool_offset + a.offset
        cmd_copy_buffer!(bq, buf, new_buf, copy_len;
                         src_off=src_off, dst_off=new_buf.pool_offset)
        # The copy pinned `buf` into the currently-recording batch. `vk_free!`
        # decides whether to defer destruction by inspecting `buf.last_write`,
        # but `last_write` is only populated by `sync_access!` at submit time
        # — between record-pin and submit it reads stale, so `vk_free!` would
        # free-immediately and the DEAD buffer would trip `sync_access!`'s
        # ALIVE assertion when the batch eventually submits.  Flushing here
        # closes that window: the copy submits + completes, `last_write` is
        # set, then `unsafe_free!` below correctly sees the buffer as in-use
        # and routes through `deferred_frees`.  Only fires in the slow grow
        # path (capacity exceeded) — within-capacity resizes are zero-sync.
        flush!(bq, bq.device)
    end
    new_ref = GPUArrays.DataRef(new_buf) do b
        vk_free!(b)
    end
    old_ref = a.buf
    a.buf = new_ref
    a.dims = new_dims
    a.offset = 0
    GPUArrays.unsafe_free!(old_ref)
    return a
end

# Variadic / scalar dispatch forwarders (N-D and 1-D calling conventions).
Base.resize!(a::LavaArray{T,N}, new_dims::Vararg{Integer,N}) where {T,N} =
    resize!(a, Dims{N}(Int.(new_dims)))
Base.resize!(a::LavaArray{T,1}, n::Integer) where T = resize!(a, (Int(n),))

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

# Float64: rescale by maxabs to avoid overflow.
#
# Subtlety: writing `abs(x) / maxabs` in the source is NOT enough. GPU drivers
# (and LLVM's `arcp` fast-math) lower a Float64 `x / m` to `x * (1/m)`, so when
# `maxabs` is near `floatmax(Float64)` the reciprocal `1/maxabs` is a subnormal
# (~1.1e-308, below the 2.2e-308 normal floor). Under flush-denormals-to-zero —
# the default on lavapipe AND RDNA3 (RADV) — that reciprocal becomes 0, then
# `abs(x) * 0 = 0`, `sum = 0`, and the rescaled norm collapses to 0. (Verified:
# `sum(x -> x/m, v)` returns exactly 0.0 iff `1/m` is subnormal, 2.0 otherwise.)
#
# Fix: divide by `sqrt(maxabs)` TWICE instead of by `maxabs` once.
# `(x / √m) / √m == x / m`, but `1/√m` is always normal (even for m == floatmax,
# `1/√floatmax ≈ 7.5e-155`), so no subnormal reciprocal is ever formed. This is
# correct for the overflow case (maxabs near floatmax), the underflow case
# (maxabs near floatmin), and normal magnitudes alike.
function lava_norm_p2_rescale(v::LavaArray{T}, maxabs::Float64) where T
    sm = sqrt(maxabs)
    s = sum(x -> ((abs(x) / sm) / sm)^2, v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(maxabs * sqrt(s))
end

function lava_norm_pp_rescale(v::LavaArray{T}, maxabs::Float64, spp::Float64) where T
    # Use exp2(p*log2(x)) instead of x^p to avoid ^ operator dispatch issues on GPU
    s = sum(x -> exp2(spp * log2(abs(x) / maxabs)), v; init=Float64(0))
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

    p == 2 && return lava_norm_p2_rescale(v, maxabs)
    p == 1 && return convert(RT, sum(abs, v; init=Float64(0)))
    return lava_norm_pp_rescale(v, maxabs, Float64(p))
end
