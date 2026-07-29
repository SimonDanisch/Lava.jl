# AcceleratedMatrix — an `AbstractMatrix` whose elements live in subgroup-scope
# matrix hardware (tensor cores) rather than in Julia.
#
# The operations it does support are ordinary Base functions -- `size`, `zero`,
# `*`, `muladd`, `copyto!` -- given device-specific implementations by the
# compiler. It is not an `AbstractMatrix`, because it has no addressable
# elements; see the docstring.
#
# It is *opaque*, so it holds no element storage. A cooperative matrix's value
# is distributed across the lanes of a subgroup with an implementation-defined
# layout; it lives in registers the emitter names, never in a Julia field. The
# single `handle` field is not data — it is an SSA anchor. A zero-field struct
# would be erased by LLVM and two distinct matrices would collapse into one
# value, so `muladd(a, b, c)` needs something per-instance to keep them apart.
# The emitter maps each handle to an `OpCooperativeMatrix*` result id.
#
# There is deliberately no second "scalar" implementation inside this type. On a
# device without cooperative matrices the fallback is `MMatrix`, which is
# already a perfectly good `AbstractMatrix`. Write kernels generically over
# `AbstractMatrix` and instantiate with whichever the device supports -- that is
# lava-dnn.md's "capability dispatch selected at instantiate from a device
# query", with no dispatch machinery of our own.
#
# `Use` records which operand position the matrix occupies in `A*B + C`. SPIR-V
# fixes it when the type is created, so it cannot be inferred afterwards -- the
# same way `Adjoint` and `Transpose` are part of the type in Base.

"""Which operand position a matrix occupies in `A*B + C`."""
abstract type MatrixUse end
struct MatrixA <: MatrixUse end
struct MatrixB <: MatrixUse end
struct Accumulator <: MatrixUse end

"""
    AcceleratedMatrix{T,M,N,Use}

An `M×N` matrix of `T` held in subgroup-scope matrix hardware.

Supports `size`, `zero`, `*`, `muladd`, `copyto!` and the
`(src, offset, stride)` constructor — ordinary Base functions, lowered by the
compiler to cooperative-matrix instructions.

Deliberately NOT an `AbstractMatrix`: there is no addressable `(row, column)`.
The layout across the subgroup is implementation-defined, and the only element
access SPIR-V exposes is into *this invocation's* fragment, whose length bears
no fixed relation to `(M, N)`. Subtyping `AbstractMatrix` would promise an
indexing contract the hardware cannot honour, and generic Base code would then
fail deep inside a fallback instead of at the call site. Use `MMatrix` when you
need element access — which is also the fallback on devices without
cooperative matrices.

Device-only: the value is spread across a subgroup's lanes and cannot be
allocated host-side or outlive the invocation, like `@private`/`@localmem`.
"""
struct AcceleratedMatrix{T,M,N,Use<:MatrixUse}
    # SSA anchor, not element storage — see the note above.
    handle::Int32
end

Base.size(::AcceleratedMatrix{T,M,N}) where {T,M,N} = (M, N)

matrixuse(::Type{<:AcceleratedMatrix{T,M,N,U}}) where {T,M,N,U} = U
matrixuse(m::AcceleratedMatrix) = matrixuse(typeof(m))

"""
    coopmat_shape(ctx, T, M, N, K) -> Bool

Whether the running device implements this tile. The hardware supports a fixed
set of `(M, N, K, dtype)` combinations, so a kernel picks one of those or uses
`MMatrix` instead.
"""
# VkComponentTypeKHR. Only the types a cooperative-matrix operand can currently
# have in Lava; an unmapped type reports "no such shape" rather than matching one
# by accident.
_vk_component_type(::Type{Float16}) = UInt32(0)
_vk_component_type(::Type{Float32}) = UInt32(1)
_vk_component_type(::Type{Float64}) = UInt32(2)
_vk_component_type(::Type{Int8})    = UInt32(3)
_vk_component_type(::Type{Int16})   = UInt32(4)
_vk_component_type(::Type{Int32})   = UInt32(5)
_vk_component_type(::Type{Int64})   = UInt32(6)
_vk_component_type(::Type{UInt8})   = UInt32(7)
_vk_component_type(::Type{UInt16})  = UInt32(8)
_vk_component_type(::Type{UInt32})  = UInt32(9)
_vk_component_type(::Type{UInt64})  = UInt32(10)
_vk_component_type(::Type) = nothing

const VK_SCOPE_SUBGROUP = UInt32(3)

# `T` is the A/B operand type, and it used to be accepted and then ignored: the
# match was on M, N and K alone. A device can report the same extents for
# completely different component types — this one lists 16x16x16 for
# (Float16 -> Float32), (Float16 -> Float16), (UInt8 -> Int32) and
# (Int8 -> Int32) — so an extent-only match says "yes" for Float16 on hardware
# that only does the integer forms, and the kernel then emits cooperative-matrix
# instructions the device cannot execute.
#
# The accumulator type is not checked because the signature does not carry one;
# callers that care (`coopmat_gemm_available`) rely on the operand type plus the
# extents, which is what distinguishes the shapes in practice.
function coopmat_shape(ctx::VkContext, ::Type{T}, M::Integer, N::Integer,
                       K::Integer) where {T}
    ctx.coopmat_available || return false
    want = _vk_component_type(T)
    want === nothing && return false
    any(s -> s.M == M && s.N == N && s.K == K &&
             s.ab_type == want && s.scope == VK_SCOPE_SUBGROUP,
        ctx.coopmat_shapes)
end

# ── Device operations ─────────────────────────────────────────────────────────
# Each of these is a single call site the emitter recognises and replaces with
# the corresponding SPIR-V instruction. Outside a kernel they have no meaning,
# exactly like the ray-query and subgroup intrinsics.

"""
    AcceleratedMatrix{T,M,N,Use}(src, offset, stride)

Load an `M×N` tile from `src` starting at `offset`, `stride` elements between
consecutive columns. Lowers to `OpCooperativeMatrixLoadKHR`.
"""
@inline AcceleratedMatrix{T,M,N,U}(src, offset::Integer, stride::Integer) where {T,M,N,U} =
    coopmat_load(AcceleratedMatrix{T,M,N,U}, src, offset, stride)

# The `@localmem` forms of these live in `array/ka_backend.jl`, beside
# `LavaSharedArray` — this file is included before that type exists.

"""
    copyto!(dst, offset, stride, m)

Store the tile back. Lowers to `OpCooperativeMatrixStoreKHR`.
"""
@inline Base.copyto!(dst, offset::Integer, stride::Integer, m::AcceleratedMatrix) =
    coopmat_store(dst, offset, stride, m)

@inline Base.zero(::Type{AcceleratedMatrix{T,M,N,U}}) where {T,M,N,U} =
    coopmat_zero(AcceleratedMatrix{T,M,N,U})
@inline Base.zero(::AcceleratedMatrix{T,M,N,U}) where {T,M,N,U} =
    zero(AcceleratedMatrix{T,M,N,U})

"""
    muladd(a, b, c)

`a*b + c`, accumulated in `c`'s element type. Lowers to
`OpCooperativeMatrixMulAddKHR` — this is the instruction that reaches the
tensor cores.
"""
@inline Base.muladd(a::AcceleratedMatrix{TA,M,K,MatrixA},
                    b::AcceleratedMatrix{TB,K,N,MatrixB},
                    c::AcceleratedMatrix{TC,M,N,Accumulator}) where {TA,TB,TC,M,N,K} =
    coopmat_muladd(a, b, c)

@inline Base.:*(a::AcceleratedMatrix{TA,M,K,MatrixA},
                b::AcceleratedMatrix{TB,K,N,MatrixB}) where {TA,TB,M,N,K} =
    muladd(a, b, zero(AcceleratedMatrix{promote_type(TA, TB),M,N,Accumulator}))
