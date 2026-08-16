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

# `MatrixUse` (MatrixA/MatrixB/Accumulator) and `MatrixScope`
# (SubgroupScope/WorkgroupScope) are defined in `KernelInterfaces` and imported
# at the top of `Lava.jl`. They live there because Mantle has to name the same
# concepts to record what a device supports, and Mantle cannot depend on Lava —
# it weak-depends on it for `MantleLavaExt`, so an edge back would close a cycle.
#
# They are part of the type here for the reason they were when they lived in this
# file: SPIR-V fixes both when the type is created, so neither can be inferred
# afterwards, the same way `Adjoint` is part of a type in Base. `MatrixScope`
# additionally decides how many components each invocation holds.

"""
    CoopMatrix{T,M,N,Use,Scope}

An `M×N` matrix of `T` held in matrix hardware. Spell it as one of the two
aliases rather than directly:

    AcceleratedMatrix{T,M,N,Use}   # = CoopMatrix{...,SubgroupScope}, the portable one
    WorkgroupMatrix{T,M,N,Use}     # = CoopMatrix{...,WorkgroupScope}, NVIDIA-only

`AcceleratedMatrix` is the default because subgroup scope is `VK_KHR_cooperative_matrix`,
which AMD's RDNA3 WMMA path also has. Workgroup scope comes from
`VK_NV_cooperative_matrix2` (`vk_context().coopmat2.workgroup_scope`), so a kernel
using it has to be selected against a device query — which is why it has to be
written out rather than reached by default.

Every operation requires its operands to agree on scope. Mixing them is then a
method error at the call site instead of a module the driver rejects.
"""
struct CoopMatrix{T,M,N,Use<:MatrixUse,Scope<:MatrixScope}
    # SSA anchor, not element storage — see the note above.
    handle::Int32
end

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
const AcceleratedMatrix{T,M,N,U} = CoopMatrix{T,M,N,U,SubgroupScope}

"""
    WorkgroupMatrix{T,M,N,Use}

An `M×N` matrix of `T` spread across **every invocation of the workgroup**, not
one subgroup — `Scope = Workgroup` on `OpTypeCooperativeMatrixKHR`, which
`VK_NV_cooperative_matrix2` is what enables.

The reason it exists is register pressure. A `32 x 64` fp32 accumulator is 64
components per lane at subgroup scope and 8 at workgroup scope with 256
invocations, and the subgroup-scope flash kernel is gated by a step at 128
registers. It is also what lets a kernel drop its shared-memory staging
entirely: one matrix per operand, no per-subgroup tiling to write.

**The workgroup size is part of the contract.** The legal `(M, N, K)` granularity
depends on how many invocations the workgroup has — this device wants 256 for
`M=32, N=32, K=16` and reports the whole table through
`get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv`. Launch
with a different size and the driver rejects the pipeline. Uses must also be
workgroup-uniform: every invocation reaches every operation, or the result is
undefined.
"""
const WorkgroupMatrix{T,M,N,U} = CoopMatrix{T,M,N,U,WorkgroupScope}

Base.size(::CoopMatrix{T,M,N}) where {T,M,N} = (M, N)

matrixuse(::Type{<:CoopMatrix{T,M,N,U}}) where {T,M,N,U} = U
matrixuse(m::CoopMatrix) = matrixuse(typeof(m))

matrixscope(::Type{<:CoopMatrix{T,M,N,U,S}}) where {T,M,N,U,S} = S
matrixscope(m::CoopMatrix) = matrixscope(typeof(m))

"""
    coopmat_shape(ctx, T, M, N, K) -> Bool

Whether the running device implements this tile. The hardware supports a fixed
set of `(M, N, K, dtype)` combinations, so a kernel picks one of those or uses
`MMatrix` instead.
"""
# VkComponentTypeKHR. Only the types a cooperative-matrix operand can currently
# have in Lava; an unmapped type reports "no such shape" rather than matching one
# by accident.
#
# The table is written once, in one direction, and the inverse is derived from it
# — two hand-written tables are two chances for one entry to disagree, and the
# symptom would be a shape silently reported as a type it is not.
const VK_COMPONENT_TYPES = (Float16, Float32, Float64, Int8, Int16, Int32, Int64,
                            UInt8, UInt16, UInt32, UInt64)

vkcomponenttype(::Type{T}) where {T} =
    (i = findfirst(==(T), VK_COMPONENT_TYPES); i === nothing ? nothing : UInt32(i - 1))

"The Julia type a `VkComponentTypeKHR` code names, or `nothing` if Lava has none."
juliacomponenttype(code::Integer) =
    (1 <= code + 1 <= length(VK_COMPONENT_TYPES)) ? VK_COMPONENT_TYPES[code+1] : nothing

const VK_SCOPE_SUBGROUP = UInt32(3)
const VK_SCOPE_WORKGROUP = UInt32(2)

"""
    matrixshapes(ctx) -> Vector{MatrixShape}

The driver's cooperative-matrix table as [`MatrixShape`](@ref)s.

Lava has always held this — `ctx.coopmat_shapes`, queried at device creation —
and then thrown all but a boolean away: `caps` reported a single hardcoded tile.
This is the same data in the vocabulary Mantle can also name, so a kernel picks
its tile from what the device said rather than from a constant that happened to
be right on the card it was written on.

Entries whose component type Lava does not map are dropped rather than guessed.
"""
function matrixshapes(ctx::VkContext)
    out = MatrixShape[]
    ctx.coopmat_available || return out
    for s in ctx.coopmat_shapes
        ab = juliacomponenttype(s.ab_type)
        acc = juliacomponenttype(s.c_type)
        (ab === nothing || acc === nothing) && continue
        scope = s.scope == VK_SCOPE_SUBGROUP  ? SubgroupScope()  :
                s.scope == VK_SCOPE_WORKGROUP ? WorkgroupScope() : nothing
        scope === nothing && continue
        push!(out, MatrixShape(ab, acc, s.M, s.N, s.K, scope))
    end
    return out
end

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
    want = vkcomponenttype(T)
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
@inline CoopMatrix{T,M,N,U,S}(src, offset::Integer, stride::Integer) where {T,M,N,U,S} =
    coopmat_load(CoopMatrix{T,M,N,U,S}, src, offset, stride)

"""
    AcceleratedMatrix{T,M,N,U}(src, offset, stride, Val(true))   # row-major

The `MemoryLayout` operand, which was hardcoded to column-major.

A staged GEMM needs both from the *same* block: `mul_mm.comp` stages A and B
identically and then reads A `RowMajor` and B `ColumnMajor`, because A is
`(M, K)` and B is `(K, N)` and they share the k axis. With only one layout
available, one of the two has to be transposed while staging — an extra pass over
shared memory, or a second copy of the block.
"""
@inline CoopMatrix{T,M,N,U,S}(src, offset::Integer, stride::Integer,
                              rowmajor::Val) where {T,M,N,U,S} =
    coopmat_load(CoopMatrix{T,M,N,U,S}, src, offset, stride, rowmajor)

# The `@localmem` forms of these live in `array/ka_backend.jl`, beside
# `LavaSharedArray` — this file is included before that type exists.

"""
    copyto!(dst, offset, stride, m)

Store the tile back. Lowers to `OpCooperativeMatrixStoreKHR`.
"""
@inline Base.copyto!(dst, offset::Integer, stride::Integer, m::CoopMatrix) =
    coopmat_store(dst, offset, stride, m)

"""
    convert(AcceleratedMatrix{T,M,N,U}, m)

Change a tile's component type in registers. Lowers to `OpFConvert`.

The point is the GEMM's write-out: accumulation is fp32 and the destination is
fp16, and without this the only way across is a store to an fp32 scratch and a
second kernel that reads all of it back — `mm_epilogue_kernel!`, which is 23% of
matmul time and a whole extra pass over `M x N`.

Shape, scope and use must match; only the component type changes. (The wider
conversion that also changes the USE is `coopmat_convert`, which `convert` does
not reach: `Base.convert` between two types differing in a type parameter that
means "operand position" is not something to arrive at implicitly.)
"""
@inline Base.convert(::Type{CoopMatrix{T,M,N,U,SC}},
                     m::CoopMatrix{S,M,N,U,SC}) where {T,S,M,N,U,SC} =
    coopmat_convert(CoopMatrix{T,M,N,U,SC}, m)
@inline Base.convert(::Type{CoopMatrix{T,M,N,U,SC}},
                     m::CoopMatrix{T,M,N,U,SC}) where {T,M,N,U,SC} = m

@inline Base.zero(::Type{CoopMatrix{T,M,N,U,S}}) where {T,M,N,U,S} =
    coopmat_zero(CoopMatrix{T,M,N,U,S})
@inline Base.zero(::CoopMatrix{T,M,N,U,S}) where {T,M,N,U,S} =
    zero(CoopMatrix{T,M,N,U,S})

"""
    muladd(a, b, c)

`a*b + c`, accumulated in `c`'s element type. Lowers to
`OpCooperativeMatrixMulAddKHR` — this is the instruction that reaches the
tensor cores.
"""
@inline Base.muladd(a::CoopMatrix{TA,M,K,MatrixA,S},
                    b::CoopMatrix{TB,K,N,MatrixB,S},
                    c::CoopMatrix{TC,M,N,Accumulator,S}) where {TA,TB,TC,M,N,K,S} =
    coopmat_muladd(a, b, c)

@inline Base.:*(a::CoopMatrix{TA,M,K,MatrixA,S},
                b::CoopMatrix{TB,K,N,MatrixB,S}) where {TA,TB,M,N,K,S} =
    muladd(a, b, zero(CoopMatrix{promote_type(TA, TB),M,N,Accumulator,S}))
