"""
Cooperative matrices: the type a kernel names to reach matrix hardware.

Part of the device vocabulary for the reason `MatrixUse` and `MatrixScope` next
door are: every backend names the same concepts, and the TYPE is the same case.
Vulkan reaches this through `OpCooperativeMatrixMulAddKHR`, Metal through
`simdgroup_matrix`, CUDA through `wmma`; the tile a kernel is written against is
the same object on all of them.

The type is opaque.  Its last, deliberately hidden parameter is the backend's
native representation: Lava uses one `Int32` SSA anchor, while a native GPU
compiler can keep the register tuple its matrix intrinsic returns.  Kernels
name only the first five parameters, so that representation never leaks into
portable code.  What IS a backend's is the `coopmat_*` operations below,
declared here and lowered by whichever compiler is in play.
"""

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
struct CoopMatrix{T,M,N,Use<:MatrixUse,Scope<:MatrixScope,Storage}
    # Opaque backend value.  `handle` keeps the source-level API used by the
    # SPIR-V lowering; on native backends it may be a register fragment.
    handle::Storage
end

# Hide `Storage` from kernel code.  A partial parametric type does not receive a
# default outer constructor automatically, so spell out the one public
# construction point used by every backend lowering.
@inline CoopMatrix{T,M,N,U,S}(x::R) where {T,M,N,U,S,R} =
    CoopMatrix{T,M,N,U,S,R}(x)

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

"""    matrixuse(m) -> Type{<:MatrixUse}

The operand position of a cooperative matrix or its type: `MatrixA`, `MatrixB` or `Accumulator`."""
matrixuse(::Type{<:CoopMatrix{T,M,N,U}}) where {T,M,N,U} = U
matrixuse(m::CoopMatrix) = matrixuse(typeof(m))

"""    matrixscope(m) -> Type{<:MatrixScope}

The scope of a cooperative matrix or its type: `SubgroupScope` or `WorkgroupScope`."""
matrixscope(::Type{<:CoopMatrix{T,M,N,U,S}}) where {T,M,N,U,S} = S
matrixscope(m::CoopMatrix) = matrixscope(typeof(m))

# `coopmat_shape`, `matrixshapes` and the `VkComponentTypeKHR` mappings are in
# `runtime/matrixshapes.jl`. They read `ctx.coopmat_shapes` — a driver property
# table — so they are device queries rather than device code, and this file is
# the type a kernel is compiled against.

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
    copyto!(dst, offset, stride, m, Val(true))   # row-major

The store's layout operand, the mirror of the load's. Lava declared it for its
shared arrays only, so a kernel storing a tile row-major (DNNKernels'
`attn_flash_rows!`) had no method on any other backend.
"""
@inline Base.copyto!(dst, offset::Integer, stride::Integer, m::CoopMatrix, rowmajor::Val) =
    coopmat_store(dst, offset, stride, m, rowmajor)

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


# ── The operations a backend lowers ───────────────────────────────────────────
#
# Declared, never defined here. Each is one call site the backend's emitter
# recognises and replaces with a matrix instruction; outside a kernel they have
# no meaning, exactly like the subgroup intrinsics.
#
# The `Base` methods above are written against these, which is what makes the
# type portable: a backend implements eleven functions and inherits `zero`, `*`,
# `muladd`, `convert` and `copyto!` unchanged.

"""
    coopmat_load(::Type{CoopMatrix{T,M,N,U,S}}, src, offset, stride, [rowmajor::Val])

Load an `M×N` tile. `rowmajor` selects the memory layout; a staged GEMM needs
both from the same block, since `A` is `(M, K)` and `B` is `(K, N)` and they
share the k axis.
"""
function coopmat_load end

"""    coopmat_store(dst, offset, stride, m)  —  store a tile back to memory."""
function coopmat_store end

"""    coopmat_muladd(a, b, c) -> `a*b + c`, accumulated in `c`'s element type."""
function coopmat_muladd end

"""    coopmat_mul(a, b) -> component-wise product of two tiles."""
function coopmat_mul end

"""    coopmat_add(a, b) -> component-wise sum of two tiles."""
function coopmat_add end

"""    coopmat_zero(::Type{CoopMatrix{…}}) -> an all-zero tile."""
function coopmat_zero end

"""    coopmat_undef(::Type{CoopMatrix{…}}) -> an uninitialised tile."""
function coopmat_undef end

"""    coopmat_convert(::Type{CoopMatrix{…}}, m) -> change component type, or use, in registers."""
function coopmat_convert end

"""
    coopmat_length(m) -> Int

How many components of `m` THIS invocation holds.

Not `M*N`: the value is spread across the lanes with an implementation-defined
layout, and the per-invocation count bears no fixed relation to the shape. It is
the bound for the per-element loops `coopmat_getcomp`/`coopmat_setcomp` drive.
"""
function coopmat_length end

"""    coopmat_getcomp(m, i) -> this invocation's `i`-th component."""
function coopmat_getcomp end

"""    coopmat_setcomp(m, i, v) -> `m` with this invocation's `i`-th component set."""
function coopmat_setcomp end

# ── Beyond load, store and multiply-add ──────────────────────────────────────
#
# Vendor extensions, not portable even across one API (see `DeviceCaps`): a kernel
# that wants one asks its predicate and carries the answer. Vulkan reaches both
# through `VK_NV_cooperative_matrix2`.

"""
    coopmat_perelement(f, m, extras...) -> CoopMatrix

`m` with every element replaced by `f(row, col, element, extras...)`, `row` and
`col` zero-based. `f` must be a top-level function, not a closure: anything it would
capture goes in `extras`. Ask [`supports_coopmat_perelement`](@ref) first.
"""
function coopmat_perelement end

"""
    coopmat_reduce(f, Dst::Type{<:CoopMatrix}, m, Val(mask)) -> Dst

Combine `m` along `mask` (see [`CoopMatReduce`](@ref)) with the binary function `f`
(`max` for a row maximum), into a matrix of type `Dst` that holds the reduced value
repeated along the reduced axis — not a vector. `f` must be a top-level function.
Ask [`supports_coopmat_reduce`](@ref) first.
"""
function coopmat_reduce end

"""
The axes [`coopmat_reduce`](@ref Lava.coopmat_reduce) combines along. `Row` and `Column` are bits, so
`RowAndColumn` is their union; `TwoByTwo` is the remaining bit and is unverified.
"""
module CoopMatReduce
const Row          = UInt32(1)
const Column       = UInt32(2)
const RowAndColumn = UInt32(3)
const TwoByTwo     = UInt32(4)
end

"""    supports_coopmat_perelement(backend)::Bool — [`coopmat_perelement`](@ref) is available. The fallback is `false`."""
supports_coopmat_perelement(::KI.Backend) = false

"""    supports_coopmat_reduce(backend)::Bool — [`coopmat_reduce`](@ref) is available. The fallback is `false`."""
supports_coopmat_reduce(::KI.Backend) = false

"""
    supports_flexible_coopmat_shapes(backend)::Bool

Whether a subgroup-scope matrix may have a shape the device's table
([`matrix_shapes`](@ref)) does not list, at the table's granularity — a 64x16
operand on a device that reports only M = 16. The fallback is `false`.
"""
supports_flexible_coopmat_shapes(::KI.Backend) = false
