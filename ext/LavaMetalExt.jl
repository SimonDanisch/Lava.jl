"""
Lava's device vocabulary on Metal.jl, as overlays in `Metal.method_table`: the
cooperative matrices on Apple's 8x8 `simdgroup_matrix`, and Metal's clip space. Kernels
and render stages that Metal.jl compiles call them wherever Lava's vocabulary is used.
"""
module LavaMetalExt

using Lava
using Lava: CoopMatrix, SubgroupScope, MatrixA, MatrixB, Accumulator
using Metal
# Metal's `bfloat`.
using BFloat16s: BFloat16

# The cooperative matrices of the device vocabulary (`Lava.CoopMatrix`), on Apple's 8x8
# `simdgroup_matrix`.
#
# A tile is the `<64 x T>` value every `air.simdgroup_matrix_8x8_*` intrinsic takes and
# returns, kept as the matrix's opaque storage. What one invocation holds of it was read
# off the hardware rather than assumed: an MSL kernel writing out every lane's
# `thread_elements()` showed that each of the 32 lanes OWNS elements 0 and 1 of that
# vector (the other 62 read zero), and writing those two writes the lane's share of the
# tile. So `coopmat_length` is 2, and the component-wise operations work on those two
# elements: the vector as a whole is not this invocation's.
#
# Loads and stores are Metal.jl's `simdgroup_load`/`simdgroup_store` on an array of
# `stride` rows and eight columns that starts at the tile. Their operands are what Metal's
# own compiler emits for `simdgroup_load(m, src, E, origin, transpose)`, read from an MSL
# dynamic library compiled at run time, serialized, and parsed back: a ROW-major tile is
# `dims = (E, 8)`, `strides = (1, E)` (`Val(false)` there); a COLUMN-major one — Julia's,
# and KernelInterface's default — is the transposed load, `dims = (8, E)`,
# `strides = (E, 1)` (`Val(true)`, their default). `E = 0` takes the same formula, which is
# the stride-0 broadcast a per-row factor matrix is built with.
#
# Every operation is an overlay in `Metal.method_table`. A plain method on
# `CoopMatrix{T,8,8,…}` would be more specific than Lava's, which are plain methods on
# any shape, and a Mac compiles both backends: Lava kernels would then lower to these
# intrinsics.

"""A cooperative matrix on this backend: one 8x8 `simdgroup_matrix`."""
const SimdgroupCoop{T,U} = CoopMatrix{T,8,8,U,SubgroupScope}

const TILE_TYPES = ((:Float16, "f16", "half"), (:Float32, "f32", "float"),
                    (:BFloat16, "bf16", "bfloat"))

for (T, _, ty) in TILE_TYPES
    @eval begin
        # This invocation's components, by index into the tile vector.
        @inline tile_get(t::NTuple{64,VecElement{$T}}, i::Int32) = Base.llvmcall($"""
            %r = extractelement <64 x $ty> %0, i32 %1
            ret $ty %r""", $T, Tuple{NTuple{64,VecElement{$T}},Int32}, t, i)
        @inline tile_set(t::NTuple{64,VecElement{$T}}, i::Int32, v::$T) = Base.llvmcall($"""
            %r = insertelement <64 x $ty> %0, $ty %2, i32 %1
            ret <64 x $ty> %r""", NTuple{64,VecElement{$T}},
            Tuple{NTuple{64,VecElement{$T}},Int32,$T}, t, i, v)
    end
end

# Any mix of operand types: the intrinsic is named for result, A, B and C in that
# order, and Apple's own libraries use mixed ones (`v64f32.v64f16.v64f16.v64f32`).
# Metal.jl's `simdgroup_multiply_accumulate` takes one type for all four.
for (TC, sc, _) in TILE_TYPES, (TA, sa, _) in TILE_TYPES, (TB, sb, _) in TILE_TYPES
    @eval Metal.@device_function tile_muladd(a::NTuple{64,VecElement{$TA}}, b::NTuple{64,VecElement{$TB}},
                                       c::NTuple{64,VecElement{$TC}}) =
        ccall($"extern air.simdgroup_matrix_8x8_multiply_accumulate.v64$sc.v64$sa.v64$sb.v64$sc",
              llvmcall, NTuple{64,VecElement{$TC}},
              (NTuple{64,VecElement{$TA}}, NTuple{64,VecElement{$TB}}, NTuple{64,VecElement{$TC}}),
              a, b, c)
end

# The array a tile `stride` elements apart is loaded from or stored to, and the
# `transpose` flag of `simdgroup_load`/`simdgroup_store` for either layout.
@inline tilearray(p::Core.LLVMPtr, stride::Integer) = Metal.MtlDeviceArray((Int(stride), 8), p)
@inline tiletranspose(::Val{ROWMAJOR}) where {ROWMAJOR} = Val(!ROWMAJOR)

# A tile with every element `zero(T)`. Filled rather than left undefined: Metal.jl's
# simdgroup code starts every tile from `init_filled`, the only constructor MSL has, and
# an `undef` vector has no matrix provenance for the compiler to lay out.
@inline zerotile(::Type{T}) where {T} = Metal.MtlSimdgroupMatrix{T,8,8}(zero(T)).data

Metal.@device_override @inline Lava.coopmat_load(
        ::Type{CoopMatrix{T,8,8,U,SubgroupScope}}, src::Core.LLVMPtr{T}, offset::Integer,
        stride::Integer, rowmajor::Val = Val(false)) where {T,U} =
    SimdgroupCoop{T,U}(Metal.simdgroup_load(tilearray(src + (offset - 1) * sizeof(T), stride),
                                            (1, 1), tiletranspose(rowmajor)))
Metal.@device_override @inline Lava.coopmat_load(
        ::Type{CoopMatrix{T,8,8,U,SubgroupScope}}, src::Metal.MtlDeviceArray{T}, offset::Integer,
        stride::Integer, rowmajor::Val = Val(false)) where {T,U} =
    Lava.coopmat_load(SimdgroupCoop{T,U}, pointer(src), offset, stride, rowmajor)

Metal.@device_override @inline function Lava.coopmat_store(
        dst::Core.LLVMPtr{T}, offset::Integer, stride::Integer,
        m::CoopMatrix{T,8,8,<:Any,SubgroupScope}, rowmajor::Val = Val(false)) where {T}
    Metal.simdgroup_store(m.handle, tilearray(dst + (offset - 1) * sizeof(T), stride),
                          (1, 1), tiletranspose(rowmajor))
    return nothing
end
Metal.@device_override @inline Lava.coopmat_store(
        dst::Metal.MtlDeviceArray{T}, offset::Integer, stride::Integer,
        m::CoopMatrix{T,8,8,<:Any,SubgroupScope}, rowmajor::Val = Val(false)) where {T} =
    Lava.coopmat_store(pointer(dst), offset, stride, m, rowmajor)

Metal.@device_override @inline Lava.coopmat_muladd(
        a::CoopMatrix{<:Any,8,8,MatrixA,SubgroupScope},
        b::CoopMatrix{<:Any,8,8,MatrixB,SubgroupScope},
        c::CoopMatrix{TC,8,8,Accumulator,SubgroupScope}) where {TC} =
    SimdgroupCoop{TC,Accumulator}(tile_muladd(a.handle, b.handle, c.handle))

Metal.@device_override @inline Lava.coopmat_zero(
        ::Type{CoopMatrix{T,8,8,U,SubgroupScope}}) where {T,U} =
    SimdgroupCoop{T,U}(zerotile(T))

# A filled tile, for the reason at `zerotile`.
Metal.@device_override @inline Lava.coopmat_undef(
        ::Type{CoopMatrix{T,8,8,U,SubgroupScope}}) where {T,U} =
    SimdgroupCoop{T,U}(zerotile(T))

Metal.@device_override @inline Lava.coopmat_length(
        ::Type{CoopMatrix{T,8,8,U,SubgroupScope}}) where {T,U} = Int32(2)

Metal.@device_override @inline Lava.coopmat_getcomp(
        m::CoopMatrix{T,8,8,<:Any,SubgroupScope}, i::Int32) where {T} =
    tile_get(m.handle, i)

Metal.@device_override @inline Lava.coopmat_setcomp(
        m::CoopMatrix{T,8,8,U,SubgroupScope}, i::Int32, v::T) where {T,U} =
    SimdgroupCoop{T,U}(tile_set(m.handle, i, v))

# Every use is laid out the same on this hardware — there is one matrix type, and the
# use is the operand position it is passed in — so a change of use moves nothing, and
# a change of element type converts this invocation's two components.
Metal.@device_override @inline function Lava.coopmat_convert(
        ::Type{CoopMatrix{T,8,8,U,SubgroupScope}},
        m::CoopMatrix{S,8,8,<:Any,SubgroupScope}) where {T,U,S}
    t = zerotile(T)
    t = tile_set(t, Int32(0), convert(T, tile_get(m.handle, Int32(0))))
    t = tile_set(t, Int32(1), convert(T, tile_get(m.handle, Int32(1))))
    return SimdgroupCoop{T,U}(t)
end

@inline function tile_map(f, a::CoopMatrix{T,8,8,U,SubgroupScope},
                          b::CoopMatrix{T,8,8,U,SubgroupScope}) where {T,U}
    t = tile_set(a.handle, Int32(0), f(tile_get(a.handle, Int32(0)), tile_get(b.handle, Int32(0))))
    t = tile_set(t, Int32(1), f(tile_get(a.handle, Int32(1)), tile_get(b.handle, Int32(1))))
    return SimdgroupCoop{T,U}(t)
end

Metal.@device_override @inline Lava.coopmat_add(
        a::CoopMatrix{T,8,8,U,SubgroupScope}, b::CoopMatrix{T,8,8,U,SubgroupScope}) where {T,U} =
    tile_map(+, a, b)

Metal.@device_override @inline Lava.coopmat_mul(
        a::CoopMatrix{T,8,8,U,SubgroupScope}, b::CoopMatrix{T,8,8,U,SubgroupScope}) where {T,U} =
    tile_map(*, a, b)

# Metal's clip-space +y is up, Vulkan's (the portable one) down. An overlay, not a method
# redefinition, so the host answer is unchanged; it applies to compute too, so compute
# reprojection matches the rasteriser. A plain `@overlay`, not `@device_override` (a
# `@consistent_overlay`): Julia would constant-fold calls with the host method, which
# returns `y` unchanged.
Base.Experimental.@overlay Metal.method_table Lava.clip_y(y::Float32) = -y

end
