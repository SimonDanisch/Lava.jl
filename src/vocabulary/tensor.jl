"""
Tensor addressing: a cooperative matrix loaded and stored through a DESCRIPTION of
the array in memory, rather than staged through workgroup memory first.

Part of the device vocabulary for the reason the cooperative-matrix one next door
is: every backend that has it names the same concepts. Vulkan reaches this through
`SPV_NV_tensor_addressing`; nothing else does yet, so every backend that does not
answers [`supports_tensor_addressing`](@ref) with `false` and implements none of the
operations below.

A layout is a value, like a matrix: each operation returns a new one rather than
changing its argument. `DIM` (the rank) and `CLAMP` (what a load outside the tensor
reads) are in the TYPE, because a backend builds the layout's type from them as
constants.
"""

"""
    TensorLayout{DIM,CLAMP}

How a `DIM`-dimensional array lies in memory: its extent and stride per dimension,
and what a load outside it reads (`CLAMP`, one of [`TENSOR_CLAMP_UNDEFINED`](@ref),
[`TENSOR_CLAMP_CONSTANT`](@ref), [`TENSOR_CLAMP_TO_EDGE`](@ref)).

Opaque, like [`CoopMatrix`](@ref): its last, hidden parameter is the backend's
representation.
"""
struct TensorLayout{DIM,CLAMP,Storage}
    handle::Storage
end

@inline TensorLayout{DIM,CLAMP}(x::R) where {DIM,CLAMP,R} = TensorLayout{DIM,CLAMP,R}(x)

"""
    TensorView{DIM,PERM}

A permutation of a layout's dimensions, applied at a load: `PERM = (1, 0)` reads a
2-D tensor transposed, in place.
"""
struct TensorView{DIM,PERM,Storage}
    handle::Storage
end

@inline TensorView{DIM,PERM}(x::R) where {DIM,PERM,R} = TensorView{DIM,PERM,R}(x)

"""A load outside the tensor reads an undefined value."""
const TENSOR_CLAMP_UNDEFINED = UInt32(0)
"""A load outside the tensor reads a constant, set by [`tensor_setclampvalue`](@ref).
What makes an extent that does not divide the tile legal without padding it."""
const TENSOR_CLAMP_CONSTANT = UInt32(1)
"""A load outside the tensor reads the nearest element inside it."""
const TENSOR_CLAMP_TO_EDGE = UInt32(2)

"""
    tensor_layout(Val(DIM), Val(CLAMP)) -> TensorLayout

A fresh layout. Rank and clamp mode are compile-time; the extents and strides are
set by [`tensor_setdim`](@ref) and [`tensor_setstride`](@ref).
"""
function tensor_layout end

"""    tensor_setdim(layout, dims::NTuple{DIM,Int32}) -> TensorLayout — the extent per dimension."""
function tensor_setdim end

"""    tensor_setstride(layout, strides::NTuple{DIM,Int32}) -> TensorLayout — the stride per dimension, in elements."""
function tensor_setstride end

"""
    tensor_setclampvalue(layout, bits::Int32) -> TensorLayout

The value a [`TENSOR_CLAMP_CONSTANT`](@ref) load reads outside the tensor, as the
bits [`tensor_clampbits`](@ref) gives for the matrix's component type.
"""
function tensor_setclampvalue end

"""
    tensor_clampbits(x::Real, ::Type{T}) -> Int32

The operand [`tensor_setclampvalue`](@ref) takes to fill with `x` for a matrix of
component type `T`: the bits of `x` in `T`, zero-extended. Measured on the device
rather than read off a signature — a fill of `1.0f0` arrives as `1.0` under this
reading and as `1.4e-45` under the numeric one (`mwe_tensor_clampvalue.jl` in Lava).
"""
tensor_clampbits(x::Real, ::Type{Float16}) = Int32(reinterpret(UInt16, Float16(x)))
tensor_clampbits(x::Real, ::Type{Float32}) = reinterpret(Int32, reinterpret(UInt32, Float32(x)))
tensor_clampbits(x::Real, ::Type{T}) where {T<:Integer} = Int32(x)

"""
    tensor_slice(layout, offsets::NTuple{DIM,Int32}, sizes::NTuple{DIM,Int32}) -> TensorLayout

The sub-block at `offsets` (zero-based) of extent `sizes`: what one workgroup or
one matrix of a larger product reads.
"""
function tensor_slice end

"""    tensor_view(Val(DIM), Val(PERM)) -> TensorView — a dimension permutation for a load."""
function tensor_view end

"""
    tensor_load(m::CoopMatrix, address::UInt64, layout[, view]) -> CoopMatrix

Fill a matrix of `m`'s type from the tensor at device address `address`, through
`layout` (and `view`). `m` is NOT only a destination: elements a clamping layout
leaves out keep `m`'s values, so pass a zeroed matrix when every element is in range.
"""
function tensor_load end

"""    tensor_store(m::CoopMatrix, address::UInt64, layout) — store `m` through `layout`."""
function tensor_store end

"""
    supports_tensor_addressing(backend)::Bool

Whether this device has the operations above. The fallback is `false`.
"""
supports_tensor_addressing(::KI.Backend) = false
