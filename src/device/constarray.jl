# A kernel argument nothing writes while the kernel runs.
#
# Its loads carry `!invariant.load`, which the emitter turns into a read through a
# `NonWritable` struct member (`nonwritable_pointer!`). That is the only way SPIR-V
# can say that memory behind a PSB pointer is read-only, and drivers act on it: RADV
# only reads a uniform address through the scalar cache when it knows the memory is
# not written, which made an all-pairs Hamming kernel 25% faster on a Radeon 8060S.
# CUDA.jl's `Const` is the same wrapper; there NVPTX turns the metadata into
# `ld.global.nc`.
#
# The indexing mirrors `LavaDeviceArray`'s method for method (devicearray.jl), so a
# multi-index read goes through the same Horner `linear_index`; only the load differs.

"""
    Const(a::LavaDeviceArray)

`a`, promised not to be written by anything while the kernel runs: neither by the
kernel nor through another argument that reaches the same memory. Reads compile to
loads a driver may serve from a read-only cache. Writing through it is a
`MethodError`.
"""
struct Const{T,N} <: DenseArray{T,N}
    a::LavaDeviceArray{T,N}
end

Base.Experimental.Const(a::LavaDeviceArray) = Const(a)

Base.size(c::Const) = size(c.a)
Base.IndexStyle(::Type{<:Const}) = IndexLinear()
Base.pointer(c::Const) = pointer(c.a)
Base.pointer(c::Const, i::Integer) = pointer(c.a, i)
Base.unsafe_convert(::Type{Ptr{T}}, c::Const{T}) where {T} = c.a.ptr
Base.reinterpret(::Type{T}, c::Const) where {T} = Const(reinterpret(T, c.a))

# Element types loaded with `!invariant.load`. Anything else loads as the array does,
# the way CUDA.jl's `unsafe_cached_load` falls back: a composite would have to be
# loaded field by field to carry the metadata.
const InvariantLoadable = Union{Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                                Float16, Float32, Float64}

"""
    load_invariant(ptr::Ptr{T}, i::Int) -> T

Element `i` (1-based) at `ptr`, loaded with `!invariant.load` at `T`'s natural
alignment, which is the alignment every PSB access is emitted with
(`psb_access_alignment`).
"""
@inline load_invariant(ptr::Ptr{T}, i::Int) where {T <: InvariantLoadable} =
    invariant_pointerref(ptr, i - 1, Val(Base.datatype_alignment(T)))
@inline load_invariant(ptr::Ptr, i::Int) = unsafe_load(ptr, i)

# `LLVM.Interop._pointerref` with the metadata. On Julia 1.12 a `Ptr` reaches the body
# as an opaque `ptr`, so no cast is needed.
LLVM.Interop.@llvmgenerated builder function invariant_pointerref(ptr::Ptr{T}, i::Int,
                                                                  ::Val{align})::T where {T,align}
    eltyp = convert(LLVM.LLVMType, T)
    ld = LLVM.load!(builder, eltyp, LLVM.inbounds_gep!(builder, eltyp, ptr, [i]); align)
    ld.metadata[LLVM.MD_invariant_load] = LLVM.MDNode(LLVM.Metadata[])
    ld
end

@lava_device_override @inline function Base.getindex(c::Const{T,0}) where T
    load_invariant(c.a.ptr, 1)
end

@lava_device_override @inline function Base.getindex(c::Const{T,0}, I::CartesianIndex) where T
    load_invariant(c.a.ptr, 1)
end

@lava_device_override @inline function Base.getindex(c::Const{T}, i::Integer) where T
    load_invariant(c.a.ptr, Int(i))
end

@lava_device_override @inline function Base.getindex(c::Const{T,N}, I::CartesianIndex{N}) where {T,N}
    load_invariant(c.a.ptr, linear_index(c.a.dims, I))
end

@lava_device_override @inline function Base.getindex(c::Const{T,N}, i1::Integer, i2::Integer,
                                                     Ir::Vararg{Integer}) where {T,N}
    load_invariant(c.a.ptr, linear_index(c.a.dims, CartesianIndex(i1, i2, Ir...)))
end
