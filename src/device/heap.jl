# Device memory on a WebGPU target (`TargetFeatures(webgpu = true)`).
#
# WGSL has no pointers into storage memory. A WebGPU device keeps its memory in a few
# storage buffers, the heaps, each bound as `array<u32>`; a pointer is a heap and a
# byte offset into it. Everything an array does with memory goes through `HeapPtr`'s
# `unsafe_load`/`unsafe_store!`, which become word loads and stores of a heap
# (`_lava_wgpu_heapload`/`_lava_wgpu_heapstore`, heap and word). The emitter turns
# those into access chains into the array of heaps; the kernel's arguments arrive
# through `_lava_wgpu_argload` from a read-only binding of their own.

import UnsafeAtomics

@inline heapload(h::UInt32, w::UInt32) = Base.llvmcall(("""
    declare i32 @_lava_wgpu_heapload(i32, i32)
    define i32 @entry(i32 %h, i32 %w) #0 {
        %r = call i32 @_lava_wgpu_heapload(i32 %h, i32 %w)
        ret i32 %r
    }
    attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{UInt32, UInt32}, h, w)

@inline heapstore(h::UInt32, w::UInt32, v::UInt32) = Base.llvmcall(("""
    declare void @_lava_wgpu_heapstore(i32, i32, i32)
    define void @entry(i32 %h, i32 %w, i32 %v) #0 {
        call void @_lava_wgpu_heapstore(i32 %h, i32 %w, i32 %v)
        ret void
    }
    attributes #0 = { alwaysinline }
    """, "entry"), Nothing, Tuple{UInt32, UInt32, UInt32}, h, w, v)

append!(KNOWN_INTRINSICS, ["_lava_wgpu_heapload", "_lava_wgpu_heapstore", "_lava_wgpu_argload"])

@inline heapstorepart(h::UInt32, w::UInt32, keep::UInt32, bits::UInt32) = Base.llvmcall(("""
    declare void @_lava_wgpu_atomic_storepart(i32, i32, i32, i32)
    define void @entry(i32 %h, i32 %w, i32 %keep, i32 %bits) #0 {
        call void @_lava_wgpu_atomic_storepart(i32 %h, i32 %w, i32 %keep, i32 %bits)
        ret void
    }
    attributes #0 = { alwaysinline }
    """, "entry"), Nothing, NTuple{4,UInt32}, h, w, keep, bits)

push!(KNOWN_INTRINSICS, "_lava_wgpu_atomic_storepart")

# `_lava_wgpu_argload(word)`, word `word` of the kernel's argument block, is
# called only from the entry wrapper (`wrap_entry_for_webgpu!`), never from Julia.

"""
    HeapPtr{T}

A pointer on a WebGPU device: the byte `offset` of a `T` in heap `heap`. Its 8 bytes
are those of the 64-bit address `heap << 32 | offset`, so a host that patches
recorded pointers when memory moves handles it as a device address. Arithmetic is
in bytes, as for `Ptr`, and 32-bit: it never leaves the heap (a heap binding is
below 4 GiB), which keeps addressing in the integers WGSL has.
"""
struct HeapPtr{T}
    offset::UInt32
    heap::UInt32
end

Base.:+(p::HeapPtr{T}, n::Integer) where {T} = HeapPtr{T}(p.offset + (n % UInt32), p.heap)
Base.:-(p::HeapPtr{T}, n::Integer) where {T} = HeapPtr{T}(p.offset - (n % UInt32), p.heap)

"""The heap word holding byte `byte` of a heap."""
@inline heapword(byte::UInt32) = byte >> 2

@inline Base.unsafe_load(p::HeapPtr{T}, i::Integer = 1) where {T} =
    heapread(T, p.heap, p.offset + ((i - 1) * sizeof(T)) % UInt32)

@inline function Base.unsafe_store!(p::HeapPtr{T}, v, i::Integer = 1) where {T}
    heapwrite!(p.heap, p.offset + ((i - 1) * sizeof(T)) % UInt32, convert(T, v))
    return p
end

# A value moves between the heap and registers primitive by primitive: a struct or
# tuple is assembled from its fields, never reinterpreted, because `reinterpret`
# between a struct and words asks Base for the struct's padding, which allocates
# and does not compile for a device. Padding is neither read nor written.

"""Whether a primitive of `n` bytes has a heap access: whole words, or one or two bytes of one."""
heapprimitive(n::Int) = n in (1, 2, 4, 8)

function checkheapprimitive(@nospecialize(T))
    heapprimitive(sizeof(T)) ||
        error("a $(sizeof(T))-byte $T has no load or store on a WebGPU device: the heap holds 1, 2, 4 and 8-byte primitives")
    return nothing
end

"""
The expression building a `T` out of the heap words `ws`, the `T` starting at byte
`base` of the first: each primitive leaf cut out of the word(s) holding it.
"""
function wordsvalue(@nospecialize(T), base::Int, ws::Vector{Symbol})
    Base.issingletontype(T) && return T.instance
    if isprimitivetype(T)
        checkheapprimitive(T)
        w = ws[base ÷ 4 + 1]
        n = sizeof(T)
        n == 4 && return :(reinterpret($T, $w))
        n == 8 && return :(reinterpret($T, UInt64($w) | (UInt64($(ws[base ÷ 4 + 2])) << 32)))
        bits = :($w >> $(UInt32(8 * (base % 4))))
        return n == 1 ? :(reinterpret($T, $bits % UInt8)) : :(reinterpret($T, $bits % UInt16))
    end
    fields = Any[wordsvalue(fieldtype(T, k), base + Int(fieldoffset(T, k)), ws) for k in 1:fieldcount(T)]
    return T <: Tuple ? :(tuple($(fields...))) : Expr(:new, T, fields...)
end

"""The primitive leaves of `T`: (byte offset, type, the expression reading it from `v`)."""
function heapleaves(@nospecialize(T), base::Int = 0, value = :v)
    Base.issingletontype(T) && return Any[]
    isprimitivetype(T) && (checkheapprimitive(T); return Any[(base, T, value)])
    out = Any[]
    for k in 1:fieldcount(T)
        append!(out, heapleaves(fieldtype(T, k), base + Int(fieldoffset(T, k)), :(getfield($value, $k))))
    end
    return out
end

"""The expression for heap word `k` (zero-based) of a `T` in `v`: its leaves shifted into place, padding zero."""
function wordexpr(@nospecialize(T), k::Int)
    parts = Any[]
    for (o, F, x) in heapleaves(T)
        n = sizeof(F)
        if n == 8
            o == 4k && push!(parts, :(reinterpret(UInt64, $x) % UInt32))
            o == 4k - 4 && push!(parts, :((reinterpret(UInt64, $x) >> 32) % UInt32))
        elseif o ÷ 4 == k
            bits = n == 4 ? :(reinterpret(UInt32, $x)) :
                   n == 2 ? :(UInt32(reinterpret(UInt16, $x))) : :(UInt32(reinterpret(UInt8, $x)))
            push!(parts, o % 4 == 0 ? bits : :($bits << $(UInt32(8 * (o % 4)))))
        end
    end
    isempty(parts) && return :(UInt32(0))
    return foldl((a, b) -> :($a | $b), parts)
end

"""
    heapread(T, h, byte) -> T

The `T` at byte `byte` of heap `h`. A `T` of whole words that starts on a word is
read in those words; anything else field by field, a one- or two-byte primitive out
of the word holding it (natural alignment keeps it inside one).
"""
@generated function heapread(::Type{T}, h::UInt32, byte::UInt32) where {T}
    Base.issingletontype(T) && return :($(T.instance))
    if isprimitivetype(T)
        checkheapprimitive(T)
        sizeof(T) >= 4 || return :(heapreadpart(T, h, byte))
        ws = [Symbol(:w, k) for k in 1:sizeof(T) ÷ 4]
        loads = [:($(ws[k]) = heapload(h, w0 + $(UInt32(k - 1)))) for k in eachindex(ws)]
        return quote
            w0 = heapword(byte)
            $(loads...)
            $(wordsvalue(T, 0, ws))
        end
    end
    fields = Any[:(heapread($(fieldtype(T, k)), h, byte + $(UInt32(fieldoffset(T, k))))) for k in 1:fieldcount(T)]
    perfield = T <: Tuple ? :(tuple($(fields...))) : Expr(:new, T, fields...)
    sizeof(T) % 4 == 0 || return perfield
    ws = [Symbol(:w, k) for k in 1:sizeof(T) ÷ 4]
    loads = [:($(ws[k]) = heapload(h, w0 + $(UInt32(k - 1)))) for k in eachindex(ws)]
    words = quote
        w0 = heapword(byte)
        $(loads...)
        $(wordsvalue(T, 0, ws))
    end
    Base.datatype_alignment(T) >= 4 && return words
    return :(byte & 0x3 == UInt32(0) ? $words : $perfield)
end

"""
Word `w` of heap `h` shifted right by `s` bits, in one `llvmcall`: the shift
comes from the address, and done outside it would make the value read look like
an address to Mantle's access analysis (an integer computed from one is one).
"""
@inline heapload_shifted(h::UInt32, w::UInt32, s::UInt32) = Base.llvmcall(("""
    declare i32 @_lava_wgpu_heapload(i32, i32)
    define i32 @entry(i32 %h, i32 %w, i32 %s) #0 {
        %r = call i32 @_lava_wgpu_heapload(i32 %h, i32 %w)
        %v = lshr i32 %r, %s
        ret i32 %v
    }
    attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{UInt32, UInt32, UInt32}, h, w, s)

@inline function heapreadpart(::Type{T}, h::UInt32, byte::UInt32) where {T}
    bits = heapload_shifted(h, heapword(byte), UInt32(8) * (byte & 0x3))
    return sizeof(T) == 1 ? reinterpret(T, bits % UInt8) : reinterpret(T, bits % UInt16)
end

"""
    heapwrite!(h, byte, v)

Store `v` at byte `byte` of heap `h`. A `v` of whole words that starts on a word is
written as those words, its padding zero; anything else field by field, a one- or
two-byte primitive with [`storepart`](@ref), since the rest of its word may be
another invocation's.
"""
@generated function heapwrite!(h::UInt32, byte::UInt32, v::T) where {T}
    Base.issingletontype(T) && return :(nothing)
    if isprimitivetype(T)
        checkheapprimitive(T)
        sizeof(T) >= 4 || return :(storepart(T, h, heapword(byte), byte & 0x3, v); nothing)
    end
    perfield = isprimitivetype(T) ? :(nothing) :
        Expr(:block, [:(heapwrite!(h, byte + $(UInt32(fieldoffset(T, k))), getfield(v, $k)))
                      for k in 1:fieldcount(T)]...)
    sizeof(T) % 4 == 0 || return :($perfield; nothing)
    stores = [:(heapstore(h, w0 + $(UInt32(k)), $(wordexpr(T, k)))) for k in 0:sizeof(T) ÷ 4 - 1]
    words = quote
        w0 = heapword(byte)
        $(stores...)
    end
    (isprimitivetype(T) || Base.datatype_alignment(T) >= 4) && return :($words; nothing)
    return :(byte & 0x3 == UInt32(0) ? $words : $perfield; nothing)
end

"""
Store a one- or two-byte `v` at byte `at` of word `w` of heap `h`. The other bytes
of the word belong to other elements, which other invocations may be writing, so it
is two atomics: clear the bytes, then set them. Between the two only this
invocation touches them. Atomics declare the kernel's heaps atomic (see below).
"""
@inline function storepart(::Type{T}, h::UInt32, w::UInt32, at::UInt32, v::T) where {T}
    shift = 8 * at
    mask = (sizeof(T) == 1 ? 0x000000ff : 0x0000ffff) << shift
    bits = (sizeof(T) == 1 ? UInt32(reinterpret(UInt8, v)) : UInt32(reinterpret(UInt16, v))) << shift
    heapstorepart(h, w, ~mask, bits)
    return nothing
end

# ── Atomics ──────────────────────────────────────────────────────────────────
#
# WGSL atomics are 32-bit and act on `atomic<u32>` storage only: a kernel that uses
# them has its heaps declared atomic, and all its heap accesses are atomic
# (`compiler/spirv/webgpu.jl` says why there is no separate view). Each intrinsic
# takes the heap and its word and returns the old value.

for op in (:add, :sub, :and, :or, :xor, :xchg, :umin, :umax)
    name = "_lava_wgpu_atomic_$(op)"
    ir = """
        declare i32 @$(name)(i32, i32, i32)
        define i32 @entry(i32 %h, i32 %w, i32 %v) #0 {
            %r = call i32 @$(name)(i32 %h, i32 %w, i32 %v)
            ret i32 %r
        }
        attributes #0 = { alwaysinline }
        """
    @eval @inline $(Symbol(:heapatomic_, op))(h::UInt32, w::UInt32, v::UInt32) =
        Base.llvmcall(($ir, "entry"), UInt32, Tuple{UInt32, UInt32, UInt32}, h, w, v)
    push!(KNOWN_INTRINSICS, name)
end

@inline heapatomic_load(h::UInt32, w::UInt32) = Base.llvmcall(("""
    declare i32 @_lava_wgpu_atomic_load(i32, i32)
    define i32 @entry(i32 %h, i32 %w) #0 {
        %r = call i32 @_lava_wgpu_atomic_load(i32 %h, i32 %w)
        ret i32 %r
    }
    attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{UInt32, UInt32}, h, w)

@inline heapatomic_store(h::UInt32, w::UInt32, v::UInt32) = Base.llvmcall(("""
    declare void @_lava_wgpu_atomic_store(i32, i32, i32)
    define void @entry(i32 %h, i32 %w, i32 %v) #0 {
        call void @_lava_wgpu_atomic_store(i32 %h, i32 %w, i32 %v)
        ret void
    }
    attributes #0 = { alwaysinline }
    """, "entry"), Nothing, Tuple{UInt32, UInt32, UInt32}, h, w, v)

"""Strong compare-and-swap of word `w` of heap `h`: the value it held; it now holds `new` if that was `cmp`."""
@inline heapatomic_cas(h::UInt32, w::UInt32, cmp::UInt32, new::UInt32) = Base.llvmcall(("""
    declare i32 @_lava_wgpu_atomic_cas(i32, i32, i32, i32)
    define i32 @entry(i32 %h, i32 %w, i32 %c, i32 %n) #0 {
        %r = call i32 @_lava_wgpu_atomic_cas(i32 %h, i32 %w, i32 %c, i32 %n)
        ret i32 %r
    }
    attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{UInt32, UInt32, UInt32, UInt32}, h, w, cmp, new)

append!(KNOWN_INTRINSICS, ["_lava_wgpu_atomic_load", "_lava_wgpu_atomic_store", "_lava_wgpu_atomic_cas"])

const Word32 = Union{Int32, UInt32}

"""The WGSL atomic of `op` on a `T`, or `nothing` where it takes a compare-and-swap loop."""
heapatomicop(::typeof(+), ::Type{<:Word32}) = heapatomic_add
heapatomicop(::typeof(-), ::Type{<:Word32}) = heapatomic_sub
heapatomicop(::typeof(&), ::Type{<:Word32}) = heapatomic_and
heapatomicop(::typeof(|), ::Type{<:Word32}) = heapatomic_or
heapatomicop(::typeof(xor), ::Type{<:Word32}) = heapatomic_xor
heapatomicop(::typeof(UnsafeAtomics.right), ::Type{T}) where {T} = sizeof(T) == 4 ? heapatomic_xchg : nothing
heapatomicop(::typeof(min), ::Type{UInt32}) = heapatomic_umin
heapatomicop(::typeof(max), ::Type{UInt32}) = heapatomic_umax
heapatomicop(op, ::Type) = nothing

"""Only 32-bit values have WGSL atomics."""
@generated function atomicword(::Type{T}) where {T}
    sizeof(T) == 4 || error("a $(sizeof(T))-byte $T has no atomics on a WebGPU device: WGSL's are 32-bit")
    return :(nothing)
end

@inline function UnsafeAtomics.modify!(p::HeapPtr{T}, op::OP, x, order, scope...) where {T, OP}
    atomicword(T)
    h, w = p.heap, heapword(p.offset)
    v = convert(T, x)
    native = heapatomicop(op, T)
    if native !== nothing
        old = reinterpret(T, native(h, w, reinterpret(UInt32, v)))
        return old => op(old, v)
    end
    old = reinterpret(T, heapatomic_load(h, w))
    while true
        new = op(old, v)
        seen = heapatomic_cas(h, w, reinterpret(UInt32, old), reinterpret(UInt32, new))
        seen == reinterpret(UInt32, old) && return old => new
        old = reinterpret(T, seen)
    end
end

@inline function UnsafeAtomics.load(p::HeapPtr{T}, order, scope...) where {T}
    atomicword(T)
    return reinterpret(T, heapatomic_load(p.heap, heapword(p.offset)))
end

@inline function UnsafeAtomics.store!(p::HeapPtr{T}, x, order, scope...) where {T}
    atomicword(T)
    heapatomic_store(p.heap, heapword(p.offset), reinterpret(UInt32, convert(T, x)))
    return nothing
end

@inline function UnsafeAtomics.cas!(p::HeapPtr{T}, cmp, new, success, failure, scope...) where {T}
    atomicword(T)
    c = reinterpret(UInt32, convert(T, cmp))
    old = heapatomic_cas(p.heap, heapword(p.offset), c, reinterpret(UInt32, convert(T, new)))
    return (; old = reinterpret(T, old), success = old == c)
end

"""
    HeapDeviceArray{T,N}

The kernel's view of an array on a WebGPU device: where its first element is in a
heap, and its dimensions. Indexed like `LavaDeviceArray`.
"""
struct HeapDeviceArray{T,N} <: GPUArrays.AbstractDeviceArray{T,N}
    ptr::HeapPtr{T}
    dims::NTuple{N,Int}
end

Base.size(a::HeapDeviceArray) = a.dims
Base.pointer(a::HeapDeviceArray{T}) where {T} = a.ptr
Base.pointer(a::HeapDeviceArray{T}, i::Integer) where {T} = a.ptr + (i - 1) * sizeof(T)

function Base.reinterpret(::Type{T}, a::HeapDeviceArray{S,N}) where {T,S,N}
    err = GPUArrays._reinterpret_exception(T, a)
    err === nothing || throw(err)
    d = a.dims
    n1 = div(d[1] * sizeof(S), sizeof(T))
    return HeapDeviceArray{T,N}(HeapPtr{T}(a.ptr.offset, a.ptr.heap), (n1, Base.tail(d)...))
end
