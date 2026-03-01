# Atomix.jl integration for Lava.jl
#
# Provides atomic operations on LavaArray elements in GPU kernels.
# Uses LLVM atomicrmw with syncscope("device") — required for Vulkan SPIR-V.
# The default CrossDevice scope is invalid in Vulkan.
#
# On the GPU, LavaArray becomes Ptr{T} via LavaAdaptor. We need to make
# Atomix.@atomic work with raw Ptr{T}. The Atomix macro expands to:
#   Atomix.modify!(referenceable(arr)[i], +, v, seq_cst)[2]
# Since Ptr{T} isn't an AbstractArray, we provide:
# 1. A PtrRef wrapper as the "referenceable" for Ptr{T}
# 2. A PtrIndexableRef as the indexed reference
# 3. Atomix.modify! overrides that call our atomicrmw llvmcall

import Atomix
import UnsafeAtomics

# ── Ptr-based Atomix references ──

struct PtrRef{T}
    ptr::Ptr{T}
end

struct PtrIndexableRef{T}
    ptr::Ptr{T}
    index::Int
end

@lava_device_override function Atomix.Internal.referenceable(ptr::Ptr{T}) where T
    PtrRef{T}(ptr)
end

# LavaDeviceArray atomics: delegate to the underlying Ptr
@lava_device_override function Atomix.Internal.referenceable(a::LavaDeviceArray{T}) where T
    PtrRef{T}(a.ptr)
end

@lava_device_override function Base.getindex(r::PtrRef{T}, i::Integer) where T
    PtrIndexableRef{T}(r.ptr, Int(i))
end

@lava_device_override function Atomix.pointer(ref::PtrIndexableRef{T}) where T
    ref.ptr + (ref.index - 1) * sizeof(T)
end

@lava_device_override function Atomix.gcroot(ref::PtrIndexableRef{T}) where T
    nothing
end

@lava_device_override function Base.eltype(ref::PtrIndexableRef{T}) where T
    T
end

# ── Pre-computed LLVM IR constants for atomicrmw ──
# All IR strings must be compile-time constants for llvmcall.

const _ir_add_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw add ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_sub_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw sub ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_and_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw and ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_or_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw or ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_xor_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw xor ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_min_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw min ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_max_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw max ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_umin_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw umin ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_umax_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw umax ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_xchg_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw xchg ptr %ptr, i32 %val syncscope("device") monotonic
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const _ir_f32_cas = ("""
    define i32 @entry(ptr %ptr, float %val) #0 {
    entry:
        %orig = load atomic i32, ptr %ptr syncscope("device") monotonic, align 4
        br label %loop
    loop:
        %old_bits = phi i32 [ %orig, %entry ], [ %loaded, %loop ]
        %old_f = bitcast i32 %old_bits to float
        %new_f = fadd float %old_f, %val
        %new_bits = bitcast float %new_f to i32
        %result = cmpxchg ptr %ptr, i32 %old_bits, i32 %new_bits syncscope("device") monotonic monotonic
        %loaded = extractvalue { i32, i1 } %result, 0
        %success = extractvalue { i32, i1 } %result, 1
        br i1 %success, label %done, label %loop
    done:
        ret i32 %old_bits
    }
    attributes #0 = { alwaysinline }
""", "entry")

# ── Int32 atomics ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(+), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_add_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(-), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_sub_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old - val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(&), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_and_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old & val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(|), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_or_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old | val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(xor), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_xor_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => xor(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(min), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_min_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => min(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(max), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_max_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => max(old, val)
end

# ── UInt32 atomics ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(+), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_add_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(-), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_sub_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old - val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(&), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_and_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old & val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(|), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_or_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old | val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(min), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_umin_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => min(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(max), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_umax_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => max(old, val)
end

# ── Exchange (swap) ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(Atomix.right), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_xchg_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(Atomix.right), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(_ir_xchg_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => val
end

# ── Float32 atomic add via CAS loop ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Float32}, ::typeof(+), val::Float32, ::typeof(UnsafeAtomics.monotonic)
)
    old_bits = Base.llvmcall(_ir_f32_cas, UInt32, Tuple{Ptr{Float32}, Float32}, ptr, val)
    old = reinterpret(Float32, old_bits)
    old => old + val
end

# ── Seq_cst ordering wrappers ──
# Atomix defaults to seq_cst. Redirect to monotonic since Vulkan doesn't
# really distinguish orderings beyond device scope.

for T in (Int32, UInt32, Float32)
    for OP in (:(typeof(+)), :(typeof(-)), :(typeof(&)), :(typeof(|)), :(typeof(xor)),
               :(typeof(min)), :(typeof(max)), :(typeof(Atomix.right)))
        if T == Float32 && OP != :(typeof(+))
            continue
        end
        if T == Float32 && OP in (:(typeof(&)), :(typeof(|)), :(typeof(xor)))
            continue
        end
        @eval @lava_device_override @inline function UnsafeAtomics.modify!(
            ptr::Ptr{$T}, op::$OP, val::$T, ::typeof(UnsafeAtomics.seq_cst)
        )
            UnsafeAtomics.modify!(ptr, op, val, UnsafeAtomics.monotonic)
        end
    end
end

# ── Base.modifyindex_atomic! override for LavaDeviceArray ──
# Atomix 1.1.2+ @atomic macro expands to Base.modifyindex_atomic! on Julia 1.12+.
# Without this override, the kernel tries to call jl_f_throw_methoderror on GPU.

@lava_device_override @inline function Base.modifyindex_atomic!(
    arr::LavaDeviceArray{T,N}, order::Symbol, op::OP, val, i::Integer
) where {T, N, OP}
    @inbounds ptr = pointer(arr, i)
    v = convert(T, val)
    result = UnsafeAtomics.modify!(ptr, op, v, UnsafeAtomics.seq_cst)
    return result
end
