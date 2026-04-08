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

# LavaDeviceArray atomics: wrap with LavaAtomicRef to preserve dims for CartesianIndex
struct LavaAtomicRef{T, N}
    ptr::Ptr{T}
    dims::NTuple{N, Int}
end

@lava_device_override function Atomix.Internal.referenceable(a::LavaDeviceArray{T,N}) where {T,N}
    LavaAtomicRef{T,N}(a.ptr, a.dims)
end

@lava_device_override function Base.getindex(r::LavaAtomicRef{T}, i::Integer) where T
    PtrIndexableRef{T}(r.ptr, Int(i))
end

@lava_device_override function Base.getindex(r::LavaAtomicRef{T,N}, I::CartesianIndex{N}) where {T,N}
    li = atomic_linear_index(r.dims, I)
    PtrIndexableRef{T}(r.ptr, li)
end

# CartesianIndex to linear index for atomic refs (mirrors linear_index in ka_backend.jl)
@inline function atomic_linear_index(dims::NTuple{N,Int}, I::CartesianIndex{N}) where N
    li = I[1]
    stride = 1
    for d in 2:N
        stride *= dims[d-1]
        li += stride * (I[d] - 1)
    end
    return li
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

const ir_add_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw add ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_sub_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw sub ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_and_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw and ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_or_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw or ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_xor_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw xor ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_min_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw min ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_max_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw max ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_umin_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw umin ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_umax_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw umax ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_xchg_i32 = ("""
    define i32 @entry(ptr %ptr, i32 %val) #0 {
        %old = atomicrmw xchg ptr %ptr, i32 %val syncscope("device") seq_cst
        ret i32 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

# ── Int64/UInt64 atomic IR ──

const ir_add_i64 = ("""
    define i64 @entry(ptr %ptr, i64 %val) #0 {
        %old = atomicrmw add ptr %ptr, i64 %val syncscope("device") seq_cst
        ret i64 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_sub_i64 = ("""
    define i64 @entry(ptr %ptr, i64 %val) #0 {
        %old = atomicrmw sub ptr %ptr, i64 %val syncscope("device") seq_cst
        ret i64 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_xchg_i64 = ("""
    define i64 @entry(ptr %ptr, i64 %val) #0 {
        %old = atomicrmw xchg ptr %ptr, i64 %val syncscope("device") seq_cst
        ret i64 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_min_i64 = ("""
    define i64 @entry(ptr %ptr, i64 %val) #0 {
        %old = atomicrmw min ptr %ptr, i64 %val syncscope("device") seq_cst
        ret i64 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_max_i64 = ("""
    define i64 @entry(ptr %ptr, i64 %val) #0 {
        %old = atomicrmw max ptr %ptr, i64 %val syncscope("device") seq_cst
        ret i64 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_umin_i64 = ("""
    define i64 @entry(ptr %ptr, i64 %val) #0 {
        %old = atomicrmw umin ptr %ptr, i64 %val syncscope("device") seq_cst
        ret i64 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_umax_i64 = ("""
    define i64 @entry(ptr %ptr, i64 %val) #0 {
        %old = atomicrmw umax ptr %ptr, i64 %val syncscope("device") seq_cst
        ret i64 %old
    }
    attributes #0 = { alwaysinline }
""", "entry")

# ── Float CAS loops ──

const ir_f32_cas = ("""
    define i32 @entry(ptr %ptr, float %val) #0 {
    entry:
        %orig = load atomic i32, ptr %ptr syncscope("device") seq_cst, align 4
        br label %loop
    loop:
        %old_bits = phi i32 [ %orig, %entry ], [ %loaded, %loop ]
        %old_f = bitcast i32 %old_bits to float
        %new_f = fadd float %old_f, %val
        %new_bits = bitcast float %new_f to i32
        %result = cmpxchg ptr %ptr, i32 %old_bits, i32 %new_bits syncscope("device") seq_cst seq_cst
        %loaded = extractvalue { i32, i1 } %result, 0
        %success = extractvalue { i32, i1 } %result, 1
        br i1 %success, label %done, label %loop
    done:
        ret i32 %old_bits
    }
    attributes #0 = { alwaysinline }
""", "entry")

const ir_f64_cas = ("""
    define i64 @entry(ptr %ptr, double %val) #0 {
    entry:
        %orig = load atomic i64, ptr %ptr syncscope("device") seq_cst, align 8
        br label %loop
    loop:
        %old_bits = phi i64 [ %orig, %entry ], [ %loaded, %loop ]
        %old_f = bitcast i64 %old_bits to double
        %new_f = fadd double %old_f, %val
        %new_bits = bitcast double %new_f to i64
        %result = cmpxchg ptr %ptr, i64 %old_bits, i64 %new_bits syncscope("device") seq_cst seq_cst
        %loaded = extractvalue { i64, i1 } %result, 0
        %success = extractvalue { i64, i1 } %result, 1
        br i1 %success, label %done, label %loop
    done:
        ret i64 %old_bits
    }
    attributes #0 = { alwaysinline }
""", "entry")

# ── Int32 atomics ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(+), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_add_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(-), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_sub_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old - val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(&), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_and_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old & val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(|), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_or_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => old | val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(xor), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_xor_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => xor(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(min), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_min_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => min(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(max), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_max_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => max(old, val)
end

# ── UInt32 atomics ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(+), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_add_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(-), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_sub_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old - val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(&), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_and_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old & val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(|), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_or_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => old | val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(min), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_umin_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => min(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(max), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_umax_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => max(old, val)
end

# ── Exchange (swap) ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int32}, ::typeof(Atomix.right), val::Int32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_xchg_i32, Int32, Tuple{Ptr{Int32}, Int32}, ptr, val)
    old => val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt32}, ::typeof(Atomix.right), val::UInt32, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_xchg_i32, UInt32, Tuple{Ptr{UInt32}, UInt32}, ptr, val)
    old => val
end

# ── Float32 atomic add/sub via CAS loop ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Float32}, ::typeof(+), val::Float32, ::typeof(UnsafeAtomics.monotonic)
)
    old_bits = Base.llvmcall(ir_f32_cas, UInt32, Tuple{Ptr{Float32}, Float32}, ptr, val)
    old = reinterpret(Float32, old_bits)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Float32}, ::typeof(-), val::Float32, ::typeof(UnsafeAtomics.monotonic)
)
    old_bits = Base.llvmcall(ir_f32_cas, UInt32, Tuple{Ptr{Float32}, Float32}, ptr, -val)
    old = reinterpret(Float32, old_bits)
    old => old - val
end

# ── Int64 atomics ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int64}, ::typeof(+), val::Int64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_add_i64, Int64, Tuple{Ptr{Int64}, Int64}, ptr, val)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int64}, ::typeof(-), val::Int64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_sub_i64, Int64, Tuple{Ptr{Int64}, Int64}, ptr, val)
    old => old - val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int64}, ::typeof(min), val::Int64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_min_i64, Int64, Tuple{Ptr{Int64}, Int64}, ptr, val)
    old => min(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int64}, ::typeof(max), val::Int64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_max_i64, Int64, Tuple{Ptr{Int64}, Int64}, ptr, val)
    old => max(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Int64}, ::typeof(Atomix.right), val::Int64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_xchg_i64, Int64, Tuple{Ptr{Int64}, Int64}, ptr, val)
    old => val
end

# ── UInt64 atomics ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt64}, ::typeof(+), val::UInt64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_add_i64, UInt64, Tuple{Ptr{UInt64}, UInt64}, ptr, val)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt64}, ::typeof(-), val::UInt64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_sub_i64, UInt64, Tuple{Ptr{UInt64}, UInt64}, ptr, val)
    old => old - val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt64}, ::typeof(min), val::UInt64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_umin_i64, UInt64, Tuple{Ptr{UInt64}, UInt64}, ptr, val)
    old => min(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt64}, ::typeof(max), val::UInt64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_umax_i64, UInt64, Tuple{Ptr{UInt64}, UInt64}, ptr, val)
    old => max(old, val)
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{UInt64}, ::typeof(Atomix.right), val::UInt64, ::typeof(UnsafeAtomics.monotonic)
)
    old = Base.llvmcall(ir_xchg_i64, UInt64, Tuple{Ptr{UInt64}, UInt64}, ptr, val)
    old => val
end

# ── Float64 atomic add/sub via CAS loop ──

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Float64}, ::typeof(+), val::Float64, ::typeof(UnsafeAtomics.monotonic)
)
    old_bits = Base.llvmcall(ir_f64_cas, UInt64, Tuple{Ptr{Float64}, Float64}, ptr, val)
    old = reinterpret(Float64, old_bits)
    old => old + val
end

@lava_device_override @inline function UnsafeAtomics.modify!(
    ptr::Ptr{Float64}, ::typeof(-), val::Float64, ::typeof(UnsafeAtomics.monotonic)
)
    old_bits = Base.llvmcall(ir_f64_cas, UInt64, Tuple{Ptr{Float64}, Float64}, ptr, -val)
    old = reinterpret(Float64, old_bits)
    old => old - val
end

# ── Seq_cst ordering wrappers ──
# Atomix defaults to seq_cst. Keep using the same implementation path as the
# monotonic methods, but note that the underlying LLVM atomics above are
# emitted as `seq_cst` for cross-vendor correctness.

for T in (Int32, UInt32, Int64, UInt64, Float32, Float64)
    for OP in (:(typeof(+)), :(typeof(-)), :(typeof(&)), :(typeof(|)), :(typeof(xor)),
               :(typeof(min)), :(typeof(max)), :(typeof(Atomix.right)))
        if T in (Float32, Float64) && OP ∉ (:(typeof(+)), :(typeof(-)))
            continue
        end
        if T in (Float32, Float64) && OP in (:(typeof(&)), :(typeof(|)), :(typeof(xor)))
            continue
        end
        if T in (Int64, UInt64) && OP in (:(typeof(&)), :(typeof(|)), :(typeof(xor)))
            continue  # no bitwise atomics for 64-bit (not commonly needed)
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
    v = convert(T, val)
    ref = Atomix.Internal.referenceable(arr)[i]
    return Atomix.modify!(ref, op, v, UnsafeAtomics.seq_cst)
end

@lava_device_override @inline function Base.modifyindex_atomic!(
    arr::LavaDeviceArray{T,N}, order::Symbol, op::OP, val, I::CartesianIndex{N}
) where {T, N, OP}
    v = convert(T, val)
    li = atomic_linear_index(arr.dims, I)
    ref = Atomix.Internal.referenceable(arr)[li]
    return Atomix.modify!(ref, op, v, UnsafeAtomics.seq_cst)
end
