# Workgroup-shared memory: what `@localmem` becomes.
#
# DEVICE code, so it is Lava's. `lava_alloc_shared` builds an LLVM global in
# addrspace(3) and hands back a typed `LLVMPtr`; `LavaSharedArray` indexes it.
# Nothing here touches a queue, a pool or a context — the shared block is part of
# the SPIR-V module, allocated by the driver when the workgroup launches.
#
# It travelled to Mantle with `array/ka_backend.jl` in the 2026-08-27 move and
# came straight back: a test that only compiles a kernel had to import it from
# the runtime, which is how the misplacement showed up.
#
# The `CoopMatrix` constructors that load a matrix out of a shared tile are here
# too — they are device code over a device type, and keeping them beside the
# array they read is why this file is included after `acceleratedmatrix.jl`.
#
# `KA` is aliased here rather than at the top of `Lava.jl`: this is the only file
# in the compiler that overrides a KernelAbstractions function, and the alias
# came with the code from `array/ka_backend.jl`, which is Mantle's now.
import KernelAbstractions as KA

# Allocate shared memory (Workgroup storage class) for GPU kernels.
# Follows AMDGPU's pattern: create an LLVM global variable in addrspace(3),
# return a typed LLVMPtr. Each call site gets a unique global via Val{Id}.
@inline @generated function lava_alloc_shared(::Val{Id}, ::Type{T}, ::Val{N}) where {Id, T, N}
    Context() do ctx
        eltyp = convert(LLVM.LLVMType, T)

        # Unique global name from the call-site Id
        gv_name = "lava_shared_$Id"

        T_ptr = convert(LLVM.LLVMType, Core.LLVMPtr{T, 3})

        # Create a function returning ptr addrspace(3)
        llvm_f, _ = LLVM.Interop.create_function(T_ptr)
        mod = LLVM.parent(llvm_f)

        # Create global variable: [N x T] in addrspace(3)
        gv_typ = LLVM.ArrayType(eltyp, N)
        gv = LLVM.GlobalVariable(mod, gv_typ, gv_name, 3)
        LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
        LLVM.alignment!(gv, max(16, Base.datatype_alignment(T)))

        # Generate IR: GEP to get pointer to first element, return it
        @dispose builder=LLVM.IRBuilder() begin
            entry = LLVM.BasicBlock(llvm_f, "entry")
            LLVM.position!(builder, entry)
            ptr = LLVM.gep!(builder, gv_typ, gv, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)])
            LLVM.ret!(builder, ptr)
        end

        LLVM.Interop.call_function(llvm_f, Core.LLVMPtr{T, 3})
    end
end

# KA SharedMemory override: allocate workgroup-shared array, return indexable wrapper
@lava_device_override @inline function KA.SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    N = prod(Dims)
    ptr = lava_alloc_shared(Val(Id), T, Val(N))
    LavaSharedArray{T, Dims}(ptr, N)
end

# Shared array backed by LLVMPtr in addrspace(3). `Dims` (a tuple type parameter)
# carries the static shape so multi-dimensional indexing `a[i, j]` works without a
# runtime shape field. Storage is column-major, matching Julia/LavaArray.
struct LavaSharedArray{T, Dims}
    ptr::Core.LLVMPtr{T, 3}
    len::Int
end

# Backward-compatible 1-D construction (`LavaSharedArray{T}(ptr, len)`): defaults
# the shape to `(len,)`. `len` is a compile-time constant at every call site
# (literal, or `prod(Val(Dims))`), so the `(len,)` type parameter is stable.
@inline LavaSharedArray{T}(ptr::Core.LLVMPtr{T, 3}, len::Integer) where {T} =
    LavaSharedArray{T, (Int(len),)}(ptr, Int(len))

"""
    AcceleratedMatrix{T,M,N,Use}(shared, offset, stride)

Load a cooperative matrix out of `@localmem` rather than out of global memory —
the same call as the device-array form, with a `Workgroup` pointer instead of a
device address.

Here rather than beside the rest of `AcceleratedMatrix` only because
`device/acceleratedmatrix.jl` is included before this type exists.

Taking the array rather than making callers reach for `.ptr`: that field is an
implementation detail, and a kernel reading `AcceleratedMatrix{...}(tile, 1, 16)`
should say the same thing whether `tile` is shared or global.
"""
@inline CoopMatrix{T,M,N,U,S}(src::LavaSharedArray, offset::Integer,
                              stride::Integer) where {T,M,N,U,S} =
    coopmat_load(CoopMatrix{T,M,N,U,S}, src.ptr, offset, stride)

"""
    AcceleratedMatrix{T,M,N,U}(shared, offset, stride, Val(true))   # row-major

The `MemoryLayout` operand for a shared-memory load.

The global-pointer form has taken this since the staged GEMM needed A `RowMajor`
and B `ColumnMajor` out of one block, but the `@localmem` form did not, so a
kernel wanting a row-major operand out of shared memory got a method lookup
failure — which surfaces as `jl_f_throw_methoderror` in the middle of otherwise
valid GPU code rather than as a missing method at the call site.

Attention needs it in both directions. With `q`, `k`, `v` laid out `(E, L, H, B)`
the score product reads Q row-major and Kᵀ column-major from the same staging
stride, and the value product reads P and V both row-major — no transpose
anywhere, which is only available if the layout is a parameter here too.
"""
@inline CoopMatrix{T,M,N,U,S}(src::LavaSharedArray, offset::Integer,
                              stride::Integer, rowmajor::Val) where {T,M,N,U,S} =
    coopmat_load(CoopMatrix{T,M,N,U,S}, src.ptr, offset, stride, rowmajor)

"""Store a cooperative matrix into `@localmem`; see the load above."""
@inline Base.copyto!(dst::LavaSharedArray, offset::Integer, stride::Integer,
                     m::CoopMatrix) =
    coopmat_store(dst.ptr, offset, stride, m)

"""
    copyto!(shared, offset, stride, m, Val(true))   # row-major

The layout operand for a shared-memory *store*, the mirror of the load above.

The load has taken a layout since the staged GEMM needed one; the store did not,
which left the pair asymmetric for no reason — the emitter has always written
whichever `MemoryLayout` the intrinsic name carries, for stores as much as loads.

What it is for: the shape a tile is stored in decides how the scalar code that
reads it next can be parallelised. A tile produced with rows down one axis and
reduced along a row wants to be stored row-major, or the row's elements sit a
whole stride apart and land in one shared-memory bank.

`DNNKernels`' attention kernel is where that came up and it does **not** use
this: giving each subgroup a row and each lane a key measured 5% slower than the
thread-per-row loop it replaced, so the score tile went back to column-major.
The operand stays because the asymmetry was a gap rather than a decision, and
because the next kernel to want it should not have to discover that the emitter
was ready all along.
"""
@inline Base.copyto!(dst::LavaSharedArray, offset::Integer, stride::Integer,
                     m::CoopMatrix, rowmajor::Val) =
    coopmat_store(dst.ptr, offset, stride, m, rowmajor)

# Column-major linear index from an N-d cartesian index, fully constant-folded
# (both `dims` and the index arity are compile-time constants here).
@inline function lava_shared_linear(dims::NTuple{N,Int}, I::NTuple{N,Integer}) where N
    @inbounds begin
        lin = Int(I[N]) - 1
        for d in (N-1):-1:1
            lin = lin * dims[d] + (Int(I[d]) - 1)
        end
        return lin + 1
    end
end

# Linear (single-index) access — works for any shape.
Base.@propagate_inbounds function Base.getindex(a::LavaSharedArray{T}, i::Integer) where T
    @boundscheck (1 <= i <= a.len || throw(BoundsError(a, i)))
    unsafe_load(a.ptr, i)
end

Base.@propagate_inbounds function Base.setindex!(a::LavaSharedArray{T}, v, i::Integer) where T
    @boundscheck (1 <= i <= a.len || throw(BoundsError(a, i)))
    unsafe_store!(a.ptr, convert(T, v), i)
    return v
end

# Multi-dimensional access (≥2 indices; the 1-index method above handles linear).
# Without these, `a[i, j]` on a 2-D `@localmem` hits no method → a MethodError
# throw path → `gpu_gc_pool_alloc` (heap alloc in the kernel) → compile failure.
Base.@propagate_inbounds function Base.getindex(a::LavaSharedArray{T, Dims},
                                                 i1::Integer, i2::Integer, Irest::Integer...) where {T, Dims}
    lin = lava_shared_linear(Dims, (i1, i2, Irest...))
    @boundscheck (1 <= lin <= a.len || throw(BoundsError(a, (i1, i2, Irest...))))
    unsafe_load(a.ptr, lin)
end

Base.@propagate_inbounds function Base.setindex!(a::LavaSharedArray{T, Dims}, v,
                                                 i1::Integer, i2::Integer, Irest::Integer...) where {T, Dims}
    lin = lava_shared_linear(Dims, (i1, i2, Irest...))
    @boundscheck (1 <= lin <= a.len || throw(BoundsError(a, (i1, i2, Irest...))))
    unsafe_store!(a.ptr, convert(T, v), lin)
    return v
end

Base.length(a::LavaSharedArray) = a.len
Base.size(::LavaSharedArray{T, Dims}) where {T, Dims} = Dims
Base.eltype(::LavaSharedArray{T}) where T = T

