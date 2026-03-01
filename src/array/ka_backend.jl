# KernelAbstractions.jl backend for Lava.jl
#
# Implements LavaBackend <: KA.GPU for Vulkan compute dispatch.
# Pattern follows AMDGPU.jl's ROCKernels implementation.

import KernelAbstractions as KA
import Adapt

export LavaBackend

# ── Backend struct ──

struct LavaBackend <: KA.GPU end

# ── Backend queries ──

KA.get_backend(::LavaArray) = LavaBackend()
KA.synchronize(::LavaBackend) = vk_flush!()
KA.allocate(::LavaBackend, ::Type{T}, dims::Tuple) where T = LavaArray{T}(undef, Int.(dims))
KA.unsafe_free!(x::LavaArray) = unsafe_free!(x)

# Adapt: convert Array ↔ LavaArray
Adapt.adapt_storage(::LavaBackend, a::Array) = LavaArray(a)
Adapt.adapt_storage(::LavaBackend, a::LavaArray) = a
Adapt.adapt_storage(::KA.CPU, a::LavaArray) = Array(a)

# ── Launch configuration ──

function KA.launch_config(kernel::KA.Kernel{LavaBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize,)
    end

    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KA.workgroupsize(kernel) <: KA.DynamicSize && workgroupsize === nothing
        # Default workgroup size: 64 for 1D, capped to ndrange
        workgroupsize = ntuple(
            i -> i == 1 ? min(prod(ndrange), 64) : 1,
            length(ndrange))
        KA.partition(kernel, ndrange, workgroupsize)
    else
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

# ── Context creation ──

function KA.mkcontext(kernel::KA.Kernel{LavaBackend}, _ndrange, iterspace)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end

function KA.mkcontext(kernel::KA.Kernel{LavaBackend}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

# ── Argument conversion ──

# Convert kernel arguments for GPU compilation via Adapt.jl.
# LavaArray → Ptr{T} (BDA address). For compound args like Broadcasted,
# Adapt.adapt recursively converts nested LavaArray fields to Ptr{T}.
# We use Ptr{T} rather than LavaDeviceArray{T,N} because passing structs
# through BDA creates alloca+memcpy patterns that StructurizeCFG breaks.
KA.argconvert(::KA.Kernel{LavaBackend}, arg) = Adapt.adapt(LavaAdaptor(), arg)

# ── Kernel call (main entry point) ──

function (obj::KA.Kernel{LavaBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    nblocks = length(KA.blocks(iterspace))
    nthreads = length(KA.workitems(iterspace))
    nblocks == 0 && return nothing

    # Convert workgroup/blocks to 3D tuples for Vulkan dispatch
    ws_3d = _pad_to_3d(workgroupsize === nothing ? KA.get(KA.workgroupsize(obj)) : workgroupsize)

    # Collect all arguments: function first (for BDA self param), ctx, then user args
    # GPUCompiler includes typeof(f) as the first LLVM parameter.
    # wrap_entry_for_vulkan! creates a BDA slot for it (unless ghost-elided).
    # We must include f in the args so BDA packing matches the layout.
    all_args = (obj.f, ctx, KA.argconvert.(Ref(obj), args)...)

    # Launch via our compilation pipeline
    _ka_launch!(obj.f, all_args, nblocks, ws_3d)

    return nothing
end

function _pad_to_3d(t::NTuple{1,<:Integer})
    (Int(t[1]), 1, 1)
end
function _pad_to_3d(t::NTuple{2,<:Integer})
    (Int(t[1]), Int(t[2]), 1)
end
function _pad_to_3d(t::NTuple{3,<:Integer})
    (Int(t[1]), Int(t[2]), Int(t[3]))
end
# For N>3, collapse to 1D (Vulkan max is 3D dispatch; KA handles N-D via iterspace)
function _pad_to_3d(t::NTuple{N,<:Integer}) where N
    (Int(prod(t)), 1, 1)
end

"""
Internal launch function for KA kernels. Compiles and dispatches the GPU function.
"""
function _ka_launch!(@nospecialize(f), all_args::Tuple, nblocks::Int, workgroup_size::NTuple{3,Int})
    # Build type tuple for compilation (excludes f — GPUCompiler prepends typeof(f))
    # all_args[1] is f itself (included for BDA packing), rest are the actual args
    tt = Tuple{map(_ka_arg_llvm_type, Base.tail(all_args))...}

    # Compile (cached)
    compiled = _get_compiled_kernel(f, tt, workgroup_size)

    # Create pipeline (cached)
    pipeline = get_compute_pipeline(compiled.spirv_bytes, compiled.entry_name;
                                     push_constant_size=compiled.push_info.push_size)

    # Convert arguments to BDA-compatible values (filters ghost types)
    bda_args = _ka_args_to_bda(all_args)

    # Compute total size: base layout + inline struct data
    inline_extra = sum(arg isa InlineStructArg ? ((sizeof(arg.bytes) + 7) & ~7) : 0
                       for arg in bda_args; init=0)
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # Get host-visible mapped arg buffer (zero-cost write via memcpy)
    arg_buf = _get_arg_buffer(total_size)

    # Pack with self-referencing BDAs for inline structs
    arg_data = pack_kernel_args_inline(bda_args, compiled.push_info.arg_layout,
                                        compiled.push_info.arg_buffer_size,
                                        arg_buf.address)
    # Write directly to mapped memory — no staging copy needed
    unsafe_copyto!(arg_buf.mapped_ptr, pointer(arg_data), length(arg_data))

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)

    # Compute grid dimensions (nblocks is total blocks, workgroup_size is per-block)
    # nblocks from KA is the product of all block dimensions
    # For 1D, groups = (nblocks, 1, 1)
    groups = (nblocks, 1, 1)

    vk_dispatch!(pipeline, push_data, groups)

    # Don't flush here — dispatches are batched. KA.synchronize() calls vk_flush!().
    # Implicit sync points (copyto!, Array(), scalar indexing) auto-flush via
    # vk_flush!() in the memory transfer paths.

    return nothing
end

# ── Argument type mapping for KA ──

# Map KA arguments to LLVM types for compilation
# LavaDeviceArray is isbits and passed as a struct (via InlineStructArg)
_ka_arg_llvm_type(x) = typeof(x)  # Everything passes through as-is

# Convert KA arguments to BDA-compatible values for packing.
# Ghost types (Val{}, Nothing, zero-sized singletons) are filtered out —
# GPUCompiler elides them from LLVM IR so they must not appear in arg buffer.
function _ka_args_to_bda(all_args::Tuple)
    result = Any[]
    for x in all_args
        T = typeof(x)
        # Skip ghost types: GPUCompiler.isghosttype checks sizeof==0 + singleton
        if GPUCompiler.isghosttype(T) || Core.Compiler.isconstType(T)
            continue
        end
        if isbitstype(T) && !isprimitivetype(T)
            data = Vector{UInt8}(undef, sizeof(T))
            unsafe_store!(Ptr{T}(pointer(data)), x)
            push!(result, InlineStructArg(data))
        else
            push!(result, x)
        end
    end
    return tuple(result...)
end

# ── Device-side indexing for GPU arrays ──

# 0D array indexing (used by broadcasting on scalar-wrapped arrays)
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,0}) where T
    unsafe_load(a.ptr)
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,0}, v) where T
    unsafe_store!(a.ptr, convert(T, v))
    return v
end

# 0D with CartesianIndex (broadcast dispatches CartesianIndex{1} on 0D arrays)
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,0}, I::CartesianIndex) where T
    unsafe_load(a.ptr)
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,0}, v, I::CartesianIndex) where T
    unsafe_store!(a.ptr, convert(T, v))
    return v
end

# LavaDeviceArray{T,N} linear indexing via BDA pointer
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T}, i::Integer) where T
    unsafe_load(a.ptr, i)
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T}, v, i::Integer) where T
    unsafe_store!(a.ptr, convert(T, v), i)
    return v
end

# CartesianIndex indexing for multi-dimensional broadcast support (any N)
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,N}, I::CartesianIndex{N}) where {T,N}
    @inbounds unsafe_load(a.ptr, _linear_index(a.dims, I))
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
    @inbounds unsafe_store!(a.ptr, convert(T, v), _linear_index(a.dims, I))
    return v
end

# Convert CartesianIndex to linear index for LavaDeviceArray
@inline function _linear_index(dims::NTuple{1,Int}, I::CartesianIndex{1})
    I[1]
end
@inline function _linear_index(dims::NTuple{2,Int}, I::CartesianIndex{2})
    I[1] + dims[1] * (I[2] - 1)
end
@inline function _linear_index(dims::NTuple{3,Int}, I::CartesianIndex{3})
    I[1] + dims[1] * (I[2] - 1) + dims[1] * dims[2] * (I[3] - 1)
end
@inline function _linear_index(dims::NTuple{N,Int}, I::CartesianIndex{N}) where N
    idx = I[1]
    stride = 1
    for d in 2:N
        stride *= dims[d-1]
        idx += stride * (I[d] - 1)
    end
    return idx
end

# Ptr{T} indexing (used by lava_launch! path where arrays become Ptr{T})
@lava_device_override @inline function Base.getindex(p::Ptr{T}, i::Integer) where T
    unsafe_load(p, i)
end

@lava_device_override @inline function Base.setindex!(p::Ptr{T}, v, i::Integer) where T
    unsafe_store!(p, convert(T, v), i)
    return v
end

# ── Device-side index functions ──
# These are overridden via @lava_device_override to use SPIR-V builtins.
# The ctx argument (CompilerMetadata) carries ndrange/iterspace info.

@lava_device_override @inline function KA.__index_Local_Linear(ctx)
    return Int(lava_local_invocation_id_x()) + 1
end

@lava_device_override @inline function KA.__index_Group_Linear(ctx)
    return Int(lava_workgroup_id_x()) + 1
end

@lava_device_override @inline function KA.__index_Global_Linear(ctx)
    I = @inbounds KA.expand(KA.__iterspace(ctx), Int(lava_workgroup_id_x()) + 1, Int(lava_local_invocation_id_x()) + 1)
    @inbounds LinearIndices(KA.__ndrange(ctx))[I]
end

@lava_device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[Int(lava_local_invocation_id_x()) + 1]
end

@lava_device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[Int(lava_workgroup_id_x()) + 1]
end

@lava_device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx), Int(lava_workgroup_id_x()) + 1, Int(lava_local_invocation_id_x()) + 1)
end

@lava_device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), Int(lava_workgroup_id_x()) + 1, Int(lava_local_invocation_id_x()) + 1)
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end

# ── Shared Memory ──

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
    LavaSharedArray{T}(ptr, N)
end

# Simple 1D shared array backed by LLVMPtr in addrspace(3)
struct LavaSharedArray{T}
    ptr::Core.LLVMPtr{T, 3}
    len::Int
end

Base.@propagate_inbounds function Base.getindex(a::LavaSharedArray{T}, i::Integer) where T
    @boundscheck (1 <= i <= a.len || throw(BoundsError(a, i)))
    unsafe_load(a.ptr, i)
end

Base.@propagate_inbounds function Base.setindex!(a::LavaSharedArray{T}, v, i::Integer) where T
    @boundscheck (1 <= i <= a.len || throw(BoundsError(a, i)))
    unsafe_store!(a.ptr, convert(T, v), i)
    return v
end

Base.length(a::LavaSharedArray) = a.len
Base.eltype(::LavaSharedArray{T}) where T = T

# ── Synchronization ──

@lava_device_override @inline function KA.__synchronize()
    lava_workgroup_barrier()
end

# ── Print (no-op on GPU) ──

@lava_device_override @inline function KA.__print(args...)
    # GPU print not yet supported
end
