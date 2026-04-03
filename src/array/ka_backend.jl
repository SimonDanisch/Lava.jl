# KernelAbstractions.jl backend for Lava.jl
#
# Implements LavaBackend <: KA.GPU for Vulkan compute dispatch.
# Pattern follows AMDGPU.jl's ROCKernels implementation.

import KernelAbstractions as KA
import Adapt

export LavaBackend

# ── Dispatch name helper ──
# Extract a descriptive kernel name for dispatch logging.
# For workqueue foreach dispatches, the KA kernel is always `_workqueue_map_kernel!`
# but the actual inner function (e.g. `vp_trace_shadow_rays_kernel!`) is in args.
# all_args layout: (kernel_func, ctx, inner_func, queue, extra_args...)
function _dispatch_name(@nospecialize(f), @nospecialize(all_args))
    fname = nameof(typeof(f))
    # Check if this is a workqueue_map_kernel dispatch — inner func is arg 3
    if length(all_args) >= 3 && occursin("workqueue_map_kernel", string(fname))
        inner = all_args[3]
        return "$(fname)[$(nameof(typeof(inner)))]"
    end
    return string(fname)
end

# ── Backend struct ──

struct LavaBackend <: KA.GPU end

# ── Backend queries ──

KA.get_backend(::LavaArray) = LavaBackend()
# KA.synchronize submits all recorded dispatches and waits for GPU completion.
# This matches CUDA/AMDGPU semantics: after synchronize(), the CPU can safely
# read GPU results. GPU-side ordering between dispatches is handled by pipeline
# barriers in record_dispatch!, so synchronize() is only needed when the CPU
# must observe GPU results (or at natural batch boundaries like end-of-sample).
KA.synchronize(::LavaBackend) = vk_flush!()
KA.supports_unified(::LavaBackend) = true
function KA.allocate(::LavaBackend, ::Type{T}, dims::Tuple; unified::Bool=false) where T
    nbytes = prod(dims) * sizeof(T)
    # Use unified (BAR) memory when explicitly requested OR for tiny allocations
    # (≤ 64 bytes, e.g. WorkQueue.size counters). BAR memory enables direct CPU
    # readback without staging copy — 2x faster for queue length checks.
    if unified || nbytes <= 64
        managed = vk_alloc_unified(nbytes)
        ref = GPUArrays.DataRef(managed) do buf
            vk_free!(buf)
        end
        return LavaArray{T,length(dims)}(ref, Int.(dims))
    end
    LavaArray{T}(undef, Int.(dims))
end
KA.unsafe_free!(x::LavaArray) = unsafe_free!(x)

function KA.copyto!(::LavaBackend, A, B)
    GC.@preserve A B begin
        copyto!(A, 1, B, 1, length(A))
    end
    return
end

# Adapt: convert Array ↔ LavaArray
# Use Adapt.adapt(LavaArray, a) for recursive element adaptation (like AMDGPU does).
# This handles non-isbits element types by recursively adapting struct fields.
Adapt.adapt_storage(::LavaBackend, a::Array) = Adapt.adapt(LavaArray, a)
Adapt.adapt_storage(::LavaBackend, a::LavaArray) = a
Adapt.adapt_storage(::KA.CPU, a::LavaArray) = Array(a)

# @Const support: stop Adapt recursion at the device array level.
# Without this, constify on SubArray{...,ReshapedArray{...,LavaDeviceArray}} recurses into
# ReshapedArray → calls Base.reshape → SignedMultiplicativeInverse constructor on GPU,
# which has a throw(ArgumentError("...$d")) that generates ijl_get_nth_field_checked.
# AMDGPU handles this by casting the pointer to constant address space; we just return
# the device array as-is since Vulkan BDA pointers are already readonly-capable.
Adapt.adapt_storage(::KA.ConstAdaptor, a::LavaDeviceArray) = a
# Prevent ReshapedArray reconstruction during constify — @Const marks readonly,
# it should not reconstruct wrapper arrays (which triggers SignedMultiplicativeInverse
# constructor with its throwing string interpolation path on GPU).
Adapt.adapt_structure(::KA.ConstAdaptor, A::Base.ReshapedArray) = A

# Type-based adapt_storage for Adapt.adapt(LavaArray, x) dispatch
Adapt.adapt_storage(::Type{<:LavaArray}, a::Array) = LavaArray(a)
Adapt.adapt_storage(::Type{<:LavaArray}, a::LavaArray) = a

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
    _validate_launch_args(args)
    # GPU-resident ndrange → indirect dispatch (no CPU readback)
    if ndrange isa LavaArray
        converted_args = KA.argconvert.(Ref(obj), args)
        # Keep original args alive — argconvert strips LavaArray → Ptr (no backing ref).
        # Pass original_args so ka_launch_indirect! can re-establish keepalive
        # after internal vk_flush!() calls that clear batch data_refs.
        keep_data_alive!(args)
        ka_launch_indirect!(obj, converted_args, ndrange, workgroupsize, args)
        return nothing
    end

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    blocks = KA.blocks(iterspace)
    nblocks = length(blocks)
    nblocks == 0 && return nothing

    # N-D block grid for dispatch (avoids exceeding per-dimension workgroup count limits).
    # Workgroups are always 1D (matching CUDA.jl/AMDGPU.jl pattern): the local thread
    # index is just threadIdx.x / lava_local_invocation_id_x(), and KA.expand() maps
    # the (linear_block, linear_local) pair back to N-D indices via the iterspace.
    block_dims = pad_to_3d(size(blocks))
    nthreads = length(KA.workitems(iterspace))
    ws_3d = (nthreads, 1, 1)

    # Collect all arguments: function first (for BDA self param), ctx, then user args
    # GPUCompiler includes typeof(f) as the first LLVM parameter.
    # wrap_entry_for_vulkan! creates a BDA slot for it (unless ghost-elided).
    # We must include f in the args so BDA packing matches the layout.
    converted_args = KA.argconvert.(Ref(obj), args)
    all_args = (obj.f, ctx, converted_args...)

    # Keep original args alive until vk_flush!() — argconvert converts
    # LavaArray → LavaDeviceArray(Ptr{T}), losing the backing buffer reference.
    # Without this, GC can free the VkManagedBuffer while the GPU command buffer
    # still references it via BDA, causing DEVICE_LOST under GC pressure.
    keep_data_alive!(args)

    # Launch via our compilation pipeline
    ka_launch!(obj.f, all_args, block_dims, ws_3d)

    return nothing
end

# Vulkan dispatch is max 3D. pad_to_3d maps the N-D block grid to a 3D dispatch,
# splitting large dimensions across Y/Z when needed to stay within device limits.
# Uses actual device maxComputeWorkGroupCount (queried once, cached).
#
# CRITICAL: pad_to_3d must NEVER over-dispatch (produce more workgroups than requested).
# Kernels like AK._accumulate_block! write to auxiliary arrays indexed by workgroup ID
# without bounds checks — phantom workgroups cause out-of-bounds GPU memory writes.

# Cached per-dimension workgroup count limits (filled on first use from device properties)
const _MAX_WG_DIMS = Ref((65535, 65535, 65535))  # conservative defaults
const _MAX_WG_DIMS_INITIALIZED = Ref(false)

function _init_max_wg_dims!()
    _MAX_WG_DIMS_INITIALIZED[] && return
    ctx = vk_context()
    props = Vulkan.get_physical_device_properties(ctx.physical_device)
    wgc = props.limits.max_compute_work_group_count
    _MAX_WG_DIMS[] = (Int(wgc[1]), Int(wgc[2]), Int(wgc[3]))
    _MAX_WG_DIMS_INITIALIZED[] = true
end

push!(_reset_callbacks, function()
    _MAX_WG_DIMS_INITIALIZED[] = false
end)

# Find exact factor of n that is ≤ max_dim, for splitting workgroup counts.
# Returns the largest factor ≤ max_dim, or max_dim if none found (over-dispatch).
function _find_split_factor(n::Int, max_dim::Int)
    # Try exact division first (fast path for powers of 2, multiples of common factors)
    for d in (max_dim, max_dim-1, max_dim-2, max_dim-3)
        d > 0 && n % d == 0 && return d
    end
    # Try small factors of n that would make the other dimension ≤ max_dim
    # We need X such that X ≤ max_dim and n/X ≤ max_dim (for 2D split)
    # So X ≥ cld(n, max_dim)
    lo = cld(n, max_dim)
    for x in lo:min(n, max_dim)
        n % x == 0 && return x
    end
    # No exact factor found — find tightest over-approximation
    # This should be extremely rare (requires n > max_dim^2 with no factors in range)
    return max_dim
end

function pad_to_3d(t::NTuple{1,<:Integer})
    _init_max_wg_dims!()
    n = Int(t[1])
    max_x, max_y, max_z = _MAX_WG_DIMS[]
    n <= max_x && return (n, 1, 1)
    # Need to split into X * Y (or X * Y * Z)
    # Find X such that X divides n exactly and X ≤ max_x
    x = _find_split_factor(n, max_x)
    y = cld(n, x)
    if x * y == n && y <= max_y
        return (x, y, 1)
    end
    # 2D split didn't work exactly, try 3D
    if y > max_y
        # Flatten into 3D
        xy = x * min(y, max_y)
        z = cld(n, xy)
        return (x, min(y, max_y), z)
    end
    # Over-dispatch (y > exact). This is dangerous for kernels without bounds checks.
    @warn "pad_to_3d: over-dispatching $n as ($x, $y, 1) = $(x*y) workgroups" maxlog=1
    return (x, y, 1)
end
function pad_to_3d(t::NTuple{2,<:Integer})
    _init_max_wg_dims!()
    x, y = Int(t[1]), Int(t[2])
    max_x, max_y, _ = _MAX_WG_DIMS[]
    if x <= max_x && y <= max_y
        return (x, y, 1)
    end
    # Flatten and re-split
    return pad_to_3d((x * y,))
end
function pad_to_3d(t::NTuple{3,<:Integer})
    (Int(t[1]), Int(t[2]), Int(t[3]))
end
# For N>3, flatten then split
function pad_to_3d(t::NTuple{N,<:Integer}) where N
    pad_to_3d((Int(prod(t)),))
end

"""
Internal launch function for KA kernels. Compiles and dispatches the GPU function.
"""
const _DBG_LAUNCH_COUNT = Ref(0)

function ka_launch!(@nospecialize(f), all_args::Tuple, block_dims::NTuple{3,Int}, workgroup_size::NTuple{3,Int})
    _DBG_LAUNCH_COUNT[] += 1
    _n = _DBG_LAUNCH_COUNT[]
    # Build type tuple for compilation (excludes f — GPUCompiler prepends typeof(f))
    # all_args[1] is f itself (included for BDA packing), rest are the actual args
    tt = Tuple{map(ka_arg_llvm_type, Base.tail(all_args))...}

    # Compile + pipeline + offsets (cached, single lookup)
    compiled, pipeline, offsets, byval_sizes = _get_compiled_kernel_and_pipeline(f, tt, workgroup_size)

    # Compute total size: base layout + inline struct data
    inline_extra = _compute_inline_extra_from_byval(byval_sizes)
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # Get host-visible mapped arg buffer
    arg_buf = get_arg_buffer(total_size)

    # Pack args directly to mapped memory (zero intermediate allocations)
    _pack_args_direct!(arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    # Dispatch with N-D block grid (preserves KA's block dimensions)
    if dispatch_logging_enabled[]
        last_dispatch_info[] = "ka f=$(_dispatch_name(f, all_args)) groups=$block_dims"
    end
    bq = something(current_batch_queue(), vk_context().default_bq)
    vk_dispatch!(bq, pipeline, arg_buf.address, block_dims)

    return nothing
end

# ── Argument type mapping for KA ──

# Map KA arguments to LLVM types for compilation
# LavaDeviceArray is isbits and passed as a struct (via InlineStructArg)
ka_arg_llvm_type(x) = typeof(x)  # Everything passes through as-is

# ── Indirect dispatch support ──

function _prepare_indirect_kernel(indirect::Ptr{UInt32}, ndrange_buf::Ptr{Int32}, ws::UInt32)
    n = UInt32(unsafe_load(ndrange_buf, 1))
    groups = (n + ws - UInt32(1)) ÷ ws
    unsafe_store!(indirect, groups, 1)    # groupCountX
    unsafe_store!(indirect, UInt32(1), 2)  # groupCountY
    unsafe_store!(indirect, UInt32(1), 3)  # groupCountZ
    return nothing
end

# ── Fast prepare-indirect path ──
# Bypasses full lava_launch! to avoid per-dispatch ceremony overhead.
# The prepare-indirect kernel is always the same function with the same types,
# so we compile once and cache everything. Saves ~30K lava_launch! calls per render
# (hash lookups, validation, auto-flush, keep_alive, etc.).

const _prepare_indirect_pipeline_ref = Ref{Union{Nothing, LavaComputePipeline}}(nothing)
const _prepare_indirect_offsets_ref = Ref{Union{Nothing, Vector{Int}}}(nothing)
const _prepare_indirect_byval_ref = Ref{Union{Nothing, Vector{Int}}}(nothing)
const _prepare_indirect_arg_buf_size_ref = Ref{Int}(0)

# Register cleanup callback for vk_reset_device!
push!(_reset_callbacks, function()
    _prepare_indirect_pipeline_ref[] = nothing
    _prepare_indirect_offsets_ref[] = nothing
    _prepare_indirect_byval_ref[] = nothing
    _prepare_indirect_arg_buf_size_ref[] = 0
end)

function _init_prepare_indirect_pipeline!()
    _prepare_indirect_pipeline_ref[] !== nothing && return
    tt = Tuple{Ptr{UInt32}, Ptr{Int32}, UInt32}
    ws = (1, 1, 1)
    compiled, pipeline, offsets, byval_sizes = _get_compiled_kernel_and_pipeline(
        _prepare_indirect_kernel, tt, ws)
    _prepare_indirect_pipeline_ref[] = pipeline
    _prepare_indirect_offsets_ref[] = offsets
    _prepare_indirect_byval_ref[] = byval_sizes
    _prepare_indirect_arg_buf_size_ref[] = compiled.push_info.arg_buffer_size
end

"""
    _fast_prepare_indirect!(indirect_buf, ndrange_buf, workgroup_size)

Fast path for prepare-indirect dispatch. Bypasses lava_launch! entirely:
no validation, no auto-flush check, no keep_data_alive!, no logging overhead.
Records a single-thread direct dispatch to compute ceil(n/ws) group counts.
"""
function _fast_prepare_indirect!(indirect_buf::VkIndirectBuffer, ndrange_buf::LavaArray{<:Integer}, workgroup_size::Integer)
    _init_prepare_indirect_pipeline!()

    pipeline = _prepare_indirect_pipeline_ref[]
    offsets = _prepare_indirect_offsets_ref[]
    byval_sizes = _prepare_indirect_byval_ref[]
    arg_size = _prepare_indirect_arg_buf_size_ref[]

    # Pack args: f (ghost, skipped), Ptr{UInt32} (BDA), LavaArray (BDA), UInt32 (direct)
    all_args = (_prepare_indirect_kernel, Ptr{UInt32}(indirect_buf.address), ndrange_buf, UInt32(workgroup_size))
    arg_buf = get_arg_buffer(arg_size)
    _pack_args_direct!(arg_buf.mapped_ptr, arg_buf.address, offsets, arg_size, byval_sizes, all_args)

    # Record dispatch directly — single workgroup of 1 thread
    bq = something(current_batch_queue(), vk_context().default_bq)
    vk_dispatch_base!(bq, pipeline, arg_buf.address, 0, 0, 0, 1, 1, 1)
end

"""
    _prepare_indirect_dispatch!(indirect_buf, ndrange_buf, workgroup_size)

Compile and dispatch a tiny kernel that reads `ndrange_buf[1]` (a GPU-resident queue size)
and writes `ceil(size / workgroup_size)` to `indirect_buf` (VkDispatchIndirectCommand).
This runs as a single-thread direct dispatch (ndrange=1).
"""
function _prepare_indirect_dispatch!(indirect_buf::VkIndirectBuffer, ndrange_buf::LavaArray{<:Integer}, workgroup_size::Integer)
    bq = something(current_batch_queue(), vk_context().default_bq)
    lava_launch!(bq, _prepare_indirect_kernel,
                 Ptr{UInt32}(indirect_buf.address), ndrange_buf, UInt32(workgroup_size);
                 ndrange=1, workgroup_size=(1, 1, 1))
end

"""
    ka_launch_indirect!(obj, args, ndrange_buf, workgroupsize)

Launch a KA kernel using indirect dispatch. `ndrange_buf` is a GPU array containing
the work item count (1-element Int32 array). The prepare-indirect kernel writes
group counts to an indirect buffer, then vk_dispatch_indirect! dispatches the main kernel.
"""
function ka_launch_indirect!(obj, args, ndrange_buf::LavaArray, workgroupsize, original_args=nothing)
    # Respect static workgroup size from @kernel definition
    ws = if workgroupsize !== nothing
        workgroupsize isa Integer ? (workgroupsize,) : workgroupsize
    elseif KA.workgroupsize(obj) <: KA.StaticSize
        KA.get(KA.workgroupsize(obj))
    else
        (256,)
    end
    ws_3d = pad_to_3d(ws)
    ws_prod = prod(ws)

    # We need to compile the kernel with a static ndrange for __validindex.
    # Use DynamicCheck so the kernel checks bounds at runtime via the iterspace.
    # We compile with a large ndrange; the actual dispatch count comes from indirect buffer.
    # Key insight: __validindex checks `I in __ndrange(ctx)`, and __ndrange(ctx) returns
    # the ndrange from CompilerMetadata. We set this to a large value so threads always pass.
    # The kernel's own bounds check (e.g. `if i <= queue.size[1]`) handles the real bound.
    big_ndrange = (1024 * 1024 * ws_prod,)
    ndrange_tuple = length(ws) == 1 ? big_ndrange :
                    length(ws) == 2 ? (big_ndrange[1], 1) :
                    (big_ndrange[1], 1, 1)
    iterspace, dynamic = KA.partition(obj, ndrange_tuple, ws)
    ctx = KA.mkcontext(obj, ndrange_tuple, iterspace)

    all_args = (obj.f, ctx, args...)

    # Build type tuple for compilation
    tt = Tuple{map(ka_arg_llvm_type, Base.tail(all_args))...}
    compiled, pipeline, offsets, byval_sizes = _get_compiled_kernel_and_pipeline(obj.f, tt, ws_3d)

    # Precompute arg buffer size (allocation deferred until after flush)
    inline_extra = _compute_inline_extra_from_byval(byval_sizes)
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # GC.@preserve ensures original_args (the pre-argconvert LavaArrays) stay alive
    # throughout this function. This is critical because:
    # - argconvert strips LavaArray → Ptr{T}, losing buffer references
    # - vk_flush!() clears batch data_refs and calls maybe_collect()
    # - Without @preserve, GC can free LavaArray buffers whose BDA pointers
    #   are embedded in the arg buffer, causing DEVICE_LOST on NVIDIA
    GC.@preserve original_args begin

    arg_buf = get_arg_buffer(total_size)
    _pack_args_direct!(arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    indirect_buf = _get_indirect_buffer()
    _fast_prepare_indirect!(indirect_buf, ndrange_buf, ws_prod)

    if dispatch_logging_enabled[]
        last_dispatch_info[] = "indirect f=$(_dispatch_name(obj.f, all_args))"
    end
    keep_data_alive!(args)
    if original_args !== nothing
        keep_data_alive!(original_args)
    end
    bq = something(current_batch_queue(), vk_context().default_bq)
    vk_dispatch_indirect!(bq, pipeline, arg_buf.address, indirect_buf)

    end # GC.@preserve

    return nothing
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

# Compute linear 1-based block index from 3D workgroup IDs.
# Uses lava_num_workgroups (GPU builtin = actual dispatch dimensions),
# so this stays correct even when pad_to_3d splits large 1D dispatches.
@inline function linear_block_index()
    bx = Int(lava_workgroup_id_x())
    by = Int(lava_workgroup_id_y())
    bz = Int(lava_workgroup_id_z())
    nx = Int(lava_num_workgroups_x())
    ny = Int(lava_num_workgroups_y())
    return bx + by * nx + bz * nx * ny + 1
end

@lava_device_override @inline function KA.__index_Local_Linear(ctx)
    return Int(lava_local_invocation_id_x()) + 1
end

@lava_device_override @inline function KA.__index_Group_Linear(ctx)
    return linear_block_index()
end

@lava_device_override @inline function KA.__index_Global_Linear(ctx)
    I = @inbounds KA.expand(KA.__iterspace(ctx), linear_block_index(), Int(lava_local_invocation_id_x()) + 1)
    @inbounds LinearIndices(KA.__ndrange(ctx))[I]
end

@lava_device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[Int(lava_local_invocation_id_x()) + 1]
end

@lava_device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[linear_block_index()]
end

@lava_device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx), linear_block_index(), Int(lava_local_invocation_id_x()) + 1)
end

@lava_device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), linear_block_index(), Int(lava_local_invocation_id_x()) + 1)
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
