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
# `@noinline`: callers reach this only through `invokelatest`, so the string
# machinery it touches stays out of the launch path's inference chain.
@noinline function dispatch_name(@nospecialize(f), @nospecialize(all_args))
    fname = nameof(typeof(f))
    # Check if this is a workqueue_map_kernel dispatch — inner func is arg 3
    if length(all_args) >= 3 && occursin("workqueue_map_kernel", string(fname))
        inner = all_args[3]
        return "$(fname)[$(nameof(typeof(inner)))]"
    end
    return string(fname)
end

# ── Backend struct ──

"""
    LavaBackend <: KA.GPU

Lava's GPU compute backend. Carries the Vulkan context and batch queues explicitly.

  * `dispatch_bq`: where KA kernel dispatches are recorded.
  * `upload_bq`:   where CPU→GPU transfers (upload!, download!, staging)
                   record their copies. May be the same queue as dispatch_bq
                   (single-queue mode) or a separate async queue for true
                   upload/compute overlap.

    LavaBackend()                      # default: dispatch + upload both on default_bq
    LavaBackend(bq)                    # single queue for both
    LavaBackend(dispatch_bq, upload_bq)  # split — enables pipelining

`LavaBackend()` with no arguments resolves `dispatch_bq` / `upload_bq`
lazily via `vk_context().default_bq` at every property access.  Pinning
would break after `vk_reset_device!()`: a `const BACKEND = LavaBackend()`
created at module-load would keep a stale `BatchQueue` tied to the old
`VkDevice`, and every subsequent buffer created via that backend would end
up allocated on the dead device — later triggering
`VUID-vkCmdCopyBuffer-commonparent` and a page-aligned GPUVM fault.
Explicit queues passed to `LavaBackend(bq)` / `LavaBackend(d, u)` are
pinned on purpose (the caller wants those exact queues, e.g. an async
upload queue).
"""
struct LavaBackend <: KA.GPU
    # nothing = "use vk_context().default_bq at each access" (survives resets)
    # non-nothing = caller pinned this specific queue
    dispatch_bq::Union{BatchQueue, Nothing}
    upload_bq::Union{BatchQueue, Nothing}
end

LavaBackend() = LavaBackend(nothing, nothing)
LavaBackend(ctx::VkContext) = (let bq = ctx.default_bq; LavaBackend(bq, bq); end)
LavaBackend(bq::BatchQueue) = LavaBackend(bq, bq)

# Property access resolves a `nothing`-pinned queue through the live
# `vk_context()` so a module-level `const BACKEND = LavaBackend()` keeps
# working across `vk_reset_device!()`. `:bq` stays a back-compat alias for
# `:dispatch_bq`.
function Base.getproperty(b::LavaBackend, s::Symbol)
    if s === :dispatch_bq
        f = getfield(b, :dispatch_bq)
        return f === nothing ? vk_context().default_bq : f
    elseif s === :upload_bq
        f = getfield(b, :upload_bq)
        return f === nothing ? vk_context().default_bq : f
    elseif s === :bq
        return getproperty(b, :dispatch_bq)
    end
    return getfield(b, s)
end

# ── Backend queries ──

# Derive the backend from the array's own context so cross-context arrays
# dispatch against the correct queue instead of the global default.
KA.get_backend(a::LavaArray) = LavaBackend((a.buf[].ctx)::VkContext)
# KA.synchronize submits all recorded dispatches and waits for GPU completion.
# This matches CUDA/AMDGPU semantics: after synchronize(), the CPU can safely
# read GPU results. GPU-side ordering between dispatches is handled by pipeline
# barriers in record_dispatch!, so synchronize() is only needed when the CPU
# must observe GPU results (or at natural batch boundaries like end-of-sample).
function KA.synchronize(backend::LavaBackend)
    flush!(backend.dispatch_bq, backend.dispatch_bq.device)
    # If split, the upload queue may have pending transfers worth flushing too.
    if backend.upload_bq !== backend.dispatch_bq
        flush!(backend.upload_bq, backend.upload_bq.device)
    end
    return
end
KA.supports_unified(::LavaBackend) = true
function KA.allocate(backend::LavaBackend, ::Type{T}, dims::Tuple; unified::Bool=false) where T
    bq = backend.dispatch_bq
    nbytes = prod(dims) * sizeof(T)
    # Auto-unified for tiny allocations (≤ 64 bytes, e.g. WorkQueue.size
    # counters) — BAR memory enables direct CPU readback without staging.
    want_unified = unified || nbytes <= 64
    return LavaArray{T,length(dims)}(undef, Int.(dims); bq, unified=want_unified)
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
        # Default workgroup size: 64 for 1D, capped to ndrange.
        #
        # `Val(length(ndrange))`, not a plain `Int`: the dynamic form goes through
        # `Base._ntuple`, whose body is `1:n` with `n` unknown, so it records an
        # edge on `Tuple{Colon, Int64, Any}`. Any package that adds a `(:)` method
        # — `Unitful` adds `Colon(::Any, ::Quantity)` — then makes that edge stale,
        # and every precompiled CodeInstance that inferred through it is thrown
        # away when the image loads. That single edge accounts for 25 025 of
        # Lava's 36 245 rejected CodeInstances when SAM 2's image loads after
        # VideoEditor. `Val` makes the tuple length static and the range vanishes.
        workgroupsize = ntuple(
            i -> i == 1 ? min(prod(ndrange), 64) : 1,
            Val(length(ndrange)))
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
# LavaArray → LavaDeviceArray (Ptr-wrapping). The `LavaAdaptor(batch)` in this
# conversion path is the single place a LavaArray gets stripped to a pointer,
# and it pins the original LavaArray into `batch` at the same point.  There is
# no separate pinning walker — Adapt.jl's own recursion over wrapper structs /
# Broadcasted / NamedTuple lands every LavaArray at `adapt_storage`, and that
# method does both the strip and the pin.
#
# `obj.f` is the kernel closure — it goes through the same adaptor so any
# LavaArray captured in closure-over fields is pinned too.
#
# `KA.argconvert(kernel, arg)` is a pure strip used by callers (Raycore's
# MultiTypeSet, etc.) to cache a device-side view of a LavaArray into their
# own CPU-side structs *outside* any kernel dispatch.  No pin here — the
# caller is responsible for keeping the original LavaArray alive until the
# cached device form is used, and the subsequent kernel dispatch pins that
# original LavaArray via `LavaAdaptor`.
KA.argconvert(::KA.Kernel{LavaBackend}, a::LavaArray{T,N}) where {T,N} =
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
KA.argconvert(::KA.Kernel{LavaBackend}, x) = x

# ── Kernel call (main entry point) ──
#
# `find_tlas_in_args` scans the kernel's pre-adapt arg tuple for any value
# that holds a live HWTLAS reference, so callers can write
#   `kernel(accel, args...; ndrange=...)`
# with `accel::HWAdaptedAccel` (or any value carrying one) and have
# `enable_ray_query=true` + the TLAS descriptor binding wired automatically
# at compile + dispatch.  This keeps the SW/HW swap a single-arg change at
# the call site — Hikari's `Raycore.closest_hit(accel, ray)` polymorphism
# falls through unchanged.
#
# Whether an argument *can* carry a TLAS is a property of its type, so the scan
# is resolved at compile time: a compute kernel — every kernel in a DNN or
# broadcast workload — gets `nothing` and no code at all. The recursive
# `Base.tail` walk it replaces did not fold away on the long argument tuples KA
# produces (kernel args plus a `CompilerMetadata` plus a `Val` per static
# parameter), and showed up as the second-largest host cost in the MatAnyone
# inference loop, on kernels that have no TLAS and never could.
#
# Later arguments win, matching the accumulate-forward order of the walk.
@generated function find_tlas_in_args(args::Tuple)
    ts = fieldtypes(args)
    all(isconcretetype, ts) || return :(_find_tlas(args, nothing))
    expr = :(nothing)
    for (i, T) in enumerate(ts)
        hasfield(T, :hwtlas) || continue
        expr = :(let h = getfield(args[$i], :hwtlas)
                     h === nothing ? $expr : h
                 end)
    end
    expr
end
@inline _find_tlas(::Tuple{}, acc) = acc
@inline _find_tlas(args::Tuple, acc) = _find_tlas(Base.tail(args), _maybe_tlas(first(args), acc))
# `HWAdaptedAccel` is declared in raytracing/hwtlas.jl (loaded after this file);
# loose-typed dispatch + a runtime field check keeps the include order intact.
@inline function _maybe_tlas(x, acc)
    if hasfield(typeof(x), :hwtlas)
        h = getfield(x, :hwtlas)
        h !== nothing && return h
    end
    return acc
end

# Cached per-kernel iteration plan. Key = (typeof(obj), ndrange, workgroupsize).
# The plan only depends on those — `typeof(obj)` carries the F type plus KA's
# static NDRange / WG-size type parameters, so two Kernel instances with the same
# shape share the plan. Built lazily on first launch.
struct _IterPlan{Ctx}
    ka_ctx::Ctx
    block_dims::NTuple{3, Int}
    ws_3d::NTuple{3, Int}
    nblocks::Int
end
const KERNEL_ITER_PLAN_CACHE = Dict{Any, _IterPlan}()

function get_or_build_iter_plan(obj::KA.Kernel{LavaBackend}, ndrange, workgroupsize,
                                vkctx::VkContext)
    key = (typeof(obj), ndrange, workgroupsize)
    plan = get(KERNEL_ITER_PLAN_CACHE, key, nothing)
    plan === nothing || return plan::_IterPlan

    ndr_canon, _ws, iterspace, _dyn = KA.launch_config(obj, ndrange, workgroupsize)
    ka_ctx  = KA.mkcontext(obj, ndr_canon, iterspace)
    blocks  = KA.blocks(iterspace)
    nblocks = length(blocks)
    if nblocks == 0
        new_plan = _IterPlan(ka_ctx, (0, 0, 0), (0, 0, 0), 0)
    else
        block_dims = pad_to_3d(vkctx, size(blocks))
        nthreads   = length(KA.workitems(iterspace))
        ws_3d      = (nthreads, 1, 1)
        new_plan   = _IterPlan(ka_ctx, block_dims, ws_3d, nblocks)
    end
    KERNEL_ITER_PLAN_CACHE[key] = new_plan
    return new_plan
end

"""
    BARRIER_ELISION[] :: Bool

Drop the barrier in front of a dispatch whose buffers are disjoint from every
buffer touched since the last barrier.

Sound without read/write annotations: disjoint memory cannot alias, so it does
not matter which side reads and which writes. Unsound only if a kernel reaches
device memory that is not in its argument tree — Lava kernels address memory
through `LavaDeviceArray` BDAs derived from exactly that tree.

Off by default because the check is O(ranges since last barrier) *per dispatch*,
which is real host cost. It is meant to be turned on around `Lava.capture`: the
analysis is then paid once and every `replay!` gets the shorter command buffer
for free. On the MatAnyone step barriers are 3.35 ms of an 11.07 ms replay.
"""
const BARRIER_ELISION = Ref{Bool}(false)

# Flat [lo₁,hi₁,lo₂,hi₂,…] of everything touched since the last barrier, and a
# scratch list for the dispatch being recorded. Reused, never reallocated.
const TOUCHED_RANGES = UInt64[]
const DISPATCH_RANGES = UInt64[]

push!(RESET_CALLBACKS, function()
    empty!(TOUCHED_RANGES)
    empty!(DISPATCH_RANGES)
end)

"""Reset the elision state — call whenever the queue is known to be drained."""
@inline reset_barrier_elision!() = (empty!(TOUCHED_RANGES); nothing)

"""
    poison_barrier_elision!()

Record that something touched memory we cannot enumerate, so nothing may be
elided until a barrier clears it.

Only the KA launch path knows a dispatch's buffers. Everything else that records
into the same command buffer — `cmd_copy_buffer!`, `lava_launch!` used directly
by Lava's own internals, indirect prepares — writes memory this tracker never
sees. Left alone that is not conservative but *wrong*: a later dispatch reading
what a copy just wrote finds no overlap in the tracker and drops the barrier it
needed. It cost 0.024 of alpha on the MatAnyone step, which is the sort of small
plausible error a race gives you.

Modelled as one range covering the whole address space: every subsequent
dispatch overlaps it, takes its barrier, and clears it — no special cases.
"""
@inline function poison_barrier_elision!()
    empty!(TOUCHED_RANGES)
    push!(TOUCHED_RANGES, UInt64(0))
    push!(TOUCHED_RANGES, typemax(UInt64))
    nothing
end

"""Set by the KA launch path for the dispatch it is about to record."""
const RANGES_DECLARED = Ref{Bool}(false)

"""
True when `new` overlaps anything in `TOUCHED_RANGES`. On overlap the caller
emits a barrier and `TOUCHED_RANGES` restarts from `new`; otherwise `new` is
merged in and the barrier is skipped.
"""
function barrier_needed!(new::Vector{UInt64})
    hit = false
    @inbounds for i in 1:2:length(new)
        lo, hi = new[i], new[i+1]
        for j in 1:2:length(TOUCHED_RANGES)
            if lo < TOUCHED_RANGES[j+1] && TOUCHED_RANGES[j] < hi
                hit = true
                break
            end
        end
        hit && break
    end
    # The set only clears when a barrier fires, so a long run of elided
    # dispatches would make this scan quadratic. Past the cap, force a barrier
    # and start over — conservative, so it can only cost performance.
    if hit || length(TOUCHED_RANGES) > 512
        empty!(TOUCHED_RANGES)
        hit = true
    end
    append!(TOUCHED_RANGES, new)
    hit
end

"""
    interior_unit_workgroup(W) -> Bool

Whether the static workgroup `W` has an extent of 1 in an *interior* dimension —
anything but the first or the last.

That is the exact trigger for a Lava codegen fault: with such a workgroup baked
into the kernel's TYPE, the block index decode takes the wrong divisor for
dimension 2 and only `min(1, blocks[3] / blocks[2])` of the output is written,
silently. A *trailing* unit extent is harmless, which is why ranks 1-3 look
clean — `(32, 4, 1)`'s unit extent is the last one.

Established as Lava's and not KernelAbstractions': the identical kernel,
workgroup and ndrange are correct on `KA.CPU()`, which runs the same `expand`
and `NDRange` machinery with no SPIR-V emitter anywhere. `test_static_workgroup.jl`
pins the trigger, the law, and the interior-versus-trailing rule.
"""
const WORKGROUP_FALLBACK = Ref(true)

@inline function interior_unit_workgroup(W::NTuple{N,Int}) where {N}
    N >= 3 || return false
    @inbounds for d in 2:(N - 1)
        W[d] == 1 && return true
    end
    false
end
interior_unit_workgroup(::Any) = false

"""
Re-launch a kernel whose workgroup lives in its TYPE as one that takes the size
at the call, which is the path that computes correct indices.

Set `WORKGROUP_FALLBACK[] = false` to take the raw path anyway — only
`test_static_workgroup.jl` does that, to check the underlying fault is still
present and this guard is still load-bearing.

Only for shapes `interior_unit_workgroup` rejects, so this cannot change any
launch that is currently right — it turns a silently wrong answer into a correct
one. It is slower (the index arithmetic stops being compile-time constant, ~2x on
the kernels measured), and that is the trade until the codegen fault is fixed.
"""
@inline function relaunch_dynamic(obj::KA.Kernel{LavaBackend, KA.NDIteration.StaticSize{W}, ND, F},
                                  args, ndrange) where {W, ND, F}
    dyn = KA.Kernel{LavaBackend, KA.NDIteration.DynamicSize, ND, F}(obj.backend, obj.f)
    dyn(args...; ndrange = ndrange, workgroupsize = W)
end

function (obj::KA.Kernel{LavaBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    validate_launch_args(args)
    # A static workgroup with an interior unit extent miscompiles; take the
    # dynamic path instead. Checked on the TYPE, so it folds away for every
    # kernel that is not affected.
    if WORKGROUP_FALLBACK[] && workgroupsize === nothing && ndrange isa Tuple
        WGT = typeof(obj).parameters[2]
        if WGT <: KA.NDIteration.StaticSize && interior_unit_workgroup(WGT.parameters[1])
            return relaunch_dynamic(obj, args, ndrange)
        end
    end
    bq = obj.backend.bq

    # Auto-discover TLAS for ray-query kernels — extract BEFORE Adapt strips
    # hwtlas (kernel form has hwtlas=nothing).
    tlas = find_tlas_in_args(args)

    # GPU-resident ndrange → indirect dispatch (no CPU readback)
    if ndrange isa LavaArray
        batch = ensure_active_batch!(bq)
        pin_leaves!(batch, obj.f)
        pin_leaves!(batch, args)
        adaptor = LavaAdaptor(batch)
        converted_args = map(a -> Adapt.adapt(adaptor, a), args)
        ka_launch_indirect!(obj, converted_args, ndrange, workgroupsize, args, adaptor, bq, tlas)
        return nothing
    end

    # KA's launch_config / partition / mkcontext / blocks / workitems plus our
    # pad_to_3d add up to ~50% of per-record cost in tight loops, yet they only
    # depend on (typeof(obj), ndrange, workgroupsize) — typeof(obj) carries the
    # static NDRange/WG-size and F type parameters. Cache the whole plan so the
    # second-and-later launch with the same shape is one Dict lookup.
    plan = get_or_build_iter_plan(obj, ndrange, workgroupsize, bq.ctx::VkContext)
    plan.nblocks == 0 && return nothing
    ka_ctx     = plan.ka_ctx
    block_dims = plan.block_dims
    ws_3d      = plan.ws_3d

    batch = ensure_active_batch!(bq)
    # Side-effect pass: pin every LavaArray leaf in the closure + args into
    # `batch.pinned`, once, via @generated walker (zero alloc, straight-line
    # code).  `Adapt.adapt` below is now pure — it only strips.
    pin_leaves!(batch, obj.f)
    pin_leaves!(batch, args)
    # Same tree the pins just walked, so every buffer this dispatch can reach is
    # covered. Decided here rather than in `record_dispatch!` because this is the
    # last point that still has the pre-adapt arguments.
    if BARRIER_ELISION[]
        empty!(DISPATCH_RANGES)
        range_leaves!(DISPATCH_RANGES, obj.f)
        range_leaves!(DISPATCH_RANGES, args)
        NEXT_SKIP_BARRIER[] = !barrier_needed!(DISPATCH_RANGES)
        RANGES_DECLARED[] = true
    end
    adaptor = LavaAdaptor(batch)
    converted_f = Adapt.adapt(adaptor, obj.f)
    converted_args = map(a -> Adapt.adapt(adaptor, a), args)
    all_args = (converted_f, ka_ctx, converted_args...)

    ka_launch!(bq, converted_f, all_args, block_dims, ws_3d, tlas)

    return nothing
end

# Vulkan dispatch is max 3D. pad_to_3d maps the N-D block grid to a 3D dispatch,
# splitting large dimensions across Y/Z when needed to stay within device limits.
# Uses actual device maxComputeWorkGroupCount (queried once, cached).
#
# CRITICAL: pad_to_3d must NEVER over-dispatch (produce more workgroups than requested).
# Kernels like AK._accumulate_block! write to auxiliary arrays indexed by workgroup ID
# without bounds checks — phantom workgroups cause out-of-bounds GPU memory writes.

# Per-dimension workgroup count limits live on `VkContext.max_wg_dims` — queried
# once at device creation. `pad_to_3d` and friends reach them via `bq.ctx`.

# Find exact factor of n that is ≤ max_dim, for splitting workgroup counts.
# Returns the largest factor ≤ max_dim, or max_dim if none found (over-dispatch).
function find_split_factor(n::Int, max_dim::Int)
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

function pad_to_3d(ctx::VkContext, t::NTuple{1,<:Integer})
    n = Int(t[1])
    max_x, max_y, _ = ctx.max_wg_dims
    n <= max_x && return (n, 1, 1)
    # Need to split into X * Y (or X * Y * Z)
    # Find X such that X divides n exactly and X ≤ max_x
    x = find_split_factor(n, max_x)
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
function pad_to_3d(ctx::VkContext, t::NTuple{2,<:Integer})
    x, y = Int(t[1]), Int(t[2])
    max_x, max_y, _ = ctx.max_wg_dims
    if x <= max_x && y <= max_y
        return (x, y, 1)
    end
    # Flatten and re-split
    return pad_to_3d(ctx, (x * y,))
end
function pad_to_3d(::VkContext, t::NTuple{3,<:Integer})
    (Int(t[1]), Int(t[2]), Int(t[3]))
end
# For N>3, flatten then split
function pad_to_3d(ctx::VkContext, t::NTuple{N,<:Integer}) where N
    pad_to_3d(ctx, (Int(prod(t)),))
end

"""
Internal launch function for KA kernels. Compiles and dispatches the GPU function.
"""
const DBG_LAUNCH_COUNT = Ref(0)

"""
Everything a dispatch needs that depends only on the *types* of its arguments.

`ka_launch!` used to rebuild `Tuple{map(arg_sigtype, tail(all_args))...}` on
every single dispatch and hand it to `GPUCompiler.methodinstance`: that interns
a fresh `Type` object, does a method lookup, and then two hash lookups keyed on
that type and a freshly built compiler config — all to rediscover a pipeline it
had already compiled. Types hash and compare slowly, and at ~2000 dispatches per
MatAnyone inference step this was the largest single host cost in the loop.

`typeof(all_args)` is available for free and types are interned, so an `IdDict`
keyed on it is a pointer hash. Everything downstream — pipeline, arg layout,
buffer size — is a function of exactly that, so it is all cached together.

The world counter is stored with the entry and checked on each hit. It moves on
any method definition, so redefining a kernel (Revise, or a first-time
specialisation) drops back to the slow path for one call and re-caches; a stale
pipeline can never be served.
"""
struct LaunchPlan
    compiled::LavaGPUKernel
    pipeline::LavaComputePipeline
    offsets::Vector{Int}
    byval_sizes::Vector{Int}
    arg_buffer_size::Int
    total_size::Int
    world::UInt64
    wg::NTuple{3,Int}
    ray_query::Bool
end

const LAUNCH_PLAN_CACHE = IdDict{DataType,Vector{LaunchPlan}}()
push!(RESET_CALLBACKS, () -> empty!(LAUNCH_PLAN_CACHE))

@inline function launch_plan(bq::BatchQueue, @nospecialize(f), all_args::Tuple,
                             wg::NTuple{3,Int}, ray_query::Bool)
    key = typeof(all_args)
    world = Base.get_world_counter()
    v = get(LAUNCH_PLAN_CACHE, key, nothing)
    if v !== nothing
        for p in v
            (p.world === world && p.wg === wg && p.ray_query === ray_query) && return p
        end
    end
    build_launch_plan!(bq, f, all_args, wg, ray_query, key, world)
end

@noinline function build_launch_plan!(bq::BatchQueue, @nospecialize(f), all_args::Tuple,
                                      wg::NTuple{3,Int}, ray_query::Bool,
                                      key::DataType, world::UInt64)
    # Excludes f — GPUCompiler prepends typeof(f). all_args are already
    # post-adapt (LavaDeviceArray, not Ptr{T}).
    tt = Tuple{map(arg_sigtype, Base.tail(all_args))...}
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        bq.ctx::VkContext, f, tt, wg; enable_ray_query=ray_query)
    base = compiled.push_info.arg_buffer_size
    p = LaunchPlan(compiled, pipeline, offsets, byval_sizes, base,
                   base + compute_inline_extra_from_byval(byval_sizes),
                   world, wg, ray_query)
    v = get!(() -> LaunchPlan[], LAUNCH_PLAN_CACHE, key)
    filter!(q -> q.world === world, v)   # entries from a superseded world are dead
    push!(v, p)
    p
end

function ka_launch!(bq::BatchQueue, @nospecialize(f), all_args::Tuple,
                    block_dims::NTuple{3,Int}, workgroup_size::NTuple{3,Int},
                    tlas=nothing)  # positional, Nothing default — hot path
    DBG_LAUNCH_COUNT[] += 1

    # When `tlas` was auto-discovered from kernel args (e.g. an HWAdaptedAccel
    # was passed), enable ray_query so the SPIR-V emitter binds the TLAS
    # descriptor and accepts OpRayQueryInitializeKHR / Proceed / Get*KHR.
    plan = launch_plan(bq, f, all_args, workgroup_size, tlas !== nothing)

    # Get host-visible mapped arg buffer (per-BQ slab pool)
    arg_buf = get_arg_buffer(bq, plan.total_size)

    # The KA.Kernel entry point already ran `Adapt.adapt(LavaAdaptor(batch), ..)`
    # on each original arg, which both pinned every LavaArray (and nested
    # LavaArrays in wrapper structs) AND stripped them to LavaDeviceArray.
    # Here `all_args` is post-adapt, so pack sees no further pinnable leaves.
    ensure_active_batch!(bq)
    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, plan.offsets,
                       plan.arg_buffer_size, plan.byval_sizes, all_args)

    # Dispatch with N-D block grid (preserves KA's block dimensions)
    if DISPATCH_LOGGING_ENABLED[]
        LAST_DISPATCH_INFO[] = Base.invokelatest(dispatch_log_string, "ka f=",
                                   dispatch_name(f, all_args), " groups=", block_dims)::String
    end
    vk_dispatch!(bq, plan.pipeline, arg_buf.address, block_dims, tlas)

    return nothing
end

# ── Indirect dispatch support ──
#
# Internal Lava kernels only ever see `LavaDeviceArray` / `LavaDeviceArray`
# — never `Ptr{T}` / `unsafe_load`.  Callers wrap raw GPU memory in
# `LavaArray`/`LavaArray view` before handing it to the kernel.

function prepare_indirect_kernel(indirect::LavaDeviceArray{UInt32,1},
                                  ndrange_buf::LavaDeviceArray{Int32,1},
                                  ws::UInt32)
    n = UInt32(ndrange_buf[1])
    groups = (n + ws - UInt32(1)) ÷ ws
    indirect[1] = groups         # groupCountX
    indirect[2] = UInt32(1)      # groupCountY
    indirect[3] = UInt32(1)      # groupCountZ
    return nothing
end

# ── Fast prepare-indirect path ──
# Bypasses full lava_launch! to avoid per-dispatch ceremony overhead.
# The prepare-indirect kernel is always the same function with the same types,
# so we compile once and cache everything. Saves ~30K lava_launch! calls per render
# (hash lookups, validation, auto-flush, keep_alive, etc.).

const PREPARE_INDIRECT_PIPELINE_REF = Ref{Union{Nothing, LavaComputePipeline}}(nothing)
const PREPARE_INDIRECT_OFFSETS_REF = Ref{Union{Nothing, Vector{Int}}}(nothing)
const PREPARE_INDIRECT_BYVAL_REF = Ref{Union{Nothing, Vector{Int}}}(nothing)
const PREPARE_INDIRECT_ARG_BUF_SIZE_REF = Ref{Int}(0)

# Register cleanup callback for vk_reset_device!
push!(RESET_CALLBACKS, function()
    PREPARE_INDIRECT_PIPELINE_REF[] = nothing
    PREPARE_INDIRECT_OFFSETS_REF[] = nothing
    PREPARE_INDIRECT_BYVAL_REF[] = nothing
    PREPARE_INDIRECT_ARG_BUF_SIZE_REF[] = 0
end)

function init_prepare_indirect_pipeline!(ctx::VkContext)
    PREPARE_INDIRECT_PIPELINE_REF[] !== nothing && return
    # Kernel signature (post-adapt): indirect::LavaDeviceArray{UInt32,1},
    # ndrange_buf::LavaDeviceArray{Int32,1}, ws::UInt32.
    tt = Tuple{LavaDeviceArray{UInt32,1}, LavaDeviceArray{Int32,1}, UInt32}
    ws = (1, 1, 1)
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        ctx, prepare_indirect_kernel, tt, ws)
    PREPARE_INDIRECT_PIPELINE_REF[] = pipeline
    PREPARE_INDIRECT_OFFSETS_REF[] = offsets
    PREPARE_INDIRECT_BYVAL_REF[] = byval_sizes
    PREPARE_INDIRECT_ARG_BUF_SIZE_REF[] = compiled.push_info.arg_buffer_size
end

"""
    fast_prepare_indirect!(bq, indirect::LavaArray{UInt32,1}, ndrange_buf::LavaArray{<:Integer}, workgroup_size)

Fast path for prepare-indirect dispatch.  Bypasses lava_launch!'s validation/
logging overhead: manually adapts the two LavaArrays (pin + strip) and
packs directly.
"""
function fast_prepare_indirect!(bq::BatchQueue,
                                indirect::LavaArray{UInt32,1},
                                ndrange_buf::LavaArray{<:Integer},
                                workgroup_size::Integer)
    init_prepare_indirect_pipeline!(bq.ctx::VkContext)

    pipeline = PREPARE_INDIRECT_PIPELINE_REF[]
    offsets = PREPARE_INDIRECT_OFFSETS_REF[]
    byval_sizes = PREPARE_INDIRECT_BYVAL_REF[]
    arg_size = PREPARE_INDIRECT_ARG_BUF_SIZE_REF[]

    batch = ensure_active_batch!(bq)
    adaptor = LavaAdaptor(batch)
    dev_indirect = Adapt.adapt(adaptor, indirect)::LavaDeviceArray{UInt32,1}
    dev_ndrange  = Adapt.adapt(adaptor, ndrange_buf)

    # f is a ghost singleton (prepare_indirect_kernel), pack_args_direct! skips it.
    all_args = (prepare_indirect_kernel, dev_indirect, dev_ndrange, UInt32(workgroup_size))
    arg_buf = get_arg_buffer(bq, arg_size)
    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets, arg_size, byval_sizes, all_args)

    vk_dispatch_base!(bq, pipeline, arg_buf.address, 0, 0, 0, 1, 1, 1)
end

function prepare_indirect_dispatch!(bq::BatchQueue,
                                    indirect::LavaArray{UInt32,1},
                                    ndrange_buf::LavaArray{<:Integer},
                                    workgroup_size::Integer)
    lava_launch!(bq, prepare_indirect_kernel,
                 indirect, ndrange_buf, UInt32(workgroup_size);
                 ndrange=1, workgroup_size=(1, 1, 1))
end

"""
    ka_launch_indirect!(obj, args, ndrange_buf, workgroupsize)

Launch a KA kernel using indirect dispatch. `ndrange_buf` is a GPU array containing
the work item count (1-element Int32 array). The prepare-indirect kernel writes
group counts to an indirect buffer, then vk_dispatch_indirect! dispatches the main kernel.
"""
function ka_launch_indirect!(obj, args, ndrange_buf::LavaArray, workgroupsize, original_args,
                             adaptor::LavaAdaptor,
                             bq::BatchQueue=obj.backend.bq,
                             tlas=nothing)  # positional — same NamedTuple-avoidance as ka_launch!
    # Respect static workgroup size from @kernel definition
    ws = if workgroupsize !== nothing
        workgroupsize isa Integer ? (workgroupsize,) : workgroupsize
    elseif KA.workgroupsize(obj) <: KA.StaticSize
        KA.get(KA.workgroupsize(obj))
    else
        (256,)
    end
    ws_3d = pad_to_3d(bq.ctx::VkContext, ws)
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

    # Caller already pinned the original args via `pin_leaves!`; pin the
    # closure's captures here. `Adapt.adapt` is pure now (strip only).
    batch = adaptor.batch
    pin_leaves!(batch, obj.f)
    converted_f = Adapt.adapt(adaptor, obj.f)
    all_args = (converted_f, ctx, args...)

    # Kernel ABI is post-adapt: LavaDeviceArray etc.
    tt = Tuple{map(arg_sigtype, Base.tail(all_args))...}
    enable_ray_query = tlas !== nothing
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        bq.ctx::VkContext, converted_f, tt, ws_3d; enable_ray_query)

    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # GC.@preserve on original_args keeps the pre-strip LavaArrays reachable
    # through this function.
    GC.@preserve original_args begin

    arg_buf = get_arg_buffer(bq, total_size)
    @assert batch === bq.active_batch  "adaptor batch diverged from active bq batch"
    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    indirect_view = get_indirect_buffer(bq)

    if DISPATCH_LOGGING_ENABLED[]
        LAST_DISPATCH_INFO[] = Base.invokelatest(dispatch_log_string, "indirect f=",
                                   dispatch_name(obj.f, all_args))::String
    end
    deferred = DEFERRED_INDIRECT[]
    if deferred !== nothing
        # Inside `concurrent_indirect_group`: record NOTHING here — both the
        # prepare (fused into one multi-prepare dispatch) and the indirect
        # dispatch happen at the group's flush, so all pairs in the group
        # share two barriers total and the dispatches overlap on the GPU.
        # The packed args + indirect slot stay valid across the gap: both
        # live in this batch's slabs and the pool-reset guard keeps the
        # cursors from rewinding while the batch holds recorded dispatches.
        push!(deferred, (bq, pipeline, arg_buf.address, indirect_view, tlas,
                         ndrange_buf, ws_prod))
    else
        fast_prepare_indirect!(bq, indirect_view, ndrange_buf, ws_prod)
        vk_dispatch_indirect!(bq, pipeline, arg_buf.address, indirect_view, tlas)
    end

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

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T}, v::T, i::Integer) where T
    unsafe_store!(a.ptr, v, i)
    return v
end
@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T}, v, i::Integer) where T
    unsafe_store!(a.ptr, convert(T, v), i)
    return v
end

# CartesianIndex indexing for multi-dimensional broadcast support (any N)
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,N}, I::CartesianIndex{N}) where {T,N}
    @inbounds unsafe_load(a.ptr, linear_index(a.dims, I))
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
    @inbounds unsafe_store!(a.ptr, convert(T, v), linear_index(a.dims, I))
    return v
end

# Multi-index `a[i, j, k, …]` must route through the same Horner `linear_index`
# as the CartesianIndex path above. Without these, Base's generic fallback
# (`_to_linear_index` → `_sub2ind`) builds precisely the naive nested-product
# expansion documented below, and the NVIDIA shader compiler miscompiles it into
# an out-of-bounds PhysicalStorageBuffer offset.
#
# It only misfires once the indices are *computed* rather than literal, so it
# hides from simple tests: `a[1,2,2,1]` constant-folds and is fine, while
# `a[is[1], is[2], 2, 1]` with `is` from a delinearised loop index reads garbage.
# That is how it reached GPUArrays' `vectorized_getindex` (host/indexing.jl:84),
# silently truncating every strided `a[:, :, 2:2, :]`.
@lava_device_override @inline function Base.getindex(a::LavaDeviceArray{T,N},
                                                     i1::Integer, i2::Integer,
                                                     Ir::Vararg{Integer}) where {T,N}
    @inbounds unsafe_load(a.ptr, linear_index(a.dims, CartesianIndex(i1, i2, Ir...)))
end

@lava_device_override @inline function Base.setindex!(a::LavaDeviceArray{T,N}, v,
                                                      i1::Integer, i2::Integer,
                                                      Ir::Vararg{Integer}) where {T,N}
    @inbounds unsafe_store!(a.ptr, convert(T, v),
                            linear_index(a.dims, CartesianIndex(i1, i2, Ir...)))
    return v
end

# Convert CartesianIndex to linear index for LavaDeviceArray
@inline function linear_index(dims::NTuple{1,Int}, I::CartesianIndex{1})
    I[1]
end
@inline function linear_index(dims::NTuple{2,Int}, I::CartesianIndex{2})
    I[1] + dims[1] * (I[2] - 1)
end
# Horner form. The naive column-major expansion
#   I[1] + dims[1]*(I[2]-1) + dims[1]*dims[2]*(I[3]-1) + …
# forms a standalone `dims[1]*dims[2]` product (and shares `dims[1]` across two
# terms). NVIDIA's shader compiler miscompiles exactly that shape when the result
# feeds a PhysicalStorageBuffer load offset: it drops the I[1] term, so every read
# lands on source row 1. Horner factoring keeps each stride coefficient as a single
# value applied to a running sum, never materialising the nested product, which the
# driver evaluates correctly. Symptom was `repeat(x; inner)` with a 3-D `inner`
# (GPUArrays repeat_inner_dst_kernel! → xs[CartesianIndex(sdx)]). Pinned by
# test_repeat_inner_3d.jl. Same NVIDIA complex-integer family as the unswitch and
# shared-PSB-access-chain miscompiles.
@inline function linear_index(dims::NTuple{3,Int}, I::CartesianIndex{3})
    I[1] + dims[1] * ((I[2] - 1) + dims[2] * (I[3] - 1))
end
@inline function linear_index(dims::NTuple{N,Int}, I::CartesianIndex{N}) where N
    off = I[N] - 1
    @inbounds for d in (N-1):-1:1
        off = off * dims[d] + (I[d] - 1)
    end
    return off + 1
end

# Ptr{T} indexing overrides are intentionally absent — device kernels never
# receive raw Ptr{T} args any more.  All array-like kernel parameters come
# through LavaAdaptor as LavaDeviceArray and use the overrides above.

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

# ── Synchronization ──

@lava_device_override @inline function KA.__synchronize()
    lava_workgroup_barrier()
end

# ── @private: per-thread scratch memory via stack-allocated MArray ──

# Identical to CUDA.jl's own definition (CUDAKernels.jl:205). `@private` was
# once suspected of being ~8x slower here than plain locals; that was a
# measurement-order artifact — whichever kernel is timed first pays a one-time
# cost. Interleaved, the two are the same speed, and post-SROA IR is identical
# (66 instructions, 11 loads, 1 store, 5 phis in both). Nothing to fix here.
@lava_device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

# KA.__print is now overridden in device/printf.jl (routes @print → DebugPrintf).
