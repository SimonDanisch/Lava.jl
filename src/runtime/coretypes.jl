"""
The types `VkContext` has to name, hoisted ahead of it.

Nothing here is new and nothing here has behaviour — these are the `struct`
blocks that used to sit beside the functions that use them, moved so that
`VkContext` can hold its per-device state in **concrete typed fields** instead of
module-level dictionaries keyed by `ctx.id`.

That keying was the shortcut: it made two devices work without touching call
sites, and the probe that followed found seven real defects. But it left twelve
globals whose entries outlive the device they describe, and a `RESET_CALLBACKS`
mechanism whose entire job was emptying them. State owned by the context needs
neither.

Only the definitions moved. Every constructor, method and comment about
*behaviour* stayed where it was, because include order constrains types and not
methods — Julia resolves a call at the first invocation, long after every file
is loaded.

Two `Any` fields survive, and both predate this: `PoolBlock.pool` (a `DevicePool`
holds a `Vector{PoolBlock}`, so one direction must be untyped) and
`VkManagedBuffer.ctx`/`last_write` (a `VkContext` is what owns the pool that owns
the buffer). Those are genuine cycles; the rest were only ordering.
"""

"""
    LavaComputePipeline

A compiled compute pipeline ready for dispatch.
"""
struct LavaComputePipeline
    shader_module::Vulkan.ShaderModule
    pipeline_layout::Vulkan.PipelineLayout
    pipeline::Vulkan.Pipeline
    push_constant_size::UInt32
    needs_tlas_descriptor::Bool
    descriptor_set_layout::Union{Nothing, Vulkan.DescriptorSetLayout}
end

struct SubgroupSizeControl
    min::Int
    max::Int
    compute::Bool   # COMPUTE present in requiredSubgroupSizeStages
end

"""
    PoolBlock

One 64 MiB slab carved by bump pointer, and the count of live sub-allocations in
it. A block with `live_count == 0` is reusable.

(The docstring here described `VkManagedBuffer` — it had been attached to this
struct rather than that one since before the hoist, so it is corrected rather
than carried across.)
"""
mutable struct PoolBlock
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    base_address::UInt64       # BDA of the start of this block
    capacity::Int              # Total bytes in this block
    bump::Int                  # Next free byte offset (bump pointer for initial carving)
    live_count::Int            # Number of live sub-allocations
    # The `DevicePool` this block belongs to. `Any` because `DevicePool` holds a
    # `Vector{PoolBlock}` and Julia has no forward declaration; every use site
    # asserts `::DevicePool`, the same shape as `bq.ctx::VkContext`.
    #
    # A back-reference rather than a lookup because `return_to_pool!` runs from a
    # FINALIZER. A finalizer must not allocate and must not be able to fail on a
    # missing key, so the free path is a field hop and nothing else.
    pool::Any
end

mutable struct VkManagedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64     # BDA for PhysicalStorageBuffer access
    mapped_ptr::Ptr{UInt8}  # Non-null for unified/BAR memory
    size::Int
    pool_offset::Int    # Byte offset within pool block (0 for non-pooled)
    pool_block::Union{Nothing, PoolBlock}  # Back-reference for pool free (nothing = non-pooled)
    # Cross-queue synchronization: records which BatchQueue last wrote to
    # this buffer, at which timeline value. Consumed by sync_access! to
    # auto-insert semaphore waits when a dispatch on a different queue
    # takes this buffer as an argument. Nothing = never written.
    # Typed as Any so BatchQueue (defined later) doesn't force a cyclic include.
    # @atomic so the finalizer thread (vk_free!) and main thread (record /
    # sync_access!) can read/write it safely.
    @atomic last_write::Union{Nothing, Tuple{Any, UInt64}}
    # Lifecycle state — see BUF_STATE_* constants above.  @atomic CAS is the
    # single point where double-free / use-after-free is ruled out.
    @atomic state::UInt8
    # Number of live CommandBatches that have `pin!`ed an array backed by this
    # buffer.  Incremented at pin time, decremented when the batch releases its
    # pins (`release_pinned_refs!`, i.e. the batch completed or its submit
    # failed).  A buffer with pins > 0 is REACHABLE BY A BATCH THAT CAN STILL
    # SUBMIT, so `vk_free!` must not touch it — not even to mark it DEFERRED,
    # because `sync_access!` asserts the buffer is ALIVE at submit.
    #
    # This is what makes the guarantee structural rather than a timing accident:
    # `last_write` only tells us about work already *submitted*, so a buffer
    # pinned into a still-open batch looks idle to the timeline check.
    @atomic pins::Int
    # A free was requested while pins > 0.  The free is not lost, just owed: the
    # last `unpin_buffer!` performs it.
    @atomic free_requested::Bool
    # Owning VkContext — so upload!/download!/vk_free! don't need the global.
    # Loose type because VkContext is declared in device.jl, included first.
    ctx::Any
end

"""
One device's memory: its 64 MiB blocks and its per-size-class free lists.

**This was two module-level globals**, `POOL_BLOCKS` and `POOL_FREE_LISTS`, and
`PoolBlock` carried no device. So an allocation on a second device was served out
of the first device's block — measured directly: allocate on the GPU (one block
created), then allocate on lavapipe, and `length(POOL_BLOCKS)` was *still 1*.

The buffer's `ctx` was right and the memory under it belonged to the other
device, which is the worst shape a bug can have: `fill!` on the second context
read back **0.0** because it wrote into memory that device does not own, and the
same sequence in a different order segfaulted instead.

Worth separating from `GUARDRAILS.md` §8, which lists four caches holding
pipeline *handles*. This hands out *memory*. A stale handle is undefined
behaviour the driver usually catches; memory from the wrong device is silent
corruption, and no amount of cache keying reaches it. Every one of those four
caches was keyed per device before this, and two devices still did not work.
"""
mutable struct DevicePool
    blocks::Vector{PoolBlock}
    # index i holds reusable VkManagedBuffer objects of size class i
    free_lists::Vector{Vector{VkManagedBuffer}}

    # ── Policy. These were eleven module-level `Ref`s, which is the same mistake
    # as the caches one level up: a second device would have been trimmed,
    # capped and garbage-collected according to the first one's numbers. They are
    # defaults, so they stay mutable — but they are this pool's defaults.
    disabled::Bool
    accounting::Bool
    soft_cap::Int
    trim_threshold::Int
    trim_min_interval::Float64
    trim_full_gc_interval::Float64
    gc_mingap::Float64
    gc_full_mingap::Float64
    track_allocs::Bool

    # ── Bookkeeping: when this pool last trimmed or collected, and how long it
    # has spent doing it. Per device for the obvious reason — one device's
    # collection must not suppress another's.
    last_trim::Float64
    last_full_gc::Float64
    gc_last::Float64
    gc_full_last::Float64
    gc_seconds::Float64

    # ── Accounting. `live_bytes` is the numerator `gpu_memory_pressure` divides
    # by the heap size, and `live_buffers` is every buffer this device handed
    # out. Module-level, these were one number for two heaps: with a second
    # device the pressure ratio, the trim threshold and the OOM retry all read
    # the SUM of both devices against ONE device's capacity — so a busy discrete
    # GPU would drive collection on an idle integrated one, and neither would
    # report its own footprint. Atomics because `destroy_buffer!` is a finalizer.
    live_bytes::Threads.Atomic{Int}
    live_buffers::Set{VkManagedBuffer}
    # Allocation counters, zeroed by `reset_pool_accounting!`.
    requested::Threads.Atomic{Int}
    rounded::Threads.Atomic{Int}
    nalloc::Threads.Atomic{Int}
    gc_count::Threads.Atomic{Int}
    # Guards re-entry into reclamation through `flush!`'s own allocation path.
    # Per pool: one device quiescing must not make another's reclaim a no-op.
    reclaiming::Threads.Atomic{Bool}
end

# Linked result: session-dependent, stored in the cache Dict.
struct LavaLinkedKernel
    compiled::LavaGPUKernel        # SPIR-V bytes + push_info (serializable)
    pipeline::LavaComputePipeline  # VkPipeline (session-dependent, NOT serializable)
    offsets::Vector{Int}           # arg layout offsets (derived from push_info)
    byval_sizes::Vector{Int}      # LLVM byval sizes (derived from push_info)
end

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

"""
The compiled prepare-indirect kernel for one device.

It owns a `VkPipeline`, so it is per device (`GUARDRAILS.md` §8) — four separate
`Ref`s before, which meant four things that had to be reset together and were
reachable from the wrong device in exactly the same way. Bundling them makes the
per-device dict hold one value instead of four parallel ones.
"""
struct PrepareIndirect
    pipeline::LavaComputePipeline
    offsets::Vector{Int}
    byval_sizes::Vector{Int}
    arg_buffer_size::Int
end

"""
    CompiledGraphicsPipeline

A compiled graphics pipeline ready for draw commands.
"""
struct CompiledGraphicsPipeline
    pipeline::Vulkan.Pipeline
    pipeline_layout::Vulkan.PipelineLayout
    modules::Vector{Vulkan.ShaderModule}
    push_constant_size::UInt32
    descriptor_set_layout::Union{Nothing, Vulkan.DescriptorSetLayout}
    push_stage_flags::Vulkan.ShaderStageFlag
    # Pipeline state (for debug/inspection). Plural: with dynamic rendering a
    # pipeline is built for a whole set of colour attachment formats.
    color_formats::Vector{Vulkan.Format}
    has_depth::Bool
end

"""
    Diagnostics

Every debugging and instrumentation toggle Lava has, owned by the context they
describe.

**These were eighteen module-level `Ref`s.** They are the same mistake as the
caches and the pool policy one level up, with a milder symptom: turning on
allocation tracing or dispatch logging did it for *every* device in the process,
and a second device could not be instrumented independently of the first. Two of
them carry counters (`spirv_dump_counter`, `kernel_debug_counter`) whose values
were shared across devices that emit different kernels.

`DNNKernels` did exactly this in its own step 3 — `OPTIMES`, `OPDOUBLE`,
`OPDOUBLEFILTER`, `PLAN_MISSES` and `LAUNCH_PROBE` became `Ctx.diag` — and the
argument there applies here: two differently-instrumented runs at once become
possible, and nothing has to be reset.

Off by default, and free when off: every read is a field load behind a branch the
compiler hoists, which is what the `Ref`s cost too.
"""
mutable struct Diagnostics
    # ── allocation / free
    alloc_debug::Bool
    free_debug::Bool
    freed_bda_scan::Bool
    destroy_freed_bdas_throws::Bool
    presubmit_scan::Bool
    presubmit_scan_throws::Bool
    pack_arg_assert_live::Bool
    slab_dump_target::Any
    # ── dispatch / batch
    batch_timing::Bool
    dispatch_logging::Bool
    dispatch_log_file::Union{Nothing,String}
    dispatch_timing::Bool
    # ── compiler / cache
    frozen_log_misses::Bool
    launch_arg_validation::Bool
    spirv_dump_dir::Union{Nothing,String}
    spirv_dump_counter::Int
    kernel_debug_counter::Int
    broadcast_probe::Any

    # ── The buffers the flags above fill. They were module-level `Vector`s and
    # `Dict`s, which the census that drove this refactor could not see: it
    # matched `= Ref` and these are collections. Same defect all the same — with
    # two devices every log interleaved both, so the record of what one device
    # allocated, dispatched or waited on was a record of neither.
    alloc_log::Vector{NamedTuple}
    free_log::Vector{NamedTuple}
    freed_bda_scan_log::Vector{NamedTuple}
    slab_dump_log::Vector{NamedTuple}
    # Allocation totals by tag, filled when `track_allocs` is on.
    alloc_trace::Dict{Symbol,Tuple{Int,Int}}
    alloc_trace_lock::ReentrantLock
    # The rolling dispatch log `throw_with_validation_context` prints, and the
    # per-batch wait times behind `batch_timing`.
    dispatch_log::Vector{String}
    batch_wait_times::Vector{Float64}
    batch_wait_info::Vector{String}
    batch_wait_dispatches::Vector{Int}
    # Pipelines this device compiled or refused to.
    pipeline_compile_misses::Vector{String}
    pipeline_compiles_refused::Int
    # Lifetime dispatch counters. Atomics: `submit!` may run off the main thread.
    flush_counter::Threads.Atomic{Int}
    total_dispatches::Threads.Atomic{Int}
end

Diagnostics() = Diagnostics(false, false, false, false, false, false, false, nothing,
                            false, false, nothing, false,
                            false, true, nothing, 0, 0, nothing,
                            NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[],
                            Dict{Symbol,Tuple{Int,Int}}(), ReentrantLock(),
                            String[], Float64[], String[], Int[],
                            String[], 0,
                            Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))

# ── Per-device state ─────────────────────────────────────────────────────────

const VAL_RING_SLOTS      = 64
const VAL_RING_SLOT_BYTES = 2048
const MAX_VALIDATION_MESSAGES = 50
const MAX_PRINTF_MESSAGES     = 4096

"""
    ValidationRingRaw

The five pointers `debug_callback` needs, in a layout it can read off
`pUserData` with one `unsafe_load`.

`isbits`, so that load is a plain memory read: the callback runs on a driver
thread, re-entrantly from inside a blocking `ccall` (GPU-AV reads back during
`vkWaitSemaphores` with driver locks held), where allocating or entering the
Julia runtime deadlocks or corrupts. That constraint is why the ring was raw
preallocated arrays — it is not a reason for them to be *module-level*, which is
what this replaces.
"""
struct ValidationRingRaw
    buf::Ptr{UInt8}
    len::Ptr{Cint}
    sev::Ptr{UInt32}
    typ::Ptr{UInt32}
    write::Ptr{Int}
end

"""
    ValidationRing

One device's validation messages: a slot ring the debug-utils callback fills on
a driver thread, and the drained strings the main thread reads.

**This was eight module-level globals, and that was a two-device bug rather than
untidiness.** `create_vulkan_context` builds a *fresh* `Vulkan.Instance` and
`DebugUtilsMessengerEXT` per context — `VkContext` already holds both as fields —
so two contexts meant two messengers writing into ONE ring. Device A's validation
errors surfaced in device B's `get_validation_messages()`, and
`check_validation_errors!` would raise one device's fault at the other device's
call site. The drain cursor was shared too, so whichever device drained first
consumed the other's messages.

`VkDebugUtilsMessengerCreateInfoEXT` carries a `pUserData` pointer for exactly
this, and `debug_callback` already took (and ignored) it.

The arrays are allocated once and never resized, so the pointers cached in `raw`
stay valid; `raw` is a `RefValue` because its own address is what gets handed to
the driver, and Julia's GC does not move objects that something roots — here, the
context.
"""
mutable struct ValidationRing
    buf::Vector{UInt8}
    len::Vector{Cint}
    sev::Vector{UInt32}
    typ::Vector{UInt32}
    write::Vector{Int}      # 1 element; total messages written by the callback
    read::Int               # main-thread drain cursor
    raw::Base.RefValue{ValidationRingRaw}
    # Drained, classified output. Hard errors for `check_validation_errors!`;
    # `@lava_printf` text kept separate so a debug print is not an error.
    messages::Vector{String}
    printf::Vector{String}
end

function ValidationRing()
    buf = zeros(UInt8,  VAL_RING_SLOTS * VAL_RING_SLOT_BYTES)
    len = zeros(Cint,   VAL_RING_SLOTS)
    sev = zeros(UInt32, VAL_RING_SLOTS)
    typ = zeros(UInt32, VAL_RING_SLOTS)
    wr  = zeros(Int, 1)
    raw = Ref(ValidationRingRaw(pointer(buf), pointer(len), pointer(sev),
                                pointer(typ), pointer(wr)))
    ValidationRing(buf, len, sev, typ, wr, 0, raw, String[], String[])
end

"""The `pUserData` this ring is reached through. Stable for the ring's lifetime."""
ring_user_data(r::ValidationRing) = Ptr{Cvoid}(pointer_from_objref(r.raw))

"""
    IterPlan

A kernel's launch decomposition: the KA context, the 3-D block grid, the
workgroup size, and the block count.

Lives here rather than beside its only user in `ka_backend.jl` so
`DeviceCaches.iterplans` can name its element type. `block_dims` comes from
`pad_to_3d(ctx, …)`, which reads `ctx.max_wg_dims` — so a plan describes one
device, and the cache holding it has to belong to that device.
"""
struct IterPlan{Ctx}
    ka_ctx::Ctx
    block_dims::NTuple{3, Int}
    ws_3d::NTuple{3, Int}
    nblocks::Int
end

"""
    DeviceCaches

Everything a `VkContext` caches, owned by the context that owns the handles.

**These were twelve module-level globals.** Ten had been keyed by `ctx.id` to
make two devices work, which fixed the sharing but not the ownership: entries
outlived the device they described, `ctx.id` was a surrogate for "the object I
should have stored this on", and `RESET_CALLBACKS` existed almost entirely to
empty them. A field dies with its context, so none of that is needed.

The last two — `BLIT_PIPELINE` and `TIMESTAMP_POOL` — were never keyed at all.
They are module-level `Ref`s holding device-owned handles, which is precisely the
shape that produced the function-pointer crash and the memory-pool corruption;
they survived only because the two-device probe's path (dispatch, reduction,
GEMM) reaches neither graphics nor dispatch profiling.

`blit` is `Any` because `GraphicsPipeline` is defined near the end of the include
list — no worse than the `Ref{Any}` it replaces, and the only field that is not
concrete.
"""
mutable struct DeviceCaches
    # Compute pipelines, keyed by SPIR-V content hash. The key no longer needs
    # `ctx.id` mixed in: two devices cannot collide when they do not share a dict.
    pipelines::Dict{UInt64,LavaComputePipeline}
    pipeline_order::Vector{UInt64}
    # `Dict{Any,…}` because GPUCompiler.cached_compilation derives the key itself.
    linked::Dict{Any,LavaLinkedKernel}
    launchplans::IdDict{DataType,Vector{LaunchPlan}}
    prepare_indirect::Union{Nothing,PrepareIndirect}
    pool::DevicePool
    # 0 means "not yet queried" — the device never reports 0.
    subgroup_size::Int
    subgroup_control::Union{Nothing,SubgroupSizeControl}
    # One warning per device about a subgroup width that cannot be pinned.
    coopmat_warned::Bool
    gfx_pipelines::Dict{UInt64,CompiledGraphicsPipeline}
    gfx_shaders::Dict{UInt64,LavaGfxShader}
    blit::Any
    timestamp_pool::Union{Nothing,Vulkan.QueryPool}
    timestamp_next_slot::Int
    timestamp_period_ns::Float64
    # The dispatches this device timestamped. `Any` because `DispatchTiming` is
    # declared in `profiling.jl`; the slots it indexes are `timestamp_pool`'s, so
    # a module-level vector was a list of one device's slot numbers read against
    # whichever device's pool happened to be current.
    recorded_dispatches::Vector{Any}
    # The frozen kernel cache's session memo, holding `LavaLinkedKernel`s — each
    # of which owns a `VkPipeline`. Same §8 class as the caches above, and missed
    # for the same reason `BLIT_PIPELINE` was: nothing on the two-device probe's
    # path enables the frozen cache. Its RT sibling `FROZEN_RT_MEM` stays a
    # global on purpose — a `LavaRTShader` is bytes and metadata, no handle.
    frozen_mem::Dict{Tuple{DataType,DataType,Any},Any}
    # A baton, not a cache: `frozen_link_recording` leaves the per-kernel
    # `VkPipelineCache` here for `frozen_store` to snapshot. Narrow window, but a
    # driver object all the same, and two devices recording at once would swap it.
    frozen_last_pcache::Any
    # Launch decompositions, keyed by (kernel type, ndrange, workgroupsize) — a
    # key that does NOT name the device, while the value does: `block_dims` is
    # `pad_to_3d(ctx, …)` over `ctx.max_wg_dims`. Two devices with different
    # `maxComputeWorkGroupCount` therefore handed the second one the first one's
    # block grid. Same class as the caches above; it survived the first sweep
    # because that census matched `= Ref` and this is a `Dict`.
    iterplans::Dict{Any,IterPlan}
    # Scratch buffers. These hold DEVICE MEMORY, so a process-wide one is the
    # defect the memory pool had: the second device is handed the first's buffer.
    # Both were already keyed by context, which is the surrogate a field replaces.
    reduce_scratch::Any          # one unified cell for `vk_reduce_sum`
    gemm_split_scratch::Any      # split-K partials; grows, never shrinks
    # Grown-past scratch, retained rather than dropped: dispatches already
    # recorded point into the old buffer and its finalizer would pull it out
    # from under them.
    gemm_split_retired::Vector{Any}
end

# `DevicePool()` resolves at call time, long after `memory.jl` is loaded.
DeviceCaches() = DeviceCaches(
    Dict{UInt64,LavaComputePipeline}(), UInt64[],
    Dict{Any,LavaLinkedKernel}(), IdDict{DataType,Vector{LaunchPlan}}(),
    nothing, DevicePool(), 0, nothing, false,
    Dict{UInt64,CompiledGraphicsPipeline}(), Dict{UInt64,LavaGfxShader}(),
    nothing, nothing, 0, 1.0, Any[],
    Dict{Tuple{DataType,DataType,Any},Any}(), nothing,
    Dict{Any,IterPlan}(), nothing, nothing, Any[])
