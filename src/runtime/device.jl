# Vulkan device initialization for Lava.jl
#
# Singleton VkContext holds all persistent Vulkan state.
# Lazy initialization: first use triggers device creation.
#
# Required features: BufferDeviceAddress, VariablePointers, Int64, Float64
# Optional features: AccelerationStructure, RayTracingPipeline

"""
    RTPipelineProperties

Ray tracing pipeline properties queried from the physical device.
`nothing` if RT extensions are not available.
"""
struct RTPipelineProperties
    shader_group_handle_size::UInt32
    shader_group_base_alignment::UInt32
    shader_group_handle_alignment::UInt32
    max_ray_recursion_depth::UInt32
    max_ray_hit_attribute_size::UInt32
end

"""
    CommandBatch

A single recording batch that may span multiple Vulkan command buffers.
When the number of dispatches in the current CB segment exceeds `CB_SPLIT_THRESHOLD`,
the CB is sealed and a fresh one is started. At flush time, all sealed CBs + the
active CB are submitted in a single `vkQueueSubmit` call.

This avoids NVIDIA driver crashes from enormous command buffers (30k+ dispatches)
while keeping submission count minimal (single submit per flush).
"""
mutable struct CommandBatch
    cmd_buf::Vulkan.CommandBuffer       # Currently recording CB segment
    recording::Bool
    dispatch_count::Int                 # Total dispatches across all segments (for barriers)
    segment_dispatches::Int             # Dispatches in current CB segment (for split threshold)
    last_was_rt::Bool
    # All objects this batch keeps alive until the fence is reached.  IdSet
    # so each object is pinned (and sync-tracked) at most once per batch.
    # Populated by `pin!` during arg packing and from dispatch entry points.
    pinned::Base.IdSet{Any}
    # Retained `GPUArrays.DataRef`s for every pinned LavaArray, taken via
    # `copy(a.buf)` at `pin!` time.  `pinned` alone is not enough: it keeps the
    # *wrapper* alive, but an explicit `unsafe_free!(a)` (HW-accel BLAS/TLAS
    # teardown does exactly this) sets `a.buf.freed = true` on that DataRef, and
    # `DataRef` throws on `freed` regardless of refcount — so `submit!` would
    # later trip "Attempt to use a freed reference" dereferencing `a.buf`.
    # A `copy` is an independent, non-freed handle onto the same RefCounted, so
    # it both keeps the VkManagedBuffer alive and stays dereferenceable.
    # Released in `reclaim_batch!` once the batch's timeline has been reached.
    pinned_refs::Vector{Any}
    dispatch_log::Vector{String}
    sealed_cmd_bufs::Vector{Vulkan.CommandBuffer}  # Completed CB segments awaiting submit

    # Timeline value this batch will signal on its queue's `timeline_sem`.
    # Assigned at record time so `sync_access!` can store it into `buf.last_write`.
    signal_value::UInt64
    # Cross-queue dependencies, built up by `sync_access!(::VkManagedBuffer)` at submit.
    wait_semaphores::Vector{Tuple{Vulkan.Semaphore, UInt64, Vulkan.PipelineStageFlag2}}
    # Back-reference to the owning BatchQueue.  Set post-construction (chicken/
    # egg: init_batch runs inside BatchQueue's constructor).  Always non-nothing
    # after the BatchQueue is fully built; checked via `batch.bq`.
    # Loose type because BatchQueue is declared above but the reverse dep still
    # makes `CommandBatch.bq::BatchQueue` fragile in the struct body.
    bq::Any
end

"""
    BatchQueue

An independent command submission channel owning a Vulkan queue, command pool,
and batch state. Multiple `BatchQueue`s can record and submit independently
(e.g., primary queue for graphics/present, compute queue for async RT).

Create with `BatchQueue(device, queue, queue_family_index)`.
"""
mutable struct BatchQueue
    device::Vulkan.Device
    queue::Vulkan.Queue
    family_index::UInt32
    cmd_pool::Vulkan.CommandPool
    active_batch::Union{Nothing, CommandBatch}
    in_flight::Vector{CommandBatch}
    free_batches::Vector{CommandBatch}
    free_cmd_bufs::Vector{Vulkan.CommandBuffer}
    # Dedicated AS-build command buffer + fence — allocated from this BQ's
    # own cmd_pool and submitted on this BQ's queue.  Keeping them on the
    # BQ (not the VkContext) means AS build, submit and queue are locked
    # together by construction.
    as_cmd_buf::Vulkan.CommandBuffer
    as_fence::Vulkan.Fence

    # ── Explicit-queue refactor additions ────────────────────────────────
    # One timeline semaphore per queue.  Each submit signals next_timeline+1.
    timeline_sem::Vulkan.Semaphore
    next_timeline::UInt64
    # Buffers queued for destruction once their last_write timeline value
    # is reached. Drained by drain_deferred_frees! at natural sync points.
    # Loose type (VkManagedBuffer is declared later in memory.jl).
    #
    # Cross-thread: finalizer threads push into this list via `vk_free!`;
    # the main thread iterates + drains via `drain_deferred_frees!`.  The
    # `deferred_frees_lock` below guards both operations on `deferred_frees`
    # AND `deferred_as_frees`.  SpinLock because contention is near-zero
    # (finalizer pushes at GC pauses, drain happens at sync points).
    deferred_frees::Vector{Any}
    # LavaBLAS / LavaTLAS queued for destruction once their `last_use`
    # timeline value is reached. Drained by `drain_deferred_as_frees!`.
    # Loose type — Lava AS types are declared later in raytracing/acceleration.jl.
    deferred_as_frees::Vector{Any}
    # Guards `deferred_frees` AND `deferred_as_frees`.  Acquired on every
    # push from finalizer threads and on every drain from the main thread.
    deferred_frees_lock::Base.Threads.SpinLock
    # Per-BQ argument-buffer slab pool.  Each submit bump-allocates from
    # the current slab; `reset_arg_buffer_pool!(bq)` (called from
    # reclaim_batch! once in_flight is empty) rewinds the bump pointer.
    # Element type is `LavaArray{UInt8,1}` (unified/BAR memory); kept loose
    # because LavaArray is declared later in array/lavaarray.jl.
    arg_slabs::Vector{Any}
    arg_slab_idx::Int
    arg_slab_offset::Int
    arg_alloc_count::Int
    # Per-BQ indirect-dispatch buffer slab pool.  Element type is
    # `LavaArray{UInt32,1}` (unified + INDIRECT_BUFFER_BIT).  Reset by
    # `reset_indirect_buffer_pool!(bq)`.
    indirect_slabs::Vector{Any}
    indirect_slab_idx::Int
    indirect_slab_offset::Int
    # Per-BQ staging buffer for CPU↔GPU transfers. A single VkManagedBuffer
    # that grows as needed via get_staging!. Reused across transfers.
    # Loose type — VkManagedBuffer is declared later in memory.jl.
    staging::Union{Nothing, Any}
    # Back-reference to owning VkContext. Set post-construction by
    # init_vulkan!/allocate_batch_queue!. `nothing` only during the brief
    # window of default_bq construction before VkContext exists.
    # Loose type — VkContext is declared below.
    ctx::Any
    # Single-writer invariant: only this thread may record into or submit
    # from this BatchQueue.  Captured at construction from `Threads.threadid()`.
    # Every dispatch-recording / sweep / slab-alloc entry point asserts that
    # it is running on this thread — an accidental cross-thread call trips
    # the assert immediately instead of silently corrupting state.
    owning_thread::Int
end

function init_batch(cb::Vulkan.CommandBuffer)
    pinned = Base.IdSet{Any}()
    sizehint!(pinned, 128)
    waits = Tuple{Vulkan.Semaphore, UInt64, Vulkan.PipelineStageFlag2}[]
    return CommandBatch(cb, false, 0, 0, false, pinned, Any[], String[],
        Vulkan.CommandBuffer[],
        UInt64(0),                       # signal_value (assigned at record time)
        waits,
        nothing,                         # bq (set after BatchQueue is fully built)
    )
end

function BatchQueue(device::Vulkan.Device, queue::Vulkan.Queue, qf_idx::UInt32, ctx;
                    n_initial_batches::Int=2)
    cmd_pool = Vulkan.CommandPool(device, qf_idx;
        flags=Vulkan.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    batches = CommandBatch[]
    for _ in 1:n_initial_batches
        alloc_info = Vulkan.CommandBufferAllocateInfo(cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
        cb = unwrap(Vulkan.allocate_command_buffers(device, alloc_info))[1]
        push!(batches, init_batch(cb))
    end
    # Dedicated AS-build command buffer + fence (same pool as this bq).
    as_alloc = Vulkan.CommandBufferAllocateInfo(cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    as_cmd_buf = unwrap(Vulkan.allocate_command_buffers(device, as_alloc))[1]
    as_fence = Vulkan.Fence(device)
    # Per-queue timeline semaphore for cross-queue ordering.
    type_info = Vulkan.SemaphoreTypeCreateInfo(Vulkan.SEMAPHORE_TYPE_TIMELINE, UInt64(0))
    timeline_sem = unwrap(Vulkan.create_semaphore(device,
        Vulkan.SemaphoreCreateInfo(; next=type_info)))
    bq = BatchQueue(device, queue, qf_idx, cmd_pool, nothing,
                    CommandBatch[], batches, Vulkan.CommandBuffer[],
                    as_cmd_buf, as_fence,
                    timeline_sem, UInt64(0),
                    Any[], Any[],    # deferred_frees, deferred_as_frees
                    Base.Threads.SpinLock(),  # deferred_frees_lock
                    Any[], 1, 0, 0,  # arg_slabs: idx=1, offset=0, count=0
                    Any[], 1, 0,     # indirect_slabs: idx=1, offset=0
                    nothing,         # staging (lazy)
                    ctx,             # owning VkContext (required)
                    Threads.threadid())  # owning_thread
    # Plug the back-reference into every pre-allocated batch so `batch.bq`
    # is non-nothing as soon as the bq is returned.  Future batches allocated
    # lazily (alloc_cmd_buf → init_batch) must set .bq themselves.
    for b in batches
        b.bq = bq
    end
    return bq
end

"""
    VkContext

Persistent Vulkan context. Batch-based command recording goes through
`default_bq::BatchQueue` (the primary queue). Use `BatchQueue(...)` to
create additional independent queues (e.g., for async compute/RT).
"""
mutable struct VkContext
    instance::Vulkan.Instance
    physical_device::Vulkan.PhysicalDevice
    device::Vulkan.Device
    queue_family_index::UInt32
    device_name::String
    # Primary batch queue — all global API functions delegate here.  Always
    # non-nothing after the inner constructor returns (BatchQueue is built
    # using `new()`-based two-phase init to break the chicken-and-egg with
    # BatchQueue.ctx).
    default_bq::BatchQueue
    # Secondary compute queue (async RT) — same family, separate queue object
    compute_queue::Vulkan.Queue
    # Ray tracing (nothing if not available)
    rt_pipeline_properties::Union{Nothing, RTPipelineProperties}
    # Debug messenger (nothing if validation layers not available)
    debug_messenger::Any
    # Queue allocation: next available index + total requested from device
    next_queue_index::Int
    max_queue_count::Int
    # Async compute family (distinct from primary). RADV family 1: 4 queues,
    # compute+transfer. Used by the explicit-queue refactor for upload_bq.
    async_queue_family_index::Union{Nothing, UInt32}
    async_queue_count::Int
    # Per-device state (was previously a global Ref).
    # Set to `true` after vkQueueSubmit returns DEVICE_LOST. Finalizers
    # holding a ref to the context check this to skip Vulkan calls on
    # invalid handles.
    device_lost::Bool
    # Cached physical-device properties (avoid re-querying on every alloc/dispatch).
    memory_properties::Vulkan.PhysicalDeviceMemoryProperties
    max_wg_dims::NTuple{3, Int}
    # Alignment (bytes) for BDAs passed as `pScratchData` in AS builds.
    # Picked by `bda_alignment_for(ctx, scratch=true)`.
    as_scratch_align::UInt64
    # Whether VK_KHR_ray_query is available on this device.
    # Set to true by B1 (device extension probe). False until proven otherwise.
    ray_query_available::Bool
    # Whether VK_NV_ray_tracing_invocation_reorder (SER) is available.
    # When true, the SPIR-V emitter declares the ShaderInvocationReorderNV
    # capability and the raygen can use `lava_rt_hit_object_*` /
    # `lava_rt_reorder_thread_*` intrinsics.  NVIDIA-only.
    ser_available::Bool
    # Whether VK_KHR_cooperative_matrix is enabled: subgroup-scope matrix
    # multiply-accumulate (tensor cores). `coopmat_shapes` holds the driver's
    # legal (M, N, K, A/B type, C/result type, saturating) combinations -- the
    # hardware only implements a fixed set, so a kernel must pick one of these
    # rather than any tile size it likes. Empty when unsupported.
    coopmat_available::Bool
    coopmat_shapes::Vector{NamedTuple{(:M, :N, :K, :ab_type, :c_type, :scope),
                                      Tuple{Int, Int, Int, UInt32, UInt32, UInt32}}}
    # Whether VK_EXT_memory_budget is enabled. When true, OOM error reporting
    # queries the driver's real per-heap budget vs usage via
    # VkPhysicalDeviceMemoryBudgetPropertiesEXT.
    memory_budget_available::Bool
    # Whether VK_KHR_external_memory(_fd) is enabled. When true,
    # `ExternalImage` can export allocations as opaque fds for zero-copy
    # sharing with other APIs (OpenGL via GL_EXT_memory_object_fd).
    external_memory_available::Bool
    # Whether VK_KHR_video_decode_queue + h264 are enabled (hardware decode into
    # Vulkan images). The decode queue + its family index are `nothing` when the
    # device has no video-decode support. No effect until a decode session is made.
    video_decode_available::Bool
    video_decode_queue::Union{Nothing, Vulkan.Queue}
    video_decode_queue_family_index::Union{Nothing, UInt32}
    # Whether GPU-Assisted Validation is active on this instance. Captured
    # so callers can check (e.g. verify_gpu_av) without re-reading env vars.
    gpu_assisted::Bool
    # Driver version (used to key the on-disk VkPipelineCache file).
    driver_version::String
    # Persistent VkPipelineCache. Seeded from disk on init, passed to every
    # vkCreate*Pipelines call, snapshotted back on vk_reset_device! + atexit.
    pipeline_cache::Vulkan.PipelineCache

    # Inner constructor: two-phase init via `new()` so we can hand a live
    # `ctx` reference to `BatchQueue(...)` while finishing the ctx's own
    # field assignments.  There is no public ctor that can leave `default_bq`
    # unset.  `primary_queue` is the raw `Vulkan.Queue` for the default bq;
    # everything else maps directly to a field.
    function VkContext(instance::Vulkan.Instance,
                       physical_device::Vulkan.PhysicalDevice,
                       device::Vulkan.Device,
                       queue_family_index::UInt32,
                       device_name::String,
                       primary_queue::Vulkan.Queue,
                       compute_queue::Vulkan.Queue,
                       rt_pipeline_properties::Union{Nothing, RTPipelineProperties},
                       debug_messenger::Any,
                       next_queue_index::Int,
                       max_queue_count::Int,
                       async_queue_family_index::Union{Nothing, UInt32},
                       async_queue_count::Int,
                       device_lost::Bool,
                       memory_properties::Vulkan.PhysicalDeviceMemoryProperties,
                       max_wg_dims::NTuple{3, Int},
                       as_scratch_align::UInt64,
                       ray_query_available::Bool=false,
                       ser_available::Bool=false,
                       coopmat_available::Bool=false,
                       coopmat_shapes=nothing,
                       memory_budget_available::Bool=false,
                       external_memory_available::Bool=false,
                       gpu_assisted::Bool=false,
                       driver_version::AbstractString="unknown",
                       video_decode_available::Bool=false,
                       video_decode_queue::Union{Nothing, Vulkan.Queue}=nothing,
                       video_decode_queue_family_index::Union{Nothing, UInt32}=nothing)
        ctx = new()
        ctx.instance = instance
        ctx.physical_device = physical_device
        ctx.device = device
        ctx.queue_family_index = queue_family_index
        ctx.device_name = device_name
        ctx.compute_queue = compute_queue
        ctx.rt_pipeline_properties = rt_pipeline_properties
        ctx.debug_messenger = debug_messenger
        ctx.next_queue_index = next_queue_index
        ctx.max_queue_count = max_queue_count
        ctx.async_queue_family_index = async_queue_family_index
        ctx.async_queue_count = async_queue_count
        ctx.device_lost = device_lost
        ctx.memory_properties = memory_properties
        ctx.max_wg_dims = max_wg_dims
        ctx.as_scratch_align = as_scratch_align
        ctx.ray_query_available = ray_query_available
        ctx.ser_available = ser_available
        ctx.coopmat_available = coopmat_available
        ctx.coopmat_shapes = coopmat_shapes === nothing ?
            eltype(fieldtype(VkContext, :coopmat_shapes))[] : coopmat_shapes
        ctx.memory_budget_available = memory_budget_available
        ctx.external_memory_available = external_memory_available
        ctx.video_decode_available = video_decode_available
        ctx.video_decode_queue = video_decode_queue
        ctx.video_decode_queue_family_index = video_decode_queue_family_index
        ctx.gpu_assisted = gpu_assisted
        ctx.driver_version = driver_version
        # Seed a persistent VkPipelineCache from disk (if any). Driver
        # validates the header — bad/stale data is silently ignored.
        ctx.pipeline_cache = create_lava_pipeline_cache(
            device, lava_pipeline_cache_path(device_name, driver_version))
        _register_pipeline_cache_atexit!()
        # Now build the default BatchQueue with the live ctx.  Sets the
        # remaining field; no nullable slot, no post-hoc mutation.
        ctx.default_bq = BatchQueue(device, primary_queue, queue_family_index, ctx)
        return ctx
    end
end

# Ring buffer of recent validation messages for context on DEVICE_LOST
const VALIDATION_MESSAGES = String[]
const MAX_VALIDATION_MESSAGES = 50

# Captured @lava_printf output (NonSemantic.DebugPrintf, delivered at INFO
# severity via the debug-utils callback). Kept separate from validation errors.
const PRINTF_MESSAGES = String[]
const MAX_PRINTF_MESSAGES = 4096

# ── Async-safe validation message capture ──────────────────────────────────
#
# The Vulkan debug-utils callback can be invoked re-entrantly from inside a
# blocking driver ccall — notably GPU-Assisted Validation reading back its
# error log during `vkWaitSemaphores`.  In that context the calling thread is
# mid-ccall with driver locks held; doing ANY Julia work that can allocate,
# log, or hit a scheduler yield/safepoint deadlocks the runtime.  (Observed on
# the RTX 4000 Ada: GPU-AV catches a BDA OOB, `@error` prints it once, then the
# process hangs forever — the `@error` yielded while the driver held a lock.)
#
# So the callback writes ONLY into preallocated memory via raw ccalls
# (strlen/memcpy) and `unsafe_store!`/`@inbounds` array writes — no allocation,
# no logging, no `push!`.  The main thread later calls
# `drain_validation_messages!()` to turn raw slots into Strings, capture the
# hard errors in VALIDATION_MESSAGES, and log everything.  These const arrays
# are never resized, so their data pointers are stable for the callback.
const VAL_RING_SLOTS      = 64
const VAL_RING_SLOT_BYTES = 2048
const VAL_RING_BUF  = zeros(UInt8,  VAL_RING_SLOTS * VAL_RING_SLOT_BYTES)
const VAL_RING_LEN  = zeros(Cint,   VAL_RING_SLOTS)
const VAL_RING_SEV  = zeros(UInt32, VAL_RING_SLOTS)
const VAL_RING_TYPE = zeros(UInt32, VAL_RING_SLOTS)
const VAL_RING_WRITE = zeros(Int, 1)   # total messages written by the callback
const VAL_RING_READ  = Ref{Int}(0)     # main-thread drain cursor

const VK_CONTEXT_REF = Ref{Union{Nothing, VkContext}}(nothing)

"""
    device_lost(ctx::VkContext)  ->  Bool

Whether this context's device has been marked lost. Prefer passing `ctx`
explicitly; the no-arg form looks up the current default context.
"""
device_lost(ctx::VkContext) = ctx.device_lost
device_lost() = let ctx = VK_CONTEXT_REF[]
    ctx === nothing ? false : ctx.device_lost
end

"""Mark `ctx`'s device as lost. All subsequent finalizers will skip Vulkan calls."""
mark_device_lost!(ctx::VkContext) = (ctx.device_lost = true; nothing)

# Callbacks for vk_reset_device! — registered by later-included files (pipeline.jl,
# command.jl, launch.jl, memory.jl) to clear their module-level caches.
const RESET_CALLBACKS = Function[]

"""
    vk_context() -> VkContext

Get or create the global Vulkan context. Lazily initializes on first call.
"""
function vk_context()
    ctx = VK_CONTEXT_REF[]
    if ctx === nothing
        ctx = init_vulkan!()
        VK_CONTEXT_REF[] = ctx
    end
    return ctx
end

vk_device() = vk_context().device

"""
    vk_reset_device!()

Reinitialize the Vulkan device after DEVICE_LOST or other unrecoverable errors.
Destroys the old context and creates a fresh one. Clears all caches (pipelines,
kernels, arg buffers).

**WARNING**: All existing `LavaArray`s become INVALID after reset — their backing
GPU buffers no longer exist. You must reallocate all GPU data.
"""
function vk_reset_device!()
    # Persist the VkPipelineCache before tearing the device down so the
    # next session can skip AMDVLK's SPIR-V → ISA recompile.
    let old = VK_CONTEXT_REF[]
        old === nothing || save_pipeline_cache!(old)
    end
    # Drop the context ref; a fresh one will be created lazily. Pre-reset
    # VkManagedBuffers hold a strong ref to the OLD ctx whose `device_lost`
    # is already true, so their finalizers will skip Vulkan calls.
    VK_CONTEXT_REF[] = nothing
    # Don't destroy old Vulkan handles — they're invalid after DEVICE_LOST.
    # GC will eventually try to destroy them; _destroy_buffer! skips when
    # DEVICE_LOST was true (and we set it false only after clearing context).
    empty!(VALIDATION_MESSAGES)
    # The old messenger is gone; drop any undrained ring entries from it so a
    # later drain doesn't replay stale messages against the new instance.
    VAL_RING_READ[] = @inbounds VAL_RING_WRITE[1]
    # Run cleanup callbacks registered by other modules
    for cb in RESET_CALLBACKS
        try
            cb()
        catch e
            @warn "Lava: reset callback failed" exception=e
        end
    end
    # Re-initialize (lazy init on next vk_context() call)
    ctx = vk_context()
    @info "Lava: device reset complete" device=ctx.device_name
    return nothing
end

"""
    has_active_recording(bq::BatchQueue) -> Bool

Whether `bq` has an open/recording CommandBatch.  Used by transfer paths
to decide "should I flush `bq` before doing my own submit/CPU write?"
Always takes the queue explicitly — no implicit default_bq lookup.
"""
has_active_recording(bq::BatchQueue) = bq.active_batch !== nothing

function init_vulkan!()
    # Create instance — target Vulkan 1.4 (device supports 1.4.335 on RADV).
    # Bumping API version unlocks 1.3/1.4 core features we enable below.
    app_info = Vulkan.ApplicationInfo(
        v"0.1.0", v"0.1.0", v"1.4.0";
        application_name="Lava.jl",
        engine_name="Lava"
    )
    # Validation layers: opt-in via LAVA_VALIDATION=1 (default: off)
    want_validation = get(ENV, "LAVA_VALIDATION", "0") != "0"
    layers = String[]
    if want_validation
        available_layers = unwrap(Vulkan.enumerate_instance_layer_properties())
        for l in available_layers
            name = String(filter(!=('\0'), collect(l.layer_name)))
            if name == "VK_LAYER_KHRONOS_validation"
                push!(layers, "VK_LAYER_KHRONOS_validation")
                break
            end
        end
    end

    # Collect all available instance extensions (driver + layer-provided)
    inst_extensions = String["VK_KHR_surface"]
    available_ext = unwrap(Vulkan.enumerate_instance_extension_properties())
    ext_names = Set(String(filter(!=('\0'), collect(e.extension_name))) for e in available_ext)
    # Also collect extensions provided by the validation layer
    has_validation = !isempty(layers)
    if has_validation
        layer_ext = unwrap(Vulkan.enumerate_instance_extension_properties(; layer_name="VK_LAYER_KHRONOS_validation"))
        for e in layer_ext
            push!(ext_names, String(filter(!=('\0'), collect(e.extension_name))))
        end
    end
    has_debug_utils = "VK_EXT_debug_utils" in ext_names
    if has_debug_utils
        push!(inst_extensions, "VK_EXT_debug_utils")
    end
    # Platform-specific surface extension (ext_names already computed above)
    if Sys.islinux()
        if "VK_KHR_xcb_surface" in ext_names
            push!(inst_extensions, "VK_KHR_xcb_surface")
        elseif "VK_KHR_xlib_surface" in ext_names
            push!(inst_extensions, "VK_KHR_xlib_surface")
        end
        if "VK_KHR_wayland_surface" in ext_names
            push!(inst_extensions, "VK_KHR_wayland_surface")
        end
    elseif Sys.iswindows()
        push!(inst_extensions, "VK_KHR_win32_surface")
    elseif Sys.isapple()
        push!(inst_extensions, "VK_EXT_metal_surface")
    end

    # Extended validation: opt-in via env vars, all require VK_EXT_validation_features.
    #   LAVA_GPU_AV=1    → shader instrumentation (OOB descriptor access, ray query misuse);
    #                      does NOT cover plain BDA ranges in current layers.
    #   LAVA_SYNC_VAL=1  → synchronization validation (reads before writes, missing barriers,
    #                      cross-submit hazards).  Catches shadow-ownership UAFs.
    #   LAVA_BEST=1      → best-practices warnings (perf hints).
    # All are very slow; use one or a combination as needed for triage.
    gpu_assisted = false
    sync_val     = false
    #   LAVA_DEBUG_PRINTF=1 → enable NonSemantic.DebugPrintf output from kernels
    #                      (@lava_printf). Mutually exclusive with GPU-AV in the
    #                      layer (both instrument shaders), so it wins if both set.
    want_gpu_av  = get(ENV, "LAVA_GPU_AV",  "0") != "0"
    want_printf  = get(ENV, "LAVA_DEBUG_PRINTF", "0") != "0"
    want_gpu_av  = want_gpu_av && !want_printf   # printf takes precedence
    want_sync_val= get(ENV, "LAVA_SYNC_VAL","0") != "0"
    want_best    = get(ENV, "LAVA_BEST",    "0") != "0"
    validation_features_reqs = Vulkan.ValidationFeatureEnableEXT[]
    if want_printf
        push!(validation_features_reqs,
              Vulkan.VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT)
    end
    if want_gpu_av
        push!(validation_features_reqs,
              Vulkan.VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT)
        push!(validation_features_reqs,
              Vulkan.VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT)
    end
    if want_sync_val
        push!(validation_features_reqs,
              Vulkan.VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT)
    end
    if want_best
        push!(validation_features_reqs,
              Vulkan.VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT)
    end
    if !isempty(validation_features_reqs) && has_validation && "VK_EXT_validation_features" in ext_names
        push!(inst_extensions, "VK_EXT_validation_features")
        validation_features = Vulkan.ValidationFeaturesEXT(validation_features_reqs, [])
        instance = Vulkan.Instance(
            layers,
            inst_extensions;
            application_info=app_info,
            next=validation_features
        )
        gpu_assisted = want_gpu_av
        sync_val     = want_sync_val
    else
        instance = Vulkan.Instance(
            layers,
            inst_extensions;
            application_info=app_info
        )
        if has_validation && (want_gpu_av || want_sync_val || want_best)
            @warn "Vulkan validation layers active but VK_EXT_validation_features not available; extended validation disabled."
        end
    end

    # Set up debug messenger to capture validation/driver error messages
    debug_messenger = nothing
    if has_debug_utils
        debug_messenger = setup_debug_messenger(instance)
    end

    # Pick physical device (prefer discrete GPU)
    phys_devs = unwrap(Vulkan.enumerate_physical_devices(instance))
    isempty(phys_devs) && throw(LavaError(
        "device initialization",
        "No Vulkan-capable GPU found",
        "Ensure Vulkan drivers are installed"))

    phys_dev = pick_physical_device(phys_devs)
    props = Vulkan.get_physical_device_properties(phys_dev)
    dev_name = String(filter(!=('\0'), collect(props.device_name)))

    # Find queue family (prefer graphics+compute for graphics pipeline support)
    qf_idx = find_graphics_compute_queue_family(phys_dev)

    # Create logical device with required features
    qf_props = Vulkan.get_physical_device_queue_family_properties(phys_dev)
    max_queues = qf_props[qf_idx + 1].queue_count
    n_queues = min(4, Int(max_queues))
    queue_priorities = ones(Float32, n_queues)
    queue_ci = [Vulkan.DeviceQueueCreateInfo(qf_idx, queue_priorities)]

    # Additionally, request queues from any compute-capable family that
    # isn't the primary one (RADV exposes graphics+compute on family 0 with
    # 1 queue, and compute-only on family 1 with 4 queues). We need these
    # for async upload/dispatch in the explicit-queue design.
    async_qf_idx = find_async_compute_queue_family(phys_dev, qf_idx)
    async_n_queues = 0
    if async_qf_idx !== nothing
        async_max = qf_props[async_qf_idx + 1].queue_count
        async_n_queues = min(4, Int(async_max))
        push!(queue_ci, Vulkan.DeviceQueueCreateInfo(async_qf_idx, ones(Float32, async_n_queues)))
    end

    # Hardware video decode (opt-in): request one queue from the VIDEO_DECODE
    # family. Enabling the extensions + queue has no effect until a decode
    # session is created, so it's safe to always request when the device offers it.
    video_qf_idx = find_video_decode_queue_family(phys_dev)
    has_video_decode = video_qf_idx !== nothing &&
        has_extension(phys_dev, "VK_KHR_video_queue") &&
        has_extension(phys_dev, "VK_KHR_video_decode_queue") &&
        has_extension(phys_dev, "VK_KHR_video_decode_h264")
    has_video_decode || (video_qf_idx = nothing)
    video_qf_idx === nothing ||
        push!(queue_ci, Vulkan.DeviceQueueCreateInfo(video_qf_idx, Float32[1.0]))

    # Check for RT extension support
    has_rt = has_rt_extensions(phys_dev)
    has_ray_query = has_rt && has_extension(phys_dev, "VK_KHR_ray_query")
    # SER (Shader Execution Reordering) — NVIDIA-specific extension that
    # exposes the `HitObject*` API and `reorderThreadWithHitObjectNV` for
    # warp-level work reordering between traceRay and shading.  Lets
    # divergent path-tracer chits ((`vp_closesthit_shade`) execute in
    # coherent warps.  Optional; we still ship a working RT pipeline path
    # without it.
    has_ser = has_rt && has_extension(phys_dev, "VK_NV_ray_tracing_invocation_reorder")

    # Check for workgroup memory explicit layout (needed for mixed-type shared memory structs)
    has_wg_explicit = has_extension(phys_dev, "VK_KHR_workgroup_memory_explicit_layout")
    has_atomic_float = has_extension(phys_dev, "VK_EXT_shader_atomic_float")
    has_pipeline_exec_props = has_extension(phys_dev, "VK_KHR_pipeline_executable_properties")
    # VK_EXT_memory_budget — lets us read VkPhysicalDeviceMemoryBudgetPropertiesEXT
    # for real heap utilisation, used in OOM error reporting.
    has_memory_budget = has_extension(phys_dev, "VK_EXT_memory_budget")
    # Cooperative matrix — subgroup-scope matrix multiply-accumulate (tensor
    # cores). Probed once here; kernels pick a coopmat or a scalar
    # instantiation from `ctx.coopmat_available` / `ctx.coopmat_shapes`.
    #
    # Uses the portable VK_KHR_cooperative_matrix.
    has_coopmat = has_extension(phys_dev, "VK_KHR_cooperative_matrix")

    # Device extensions
    extensions = String[
        "VK_KHR_swapchain",
    ]
    if has_rt
        append!(extensions, [
            "VK_KHR_acceleration_structure",
            "VK_KHR_ray_tracing_pipeline",
            "VK_KHR_deferred_host_operations",
        ])
    end
    if has_ray_query
        push!(extensions, "VK_KHR_ray_query")
    end
    if has_ser
        push!(extensions, "VK_NV_ray_tracing_invocation_reorder")
    end
    if has_wg_explicit
        push!(extensions, "VK_KHR_workgroup_memory_explicit_layout")
    end
    if has_atomic_float
        push!(extensions, "VK_EXT_shader_atomic_float")
    end
    # Opt-in: pipeline executable properties (register count, scratch, vendor
    # ISA) for the profiling API.  Off by default because enabling the
    # extension changes pipeline-creation behavior on some drivers; user opts
    # in via `Lava.enable_pipeline_executable_properties!()` BEFORE first
    # device creation.
    if has_pipeline_exec_props && PIPELINE_EXEC_PROPERTIES_REQUESTED[]
        push!(extensions, "VK_KHR_pipeline_executable_properties")
    end
    if has_memory_budget
        push!(extensions, "VK_EXT_memory_budget")
    end
    if has_coopmat
        push!(extensions, "VK_KHR_cooperative_matrix")
    end
    # External-memory export (opaque fds for GL/other-API interop). Enabling
    # the extension has no effect until an ExternalImage is created.
    has_external_memory = has_extension(phys_dev, "VK_KHR_external_memory_fd")
    if has_external_memory
        push!(extensions, "VK_KHR_external_memory")
        push!(extensions, "VK_KHR_external_memory_fd")
    end
    # Hardware video decode extensions (family already probed + queue requested above).
    if has_video_decode
        append!(extensions, ["VK_KHR_video_queue", "VK_KHR_video_decode_queue",
                             "VK_KHR_video_decode_h264"])
        has_extension(phys_dev, "VK_KHR_video_decode_h265") &&
            push!(extensions, "VK_KHR_video_decode_h265")
    end

    # Chain required features — all Vulkan 1.2 promoted features go in Vulkan12Features
    # (can't mix Vulkan12Features with separate promoted structs like BDA/VariablePointers)
    var_ptr_features = Vulkan.PhysicalDeviceVariablePointersFeatures(
        true,   # variable_pointers_storage_buffer
        true,   # variable_pointers
    )
    # Vulkan 1.2 features: BDA, VulkanMemoryModel, shaderInt8, scalarBlockLayout
    vulkan12_features = Vulkan._PhysicalDeviceVulkan12Features(
        false,  # sampler_mirror_clamp_to_edge
        false,  # draw_indirect_count
        false,  # storage_buffer_8_bit_access
        false,  # uniform_and_storage_buffer_8_bit_access
        false,  # storage_push_constant_8
        false,  # shader_buffer_int_64_atomics
        false,  # shader_shared_int_64_atomics
        true,   # shader_float_16  ← REQUIRED (Float16 types in SPIR-V)
        true,   # shader_int_8  ← REQUIRED (i8 types in SPIR-V)
        false,  # descriptor_indexing
        false,  # shader_input_attachment_array_dynamic_indexing
        false,  # shader_uniform_texel_buffer_array_dynamic_indexing
        false,  # shader_storage_texel_buffer_array_dynamic_indexing
        false,  # shader_uniform_buffer_array_non_uniform_indexing
        false,  # shader_sampled_image_array_non_uniform_indexing
        false,  # shader_storage_buffer_array_non_uniform_indexing
        false,  # shader_storage_image_array_non_uniform_indexing
        false,  # shader_input_attachment_array_non_uniform_indexing
        false,  # shader_uniform_texel_buffer_array_non_uniform_indexing
        false,  # shader_storage_texel_buffer_array_non_uniform_indexing
        false,  # descriptor_binding_uniform_buffer_update_after_bind
        false,  # descriptor_binding_sampled_image_update_after_bind
        false,  # descriptor_binding_storage_image_update_after_bind
        false,  # descriptor_binding_storage_buffer_update_after_bind
        false,  # descriptor_binding_uniform_texel_buffer_update_after_bind
        false,  # descriptor_binding_storage_texel_buffer_update_after_bind
        false,  # descriptor_binding_update_unused_while_pending
        false,  # descriptor_binding_partially_bound
        false,  # descriptor_binding_variable_descriptor_count
        false,  # runtime_descriptor_array
        false,  # sampler_filter_minmax
        true,   # scalar_block_layout  ← BDA struct layout
        false,  # imageless_framebuffer
        false,  # uniform_buffer_standard_layout
        false,  # shader_subgroup_extended_types
        false,  # separate_depth_stencil_layouts
        false,  # host_query_reset
        true,   # timeline_semaphore  ← REQUIRED for explicit-queue cross-queue sync
        true,   # buffer_device_address  ← REQUIRED (BDA)
        false,  # buffer_device_address_capture_replay
        false,  # buffer_device_address_multi_device
        true,   # vulkan_memory_model  ← REQUIRED (QueueFamily scope)
        false,  # vulkan_memory_model_device_scope
        false,  # vulkan_memory_model_availability_visibility_chains
        false,  # shader_output_viewport_index
        false,  # shader_output_layer
        false;  # subgroup_broadcast_dynamic_id
        next=var_ptr_features
    )
    # Vulkan 1.3 core features, bundled.  We turn on the set that's
    # guaranteed by the 1.3 core spec and useful for a compute/RT backend:
    #   - synchronization2        (REQUIRED: we use vkQueueSubmit2 / cmd_pipeline_barrier_2)
    #   - dynamic_rendering       (no VkRenderPass/VkFramebuffer boilerplate)
    #   - maintenance4            (relaxes shader requirements, e.g. storage-buffer layout)
    #   - subgroup_size_control   (query/set subgroup size)
    #   - compute_full_subgroups  (FULL_SUBGROUPS flag in pipeline create)
    #   - pipeline_creation_cache_control (pipeline-cache hints in create)
    #   - shader_demote_to_helper_invocation (OpDemoteToHelperInvocation in SPIR-V)
    #   - shader_terminate_invocation         (OpTerminateInvocation in SPIR-V)
    #   - shader_integer_dot_product          (OpSDot / OpUDot in SPIR-V)
    #   - shader_zero_initialize_workgroup_memory (zero-init shared mem)
    #   - private_data            (VkPrivateDataSlot for driver-side tagging)
    vulkan13_features = Vulkan.PhysicalDeviceVulkan13Features(
        :synchronization2,
        :dynamic_rendering,
        :maintenance4,
        :subgroup_size_control,
        :compute_full_subgroups,
        :pipeline_creation_cache_control,
        :shader_demote_to_helper_invocation,
        :shader_terminate_invocation,
        :shader_integer_dot_product,
        :shader_zero_initialize_workgroup_memory,
        :private_data;
        next=vulkan12_features,
    )

    # Chain workgroup explicit layout if available
    feature_chain = vulkan13_features
    if has_wg_explicit
        wg_explicit_features = Vulkan.PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR(
            true,   # workgroup_memory_explicit_layout
            true,   # workgroup_memory_explicit_layout_8_bit_access
            true,   # workgroup_memory_explicit_layout_16_bit_access
            true;   # workgroup_memory_explicit_layout_scalar_block_layout
            next=feature_chain
        )
        feature_chain = wg_explicit_features
    end

    # Chain pipeline-executable-properties features (opt-in profiling).
    if has_pipeline_exec_props && PIPELINE_EXEC_PROPERTIES_REQUESTED[]
        exec_props_features = Vulkan.PhysicalDevicePipelineExecutablePropertiesFeaturesKHR(
            true;  # pipeline_executable_info
            next=feature_chain
        )
        feature_chain = exec_props_features
    end

    # Chain atomic-float features (used by Lava native reductions via OpAtomicFAdd)
    if has_atomic_float
        atomic_float_features = Vulkan.PhysicalDeviceShaderAtomicFloatFeaturesEXT(
            true,   # shader_buffer_float_32_atomics        — OpAtomicStore/Load/Exchange on f32 SSBO
            true,   # shader_buffer_float_32_atomic_add     — REQUIRED: OpAtomicFAdd on f32 SSBO
            false,  # shader_buffer_float_64_atomics
            false,  # shader_buffer_float_64_atomic_add
            true,   # shader_shared_float_32_atomics        — atomics on workgroup/shared memory
            true,   # shader_shared_float_32_atomic_add
            false,  # shader_shared_float_64_atomics
            false,  # shader_shared_float_64_atomic_add
            false,  # shader_image_float_32_atomics
            false,  # shader_image_float_32_atomic_add
            false,  # sparse_image_float_32_atomics
            false;  # sparse_image_float_32_atomic_add
            next=feature_chain
        )
        feature_chain = atomic_float_features
    end

    # Chain RT features if available
    if has_rt
        as_features = Vulkan.PhysicalDeviceAccelerationStructureFeaturesKHR(
            true,   # acceleration_structure
            false,  # acceleration_structure_capture_replay
            false,  # acceleration_structure_indirect_build
            false,  # acceleration_structure_host_commands
            false;  # descriptor_binding_acceleration_structure_update_after_bind
            next=feature_chain
        )
        rt_features = Vulkan.PhysicalDeviceRayTracingPipelineFeaturesKHR(
            true,   # ray_tracing_pipeline
            false,  # ray_tracing_pipeline_shader_group_handle_capture_replay
            false,  # ray_tracing_pipeline_shader_group_handle_capture_replay_mixed
            true,   # ray_tracing_pipeline_trace_rays_indirect
            false;  # ray_traversal_primitive_culling
            next=as_features
        )
        feature_chain = rt_features
    end
    if has_ray_query
        rq_features = Vulkan.PhysicalDeviceRayQueryFeaturesKHR(
            true;   # ray_query
            next=feature_chain
        )
        feature_chain = rq_features
    end
    if has_ser
        ser_features = Vulkan.PhysicalDeviceRayTracingInvocationReorderFeaturesNV(
            true;   # ray_tracing_invocation_reorder
            next=feature_chain
        )
        feature_chain = ser_features
    end
    if has_coopmat
        cm_features = Vulkan.PhysicalDeviceCooperativeMatrixFeaturesKHR(
            true,   # cooperative_matrix
            false;  # cooperative_matrix_robust_buffer_access
            next=feature_chain
        )
        feature_chain = cm_features
    end

    # Enable shader int64, float64, geometry/tessellation shaders, wide lines
    core_features = Vulkan.PhysicalDeviceFeatures(
        :shader_int_64, :shader_float_64,
        :shader_int_16,
        :geometry_shader, :tessellation_shader,
        :fill_mode_non_solid, :wide_lines, :large_points,
    )

    device = Vulkan.Device(
        phys_dev,
        queue_ci,
        [],         # layers
        extensions;
        enabled_features=core_features,
        next=feature_chain
    )

    queue = Vulkan.get_device_queue(device, qf_idx, 0)
    # Second queue for async compute/RT (falls back to same queue if only 1 available)
    compute_queue = n_queues >= 2 ?
        Vulkan.get_device_queue(device, qf_idx, 1) : queue
    video_decode_queue = video_qf_idx === nothing ? nothing :
        Vulkan.get_device_queue(device, video_qf_idx, 0)

    # Query RT pipeline properties
    rt_props = nothing
    if has_rt
        props2 = Vulkan.get_physical_device_properties_2(phys_dev,
            Vulkan.PhysicalDeviceRayTracingPipelinePropertiesKHR)
        rtp = props2.next
        rt_props = RTPipelineProperties(
            rtp.shader_group_handle_size,
            rtp.shader_group_base_alignment,
            rtp.shader_group_handle_alignment,
            rtp.max_ray_recursion_depth,
            rtp.max_ray_hit_attribute_size,
        )
    end

    # Query AS scratch alignment once and cache on the context — used by
    # `bda_alignment_for(::VkContext, scratch::Bool)` to pick the right
    # alignment for `LavaArray(...; scratch=true)` allocations.  Vulkan has no
    # usage flag for "AS scratch", so callers signal via the `scratch` kwarg.
    as_scratch_align = UInt64(1)
    if has_rt
        as_props2 = Vulkan.get_physical_device_properties_2(phys_dev,
            Vulkan.PhysicalDeviceAccelerationStructurePropertiesKHR)
        as_scratch_align = max(UInt64(1),
            UInt64(as_props2.next.min_acceleration_structure_scratch_offset_alignment))
    end

    has_validation = !isempty(layers)
    if has_rt
        @info "Lava: initialized Vulkan device with RT" device=dev_name queue_family=qf_idx handle_size=rt_props.shader_group_handle_size max_recursion=rt_props.max_ray_recursion_depth validation=has_validation gpu_assisted=gpu_assisted sync_val=sync_val debug_utils=has_debug_utils
    else
        @info "Lava: initialized Vulkan device (no RT)" device=dev_name queue_family=qf_idx validation=has_validation gpu_assisted=gpu_assisted sync_val=sync_val debug_utils=has_debug_utils
    end
    if !has_validation && want_validation
        @warn "Vulkan validation layers not found. Install vulkan-validationlayers for GPU error diagnostics."
    end

    # Clear validation messages accumulated during device creation.
    # GPU-assisted validation emits harmless "adjusting settings" warnings during
    # vkCreateDevice that would otherwise block the first shader compilation.
    clear_validation_messages!()

    # Initialize zero-alloc Vulkan function pointers for hot paths
    CMD_PIPELINE_BARRIER_FPTR[] = Vulkan.function_pointer(device, "vkCmdPipelineBarrier")

    mem_props = Vulkan.get_physical_device_memory_properties(phys_dev)
    phys_props = Vulkan.get_physical_device_properties(phys_dev)
    wgc = phys_props.limits.max_compute_work_group_count
    max_wg = (Int(wgc[1]), Int(wgc[2]), Int(wgc[3]))

    # VkContext's inner constructor builds its own default_bq via `new()`-
    # based two-phase init.  Pass the raw primary queue and all other ctx
    # fields; the ctor wires BatchQueue(device, queue, qfi, ctx) internally.
    # The hardware implements a fixed set of (M, N, K, dtype) tiles; a kernel
    # must choose one of these, it cannot pick an arbitrary tile size.
    CMShape = eltype(fieldtype(VkContext, :coopmat_shapes))
    coopmat_shapes = CMShape[]
    if has_coopmat
        try
            for p in unwrap(Vulkan.get_physical_device_cooperative_matrix_properties_khr(phys_dev))
                push!(coopmat_shapes, (M=Int(p.m_size), N=Int(p.n_size), K=Int(p.k_size),
                                       ab_type=UInt32(p.a_type), c_type=UInt32(p.c_type),
                                       scope=UInt32(p.scope)))
            end
        catch err
            @debug "cooperative-matrix property query failed; treating as unsupported" err
        end
    end

    ctx = VkContext(
        instance, phys_dev, device, qf_idx, dev_name,
        queue, compute_queue,
        rt_props, debug_messenger,
        2, n_queues,  # next_queue_index=2 (0=primary, 1=compute), max=n_queues
        async_qf_idx, async_n_queues,
        false,        # device_lost (fresh context)
        mem_props, max_wg,
        as_scratch_align,
        has_ray_query,
        has_ser,
        has_coopmat,
        coopmat_shapes,
        has_memory_budget,
        has_external_memory,
        gpu_assisted,
        string(phys_props.driver_version),
        has_video_decode, video_decode_queue, video_qf_idx,
    )
    return ctx
end

"""
    allocate_batch_queue!() -> BatchQueue

Create a new independent BatchQueue on a separate Vulkan queue (if available).
Falls back to a separate command pool on the primary queue if all queues are taken.
Used by Screen for isolated graphics rendering.
"""
function allocate_batch_queue!()
    ctx = vk_context()
    allocate_batch_queue!(ctx)
end

function allocate_batch_queue!(ctx::VkContext)
    idx = ctx.next_queue_index
    if idx < ctx.max_queue_count
        queue = Vulkan.get_device_queue(ctx.device, ctx.queue_family_index, UInt32(idx))
        ctx.next_queue_index += 1
    else
        # All hardware queues taken — reuse primary queue with separate command pool
        queue = ctx.default_bq.queue
    end
    return BatchQueue(ctx.device, queue, ctx.queue_family_index, ctx)
end

function pick_physical_device(devs)
    # Prefer discrete GPU
    for dev in devs
        props = Vulkan.get_physical_device_properties(dev)
        if props.device_type == Vulkan.PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            return dev
        end
    end
    # Fall back to first available
    return first(devs)
end

function find_graphics_compute_queue_family(phys_dev)
    qf_props = Vulkan.get_physical_device_queue_family_properties(phys_dev)
    # Prefer graphics+compute (needed for graphics pipeline support)
    for (i, qfp) in enumerate(qf_props)
        if (qfp.queue_flags & Vulkan.QUEUE_COMPUTE_BIT) != 0 &&
           (qfp.queue_flags & Vulkan.QUEUE_GRAPHICS_BIT) != 0
            return UInt32(i - 1)
        end
    end
    # Fall back to any compute-capable queue (graphics won't work but compute will)
    for (i, qfp) in enumerate(qf_props)
        if (qfp.queue_flags & Vulkan.QUEUE_COMPUTE_BIT) != 0
            return UInt32(i - 1)
        end
    end
    throw(LavaError(
        "device initialization",
        "No compute-capable queue family found",
        "Ensure your GPU supports Vulkan compute"))
end

"""
Find a secondary queue family distinct from `primary_qf_idx` that supports
compute + transfer. On RDNA/RADV the primary family is graphics+compute with
1 queue; the async family is compute-only with 4 queues — ideal for the
upload/dispatch split. Returns `nothing` if no such family exists.
"""
function find_async_compute_queue_family(phys_dev, primary_qf_idx::UInt32)
    qf_props = Vulkan.get_physical_device_queue_family_properties(phys_dev)
    for (i, qfp) in enumerate(qf_props)
        family = UInt32(i - 1)
        family == primary_qf_idx && continue
        if (qfp.queue_flags & Vulkan.QUEUE_COMPUTE_BIT) != 0 &&
           (qfp.queue_flags & Vulkan.QUEUE_TRANSFER_BIT) != 0
            return family
        end
    end
    return nothing
end

"""
Find a queue family that supports hardware video decode
(`VK_QUEUE_VIDEO_DECODE_BIT_KHR`, bit 0x20 — not surfaced as a named flag in
Vulkan.jl). Dedicated on NVIDIA (transfer + video-decode, no graphics/compute).
Returns `nothing` if the device has none.
"""
function find_video_decode_queue_family(phys_dev)
    qf_props = Vulkan.get_physical_device_queue_family_properties(phys_dev)
    for (i, qfp) in enumerate(qf_props)
        if (UInt32(qfp.queue_flags) & 0x00000020) != 0
            return UInt32(i - 1)
        end
    end
    return nothing
end

"""Check if the physical device supports the RT extensions we need."""
function has_extension(phys_dev, ext_name::String)
    available = unwrap(Vulkan.enumerate_device_extension_properties(phys_dev))
    for ext in available
        name = String(filter(!=('\0'), collect(ext.extension_name)))
        name == ext_name && return true
    end
    return false
end

function has_rt_extensions(phys_dev)
    available = unwrap(Vulkan.enumerate_device_extension_properties(phys_dev))
    names = Set{String}()
    for ext in available
        push!(names, String(filter(!=('\0'), collect(ext.extension_name))))
    end
    return "VK_KHR_acceleration_structure" in names &&
           "VK_KHR_ray_tracing_pipeline" in names &&
           "VK_KHR_deferred_host_operations" in names
end

# ── Validation layer debug messenger ──

# Async-safe Vulkan debug callback.  MUST NOT allocate, log, take locks, or
# hit a scheduler yield/safepoint — it can run re-entrantly inside a blocking
# driver ccall (GPU-AV readback during vkWaitSemaphores) with driver locks
# held, where any of those deadlocks the runtime.  All it does is copy the
# message bytes into the preallocated ring via raw ccalls + @inbounds stores.
# Classification + logging happens later on the main thread in
# `drain_validation_messages!`.  The flag args are received as raw UInt32 (the
# C ABI for VkDebugUtilsMessageSeverity/TypeFlags) so no flag-wrapper is built.
function debug_callback(
    severity::UInt32,
    type::UInt32,
    p_callback_data::Ptr{Vulkan.VkCore.VkDebugUtilsMessengerCallbackDataEXT},
    p_user_data::Ptr{Cvoid},
)
    p_callback_data == C_NULL && return UInt32(0)
    data = unsafe_load(p_callback_data)        # isbits C struct → stack value, no heap alloc
    msg_ptr = Ptr{UInt8}(data.pMessage)
    idx  = @inbounds VAL_RING_WRITE[1]
    slot = idx % VAL_RING_SLOTS
    dst  = pointer(VAL_RING_BUF) + slot * VAL_RING_SLOT_BYTES
    n = 0
    if msg_ptr != C_NULL
        n = Int(ccall(:strlen, Csize_t, (Ptr{UInt8},), msg_ptr))
        n > VAL_RING_SLOT_BYTES - 1 && (n = VAL_RING_SLOT_BYTES - 1)
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, msg_ptr, n % Csize_t)
    end
    @inbounds VAL_RING_LEN[slot + 1]  = Cint(n)
    @inbounds VAL_RING_SEV[slot + 1]  = severity
    @inbounds VAL_RING_TYPE[slot + 1] = type
    @inbounds VAL_RING_WRITE[1] = idx + 1
    # Return VK_FALSE — can't throw from a @cfunction callback (would corrupt
    # Vulkan state).  Errors are surfaced via drain_validation_messages!.
    return UInt32(0)
end

# Classify one drained message and, for hard validation errors, capture it.
# Runs ONLY on the main thread (called from drain), so logging/alloc is safe.
function _handle_validation_message(message::String, sev_u::UInt32, type_u::UInt32)
    # Message-type bits classify the source (Vulkan spec):
    #   VALIDATION (0x2) — VK_LAYER_KHRONOS_validation + driver spec checks
    #   GENERAL    (0x1) — driver runtime notes (not spec checks)
    #   PERFORMANCE (0x4), DEVICE_ADDRESS_BINDING (0x8) — self-explanatory
    # Only VALIDATION-typed errors are captured as hard failures.
    is_error = (sev_u & UInt32(Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) != 0
    is_warning = (sev_u & UInt32(Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) != 0
    is_validation = (type_u & UInt32(Vulkan.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) != 0
    # Known-benign device-creation chatter that must NOT be captured as a hard
    # validation error.  The "Device Layers have never worked" notice is a VVL
    # deprecation (VUID-VkDeviceCreateInfo-ppEnabledLayerNames-12385) triggered
    # because Vulkan.jl's generated `_DeviceCreateInfo` always passes a non-null
    # `ppEnabledLayerNames` even for an empty layer list (count is 0, so it is
    # functionally ignored by the loader).  It is a dependency-side quirk with
    # no runtime effect; log it as a warning but don't treat it as a failure.
    is_setup_noise = contains(message, "adjusting settings") ||
                     contains(message, "VALIDATION-SETTINGS") ||
                     contains(message, "Device Layers have never worked")
    # @lava_printf output arrives at INFO severity. The Khronos layer wraps it as
    #   "vkQueueSubmit2(): pSubmits[0] DebugPrintf:\n<user text>"
    # Capture just the user text into PRINTF_MESSAGES, separate from validation.
    if contains(message, "DebugPrintf")
        marker = findlast("DebugPrintf:", message)
        text = marker === nothing ? message :
               lstrip(message[last(marker) + 1 : end], ['\n', '\r', ' '])
        if length(PRINTF_MESSAGES) >= MAX_PRINTF_MESSAGES
            popfirst!(PRINTF_MESSAGES)
        end
        push!(PRINTF_MESSAGES, String(text))
        @info "lava_printf" text
        return nothing
    end
    if is_error && is_validation && !is_setup_noise
        if length(VALIDATION_MESSAGES) >= MAX_VALIDATION_MESSAGES
            popfirst!(VALIDATION_MESSAGES)
        end
        push!(VALIDATION_MESSAGES, message)
        @error "Vulkan validation error" message
    elseif is_error && !is_validation
        # Driver-general error (e.g. chatty lavapipe SPIR-V notes). Log but
        # don't capture — the authoritative signal is the VkResult of the
        # surrounding API call, which @vk_checked already unwraps.
        @warn "Vulkan driver note (type=$(type_u), not a validation error)" message
    elseif is_warning || is_setup_noise
        @warn "Vulkan validation warning" message
    end
    return nothing
end

"""
    drain_validation_messages!()

Convert any messages the async callback wrote into the ring into Strings,
capture hard validation errors in `VALIDATION_MESSAGES`, and log them. Must be
called on the main thread (it allocates + logs). Idempotent — only processes
slots written since the last drain. Call this before reading
`VALIDATION_MESSAGES` and while polling for a GPU-AV fault.
"""
function drain_validation_messages!()
    write_idx = @inbounds VAL_RING_WRITE[1]
    read_idx  = VAL_RING_READ[]
    # If the callback lapped us, skip the slots it overwrote.
    if write_idx - read_idx > VAL_RING_SLOTS
        read_idx = write_idx - VAL_RING_SLOTS
    end
    while read_idx < write_idx
        slot = read_idx % VAL_RING_SLOTS
        n    = Int(@inbounds VAL_RING_LEN[slot + 1])
        sev  = @inbounds VAL_RING_SEV[slot + 1]
        typ  = @inbounds VAL_RING_TYPE[slot + 1]
        message = n == 0 ? "(no message)" :
            unsafe_string(pointer(VAL_RING_BUF) + slot * VAL_RING_SLOT_BYTES, n)
        _handle_validation_message(message, sev, typ)
        read_idx += 1
    end
    VAL_RING_READ[] = write_idx
    return nothing
end

function setup_debug_messenger(instance::Vulkan.Instance)
    callback_ptr = @cfunction(
        debug_callback,
        UInt32,
        (UInt32,
         UInt32,
         Ptr{Vulkan.VkCore.VkDebugUtilsMessengerCallbackDataEXT},
         Ptr{Cvoid})
    )

    # Debug printf is delivered at INFO severity, so subscribe down to INFO when
    # it's enabled; otherwise stay at WARNING to avoid info-level chatter.
    min_sev = get(ENV, "LAVA_DEBUG_PRINTF", "0") != "0" ?
        Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT :
        Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
    messenger = Vulkan.DebugUtilsMessengerEXT(
        instance,
        callback_ptr;
        min_severity=min_sev,
    )
    return messenger
end

"""
    get_validation_messages() -> Vector{String}

Return recent validation layer messages. Useful for diagnosing DEVICE_LOST errors.
"""
get_validation_messages() = (drain_validation_messages!(); copy(VALIDATION_MESSAGES))

"""
    clear_validation_messages!()

Clear the validation message buffer. Drains the async ring first (so captured
hard errors are logged), then empties the capture list.
"""
clear_validation_messages!() = (drain_validation_messages!(); empty!(VALIDATION_MESSAGES))

"""
    get_printf_output() -> Vector{String}

Return captured `@lava_printf` output since the last clear. Drains the async
callback ring first. Requires `enable_debug_printf!()` to have been called.
"""
get_printf_output() = (drain_validation_messages!(); copy(PRINTF_MESSAGES))

"""
    clear_printf_output!()

Drop captured `@lava_printf` output (drains the ring first).
"""
clear_printf_output!() = (drain_validation_messages!(); empty!(PRINTF_MESSAGES))

"""
    check_validation_errors!(context::String)

Check if any validation errors were captured since the last check.
Throws `LavaError` with the error messages if any errors are found.
Call this after Vulkan operations that may trigger validation errors
(shader module creation, pipeline creation, dispatch recording).
"""
function check_validation_errors!(context::String)
    drain_validation_messages!()
    isempty(VALIDATION_MESSAGES) && return
    # Separate true errors from warnings using the severity recorded by the callback.
    # GPU-AV messages like "Unaligned pointer access" are errors, not warnings,
    # even though their text may contain strings like "WARNING-Validation".
    # We rely on the callback storing only ERROR-severity messages.
    errors = copy(VALIDATION_MESSAGES)
    isempty(errors) && return
    n = min(length(errors), 5)
    detail = join(["  [$i] $(first(errors[i], 1000))" for i in 1:n], "\n")
    # Clear after reporting to avoid re-triggering
    empty!(VALIDATION_MESSAGES)
    throw(LavaError(
        context,
        "Vulkan validation error(s):\n$detail",
        "Fix the validation errors above before proceeding."
    ))
end
