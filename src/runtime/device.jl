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
When the number of dispatches in the current CB segment exceeds `bq.cb_split_threshold`,
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
    # Timeline value the GPU must reach before the pool may be rewound: the
    # newest batch that allocated from it. 0 = nothing outstanding. Rewinding
    # earlier hands the next caller bytes an in-flight shader is still reading,
    # since a dispatch's arg address is baked into its command buffer as a push
    # constant (see `arg_pool_in_use!`).
    arg_pool_frontier::UInt64
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

    # ── Recording policy. These were six module-level `Ref`s, which made them
    # process-wide settings for something that is per queue: two BatchQueues on
    # one device already disagree about how much work to batch before submitting,
    # and a second device made it worse. They are still mutable defaults — that
    # is what they are for — but they are now this queue's.
    #
    # `auto_submit_threshold` at 64 rather than 0 is the +44% measured in
    # `perf-plan.md`: at 0, recording and execution never overlapped.
    auto_submit_threshold::Int
    cb_split_threshold::Int
    flush_timeout_ns::UInt64
    barrier_mode::Symbol
    barrier_elision::Bool
    # One-shot, consumed by exactly the next dispatch on THIS queue.
    next_skip_barrier::Bool
    # Set by the KA launch path for the dispatch it is about to record: "this
    # dispatch enumerated its buffers, so the elision tracker saw everything it
    # touches". Same one-shot shape as `next_skip_barrier`, and it was a global
    # for the same reason — the hand-off is launch → `record_dispatch!` and both
    # already have the queue.
    ranges_declared::Bool
    # The elision tracker itself. `touched_ranges` accumulates what recent
    # dispatches in the current batch wrote; `dispatch_ranges` is scratch for the
    # dispatch being recorded. Reused, never reallocated — and per queue, because
    # two queues recording concurrently into their own command buffers were
    # sharing one tracker, so a range written on one could elide a barrier on the
    # other.
    touched_ranges::Vector{UInt64}
    dispatch_ranges::Vector{UInt64}
    # Non-`nothing` inside `concurrent_indirect_group`: dispatches append here
    # instead of recording, and the group's flush fuses them.
    deferred_indirect::Union{Nothing,Vector{Any}}
    # The `CapturedSequence` being recorded on THIS queue, or `nothing`. It was a
    # module-level `Ref`, so every site that used it had to re-check `cap.bq ===
    # bq` to find out whether the capture was even this queue's — and
    # `cb_begin_flags` had no queue to check with, so a capture running on one
    # queue silently made every OTHER queue's command buffers reusable.
    # `Any` because `CapturedSequence` is declared in `command.jl`; use sites
    # assert it, the same shape as `ctx`.
    capturing::Any
    # The high-water mark of arg slabs a capture reserved on this queue, so a
    # replay does not overwrite arguments a recorded pipeline still points at.
    reserved_arg_slabs::Int
    # Highest timeline value signalled by a replay on this queue. `flush!` has to
    # wait on it: a replay puts no `CommandBatch` in `in_flight`, so the in-flight
    # scan alone would return before the GPU had run any of it. Was an
    # `IdDict{BatchQueue,UInt64}` — a per-queue value in a process-wide dict keyed
    # by the queue, which is the surrogate a field replaces.
    replay_watermark::UInt64
    # What the last dispatch on this queue was, for the dispatch log and for
    # DEVICE_LOST diagnostics. Process-wide, these attributed one queue's crash
    # to another queue's kernel.
    last_dispatch_info::String
    prev_dispatch_info::String
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
                    Any[], 1, 0, 0, UInt64(0),  # arg_slabs: idx=1, offset=0, count=0, frontier=0
                    Any[], 1, 0,     # indirect_slabs: idx=1, offset=0
                    nothing,         # staging (lazy)
                    ctx,             # owning VkContext (required)
                    Threads.threadid(),  # owning_thread
                    64, 3000, UInt64(120) * 1_000_000_000,  # auto-submit, CB split, flush timeout
                    :memory, false, false,                  # barrier mode / elision / one-shot skip
                    false, UInt64[], UInt64[],              # ranges_declared + elision tracker
                    nothing, nothing, 0, UInt64(0),         # deferred indirect, capture, reserved slabs, watermark
                    "", "")                                 # last / prev dispatch info
    # Plug the back-reference into every pre-allocated batch so `batch.bq`
    # is non-nothing as soon as the bq is returned.  Future batches allocated
    # lazily (alloc_cmd_buf → init_batch) must set .bq themselves.
    for b in batches
        b.bq = bq
    end
    return bq
end

"""
    CoopMat2Caps

The sub-features of `VK_NV_cooperative_matrix2`, as enabled on this device.
`available` is false when the extension itself is missing, in which case every
other field is false too — so a kernel may test the single field it needs.

The extension is NVIDIA-only. On AMD (RDNA3 exposes WMMA through the portable
`VK_KHR_cooperative_matrix`) all of these are false and kernels must fall back
to the plain KHR path, which is the one that ships today.

| field | what it adds | who wants it |
|:--|:--|:--|
| `workgroup_scope` | matrices spanning the whole workgroup, not one subgroup | wider GEMM tiles |
| `flexible_dimensions` | M/N/K free of the driver's fixed shape list | `E = 72` without padding to 80 |
| `reductions` | row/column reduce on an accumulator in place | flash softmax row max/sum |
| `conversions` | element conversion without a round trip through memory | mixed-precision staging |
| `per_element_operations` | a callback per element, given `(row, col)` | the flash rescale of a held `O` |
| `tensor_addressing` | a tensor descriptor drives the addressing | hand-coded root+stride staging |
| `block_loads` | load a tile straight from a descriptor | ditto |
"""
struct CoopMat2Caps
    available::Bool
    workgroup_scope::Bool
    flexible_dimensions::Bool
    reductions::Bool
    conversions::Bool
    per_element_operations::Bool
    tensor_addressing::Bool
    block_loads::Bool
end

CoopMat2Caps() = CoopMat2Caps(false, false, false, false, false, false, false, false)

"""
    DeviceCompute

How much compute the device actually has, for kernels that must size a launch
against it rather than against the problem.

A grid that does not fill the device is the dominant cost on small shapes — SAM
2's decode runs one attention at 8 workgroups on 48 SMs — so "how many
workgroups before the device is busy" is a number kernels need, and until this
struct existed they hardcoded it.

`sm_count` and `warps_per_sm` are **0 when the device does not report them**.
Vulkan has no core query: NVIDIA exposes it through `VK_NV_shader_sm_builtins`,
AMD through `VK_AMD_shader_core_properties2`, and everyone else through nothing
at all. Use [`shader_core_count`](@ref) rather than this field — it returns
`nothing` for unknown, so a missing fallback is a `MethodError` at the first
arithmetic rather than a silently empty grid.

`max_shared_memory` is a core limit and is always known.
"""
struct DeviceCompute
    sm_count::Int           # 0 = the device does not report it
    warps_per_sm::Int       # 0 = ditto; the occupancy denominator when known
    max_shared_memory::Int  # VkPhysicalDeviceLimits::maxComputeSharedMemorySize
end

DeviceCompute() = DeviceCompute(0, 0, 0)

"""
    query_device_compute(phys_dev, phys_props) -> DeviceCompute

Read the SM/CU count once, at device creation.

Vendor coverage follows llama.cpp's `ggml-vulkan.cpp:6138-6146`, which is the
same question asked by the same kind of code. Neither extension is *enabled* on
the logical device — these are physical-device property queries and enabling
would be pointless — but the chained struct is only filled for an extension the
device advertises, so each is guarded by `has_extension`.

The chain is also only filled when the instance opted into
`vkGetPhysicalDeviceProperties2`: core in Vulkan 1.1, or the
`VK_KHR_get_physical_device_properties2` instance extension before that. Lava's
instance asks for 1.4 (see `create_vulkan_context`), so this holds — but note the
failure is silent if it ever stops holding. The base properties still come back
fully populated and only the `next` chain is dropped, with no return code to
check, because `vkGetPhysicalDeviceProperties2` returns void.
"""
function query_device_compute(phys_dev, phys_props)
    smcount, warps = 0, 0
    if has_extension(phys_dev, "VK_NV_shader_sm_builtins")
        try
            p = Vulkan.get_physical_device_properties_2(
                    phys_dev, Vulkan.PhysicalDeviceShaderSMBuiltinsPropertiesNV).next
            smcount, warps = Int(p.shader_sm_count), Int(p.shader_warps_per_sm)
        catch err
            @debug "VK_NV_shader_sm_builtins query failed; SM count unknown" err
        end
    elseif has_extension(phys_dev, "VK_AMD_shader_core_properties2")
        try
            p = Vulkan.get_physical_device_properties_2(
                    phys_dev, Vulkan.PhysicalDeviceShaderCoreProperties2AMD).next
            smcount = Int(p.active_compute_unit_count)
        catch err
            @debug "VK_AMD_shader_core_properties2 query failed; CU count unknown" err
        end
    end
    DeviceCompute(smcount, warps,
                  Int(phys_props.limits.max_compute_shared_memory_size))
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
    # The messenger's ring, and its `pUserData`. Per context because the
    # messenger is — see `ValidationRing`.
    validation::ValidationRing
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
    # SM/CU count, warps per SM, shared-memory ceiling. Read once at creation;
    # see `DeviceCompute` and prefer the `shader_core_count` accessor.
    compute::DeviceCompute
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
    # VK_NV_cooperative_matrix2, per sub-feature. NVIDIA-only; every kernel that
    # branches on one of these must keep a path that runs with all of them false,
    # because that is what AMD (RDNA3 WMMA via the plain KHR extension) gets.
    coopmat2::CoopMat2Caps
    # VK_NV_cooperative_vector: matrix x vector at subgroup scope, for the shapes
    # where a cooperative *matrix* would waste most of its tile (`Lq = 23`).
    # NVIDIA-only.
    coopvec_available::Bool
    # VK_KHR_shader_maximal_reconvergence + VK_KHR_shader_subgroup_uniform_control_flow.
    # Together they make a spin-wait on a shared flag well-defined, which is how
    # a producer/consumer (warp-specialised) kernel is written without Vulkan
    # having named barriers. KHR, so RDNA3 has them too.
    maximal_reconvergence_available::Bool
    subgroup_uniform_control_flow_available::Bool
    # VK_KHR_shader_subgroup_rotate: OpGroupNonUniformRotateKHR, a shuffle by a
    # subgroup-uniform delta. KHR, promoted to Vulkan 1.4.
    subgroup_rotate_available::Bool
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
    # `vkCmdPipelineBarrier`, resolved for THIS device.
    #
    # Device function pointers are per device — `vkGetDeviceProcAddr` returns a
    # pointer valid only for the device it was asked about. This was a module
    # global (`CMD_PIPELINE_BARRIER_FPTR`), so creating a second context
    # overwrote it, and the FIRST device's command buffers were then recorded
    # through the SECOND device's driver.
    #
    # That is an immediate segfault rather than a wrong answer, which makes it
    # the most dangerous piece of module-scope device state in the library and
    # the one `GUARDRAILS.md` §8 did not list — it names four caches holding
    # handles and does not mention the function table. Found the first time two
    # contexts existed at once: the crash was inside `libvulkan_lvp.so` while
    # dispatching on the NVIDIA context.
    cmd_pipeline_barrier_fptr::Ptr{Nothing}
    # A stable identity for this device, for as long as it exists.
    #
    # Four caches hold device-owned Vulkan handles at module scope and were keyed
    # without the device, so two devices running the same kernel produced the
    # SAME key and the second got the first's `VkPipeline` — bound into a command
    # buffer on a different `VkDevice`, which is undefined behaviour
    # (`GUARDRAILS.md` §8). This is the missing part of every one of those keys.
    #
    # A counter rather than `objectid(ctx)`: object ids are reused after the
    # collector reclaims, and a *reused* id is exactly the failure this exists to
    # prevent — a fresh device silently inheriting a dead one's pipelines. It
    # also survives `vk_reset_device!`, which builds a new context, so entries
    # from before a reset can never be handed to after it.
    #
    # It survives as an identity — for logging, and for the probe's assertion
    # that two contexts are distinct — but it is no longer a cache key. It was
    # one, and this comment used to argue that keying was as good as ownership
    # because the cached value types are defined in files included after this
    # one, so a field would have to be `Any` and cost inference on a lookup per
    # dispatch. The premise was true and the conclusion was wrong: the fix is to
    # move the nine type definitions ahead of this file (`coretypes.jl`), not to
    # accept a surrogate key. `caches` below is concrete.
    id::UInt64

    # Per-device state, owned by the device. See `DeviceCaches` for what was
    # global before and why keying it by `id` was not the same thing: a field
    # dies with its context, so nothing outlives the handles it describes and
    # `vk_reset_device!` has nothing to clear.
    caches::DeviceCaches

    # Every debugging and instrumentation toggle, owned by the device it
    # instruments. Was eighteen module-level `Ref`s; see `Diagnostics`.
    diag::Diagnostics

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
                       validation::ValidationRing,
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
                       coopmat2::CoopMat2Caps=CoopMat2Caps(),
                       coopvec_available::Bool=false,
                       maximal_reconvergence_available::Bool=false,
                       subgroup_uniform_control_flow_available::Bool=false,
                       subgroup_rotate_available::Bool=false,
                       memory_budget_available::Bool=false,
                       external_memory_available::Bool=false,
                       gpu_assisted::Bool=false,
                       driver_version::AbstractString="unknown",
                       video_decode_available::Bool=false,
                       video_decode_queue::Union{Nothing, Vulkan.Queue}=nothing,
                       video_decode_queue_family_index::Union{Nothing, UInt32}=nothing,
                       compute::DeviceCompute=DeviceCompute())
        ctx = new()
        ctx.instance = instance
        ctx.physical_device = physical_device
        ctx.device = device
        ctx.queue_family_index = queue_family_index
        ctx.device_name = device_name
        ctx.compute_queue = compute_queue
        ctx.rt_pipeline_properties = rt_pipeline_properties
        ctx.debug_messenger = debug_messenger
        ctx.validation = validation
        ctx.next_queue_index = next_queue_index
        ctx.max_queue_count = max_queue_count
        ctx.async_queue_family_index = async_queue_family_index
        ctx.async_queue_count = async_queue_count
        ctx.device_lost = device_lost
        ctx.memory_properties = memory_properties
        ctx.max_wg_dims = max_wg_dims
        ctx.compute = compute
        ctx.as_scratch_align = as_scratch_align
        ctx.ray_query_available = ray_query_available
        ctx.ser_available = ser_available
        ctx.coopmat_available = coopmat_available
        ctx.coopmat_shapes = coopmat_shapes === nothing ?
            eltype(fieldtype(VkContext, :coopmat_shapes))[] : coopmat_shapes
        ctx.coopmat2 = coopmat2
        ctx.coopvec_available = coopvec_available
        ctx.maximal_reconvergence_available = maximal_reconvergence_available
        ctx.subgroup_uniform_control_flow_available = subgroup_uniform_control_flow_available
        ctx.subgroup_rotate_available = subgroup_rotate_available
        ctx.memory_budget_available = memory_budget_available
        ctx.external_memory_available = external_memory_available
        ctx.video_decode_available = video_decode_available
        ctx.video_decode_queue = video_decode_queue
        ctx.video_decode_queue_family_index = video_decode_queue_family_index
        ctx.gpu_assisted = gpu_assisted
        ctx.driver_version = driver_version
        # Seed a persistent VkPipelineCache from disk (if any). The header is
        # validated against this physical device before the driver sees it —
        # `vkCreatePipelineCache` is not a safe place to discover a mismatch.
        ctx.id = (VK_CONTEXT_COUNTER[] += 1)
        ctx.caches = DeviceCaches()
        ctx.diag = Diagnostics()
        # Filled by `init_vulkan!` once the device exists; null until then so a
        # barrier recorded before that point is skipped rather than jumping to
        # whatever the field happened to contain.
        ctx.cmd_pipeline_barrier_fptr = C_NULL
        ctx.pipeline_cache = create_lava_pipeline_cache(
            device, lava_pipeline_cache_path(device_name, driver_version), physical_device)
        _register_pipeline_cache_atexit!()
        # Now build the default BatchQueue with the live ctx.  Sets the
        # remaining field; no nullable slot, no post-hoc mutation.
        ctx.default_bq = BatchQueue(device, primary_queue, queue_family_index, ctx)
        return ctx
    end
end

"""
    shader_core_count(ctx = vk_context()) -> Union{Nothing,Int}

Streaming multiprocessors (NVIDIA) or active compute units (AMD), or `nothing`
when the device does not report it.

`nothing` rather than `0` on purpose. The number's whole job is to be a
denominator — llama.cpp derives its flash-attention split count as
`shader_core_count * 2 / total_workgroups` — and a `0` there yields zero splits
silently, which is a wrong launch that still produces a plausible-looking answer.
`nothing` cannot be divided, so an unguarded use fails immediately.

Supply the fallback explicitly:

    cores = something(shader_core_count(), 16)
"""
shader_core_count(ctx::VkContext = vk_context()) =
    ctx.compute.sm_count == 0 ? nothing : ctx.compute.sm_count

"""
    shader_warps_per_sm(ctx = vk_context()) -> Union{Nothing,Int}

Maximum resident subgroups per SM — the denominator for an occupancy figure.
`nothing` when unreported; NVIDIA-only in practice (`VK_NV_shader_sm_builtins`).
"""
shader_warps_per_sm(ctx::VkContext = vk_context()) =
    ctx.compute.warps_per_sm == 0 ? nothing : ctx.compute.warps_per_sm

"""
    max_shared_memory(ctx = vk_context()) -> Int

`maxComputeSharedMemorySize`: the per-workgroup shared-memory ceiling, in bytes.
Core Vulkan, so this one is always a real number — a kernel that sizes its
`@localmem` against a budget should read it here rather than assume 48 KB.
"""
max_shared_memory(ctx::VkContext = vk_context()) = ctx.compute.max_shared_memory

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
# `drain_validation_messages!(ctx)` to turn raw slots into Strings, capture the
# hard errors in `ctx.validation.messages`, and log everything. See
# `ValidationRing` for why the arrays live on the context and how the callback
# reaches them.

const VK_CONTEXT_REF = Ref{Union{Nothing, VkContext}}(nothing)
# Guards the lazy init below. `init_vulkan!` builds a whole VkDevice and has no
# idempotency guard of its own, so two threads arriving together built TWO.
const VK_CONTEXT_LOCK = ReentrantLock()

"""
    device_lost(ctx::VkContext)  ->  Bool

Whether this context's device has been marked lost. Prefer passing `ctx`
explicitly; the no-arg form looks up the current default context.
"""
device_lost(ctx::VkContext) = ctx.device_lost
device_lost() = let ctx = VK_CONTEXT_REF[]
    ctx === nothing ? false : ctx.device_lost
end

"""
Mark `ctx`'s device as lost. All subsequent finalizers will skip Vulkan calls.

Two things reach this state, and only the first is a fault:

 1. `ERROR_DEVICE_LOST` from any Vulkan call — see `mark_if_device_lost!`.
 2. **Retirement**: a context nobody will call into again, whose buffers are
    still alive in Julia and whose finalizers must therefore not touch it.
    `vk_reset_device!` retires the context it replaces; anything that builds a
    context of its own with `init_vulkan!(; select)` owns retiring it.

The flag means the same thing to every consumer either way — *do not call into
this device* — which is why the second case reuses it rather than adding a
parallel one. A context that is dropped without being retired hands its
finalizers a device that Vulkan.jl's own handle finalizers may already have
destroyed, and the crash lands inside the driver at whatever GC runs next.
"""
mark_device_lost!(ctx::VkContext) = (ctx.device_lost = true; nothing)

"""
Hands out `VkContext.id`. Monotonic and never reused, so a device destroyed and
recreated cannot inherit cache entries keyed to its predecessor.

Nothing in `src/` reads `ctx.id` any more — the dictionaries it keyed are
`VkContext` fields now, which is what made it redundant. It survives as a cheap
identity for diagnostics and for `twodevice_probe.jl`'s "two contexts must not
share an id" assertion, and this counter with it.
"""
const VK_CONTEXT_COUNTER = Ref{UInt64}(0)

# `RESET_CALLBACKS` was here: a list every later-included file pushed onto so
# `vk_reset_device!` could empty its module-level caches. Deleted rather than
# emptied — its entries were the symptom this refactor was diagnosing. State
# that outlives the device it describes has to be told to go away; state a
# `VkContext` owns simply does not. The last four went with the pool accounting,
# the dispatch counters, the capture handle and the timing records.

"""
    vk_context() -> VkContext

Get or create the global Vulkan context. Lazily initializes on first call.
"""
function vk_context()
    ctx = VK_CONTEXT_REF[]
    ctx === nothing || return ctx::VkContext
    # Double-checked under a lock. Unlocked, two threads both saw `nothing` and
    # both ran `init_vulkan!`, leaving two live VkDevices: the loser's context is
    # still reachable from every buffer it allocated (`buf.last_write` retains
    # its BatchQueue), so the next cross-queue wait passed a semaphore from one
    # device to the other and the driver segfaulted with no Julia frame to show.
    lock(VK_CONTEXT_LOCK)
    try
        ctx = VK_CONTEXT_REF[]
        ctx === nothing || return ctx::VkContext
        # `invokelatest`, not a direct call: a direct one makes inference record
        # a backedge from `vk_context` to `init_vulkan!`, and `getproperty(::
        # LavaBackend, :bq)` goes through here, so EVERY Lava operation ends up
        # inferring through Vulkan initialisation. Anything that invalidates
        # `init_vulkan!` then invalidates the whole package — loading GLMakie
        # does exactly that (FreeType adds a `Base.unsafe_load` method), and it
        # cost 50 506 Lava MethodInstances and ~41 s of re-inference on the
        # first GPU call afterwards. The dynamic dispatch is paid once, on the
        # single call that creates the context.
        ctx = Base.invokelatest(init_vulkan!)::VkContext
        VK_CONTEXT_REF[] = ctx
        return ctx::VkContext
    finally
        unlock(VK_CONTEXT_LOCK)
    end
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
    # Retire the old context BEFORE dropping it. Pre-reset `VkManagedBuffer`s
    # hold a strong ref to it, and their finalizers gate every Vulkan call on
    # `device_lost` — so marking it here is what makes that gate true.
    #
    # It used to be assumed rather than set, and the assumption only held on the
    # path that *caused* it: a reset after `ERROR_DEVICE_LOST` finds the flag
    # already true, while a voluntary `vk_reset_device!()` left it false. Then
    # dropping the ref below made the old context garbage — and its buffers
    # garbage in the SAME collection, where Julia does not order finalizers. Run
    # the context's first and `Vulkan.Device`'s own finalizer destroys the
    # device; the buffer's `vk_free!` then calls `query_timeline` on it and the
    # driver takes a SIGSEGV inside `vkGetSemaphoreCounterValue`.
    #
    #     d = KA.allocate(LavaBackend(), Float32, 1000); fill!(d, 1f0)
    #     Lava.vk_reset_device!(); d = nothing; GC.gc()   # <- segfault
    #
    # A retired context is one nothing may call into: every array that predates
    # the reset holds memory belonging to a device that is gone, so there is no
    # case where skipping is wrong. `mark_device_lost!` is the same flag the
    # `VkResult` rule sets, and this is the second way to reach it.
    let old = VK_CONTEXT_REF[]
        if old !== nothing
            # Its blocks, while it is still the thing that owns them. This used
            # to be a reset callback walking a global `POOLS` dict.
            destroy_pool!(old)
            mark_device_lost!(old)
        end
    end
    VK_CONTEXT_REF[] = nothing
    # Don't destroy old Vulkan handles — they're invalid after DEVICE_LOST.
    # GC will eventually try to destroy them; _destroy_buffer! skips when
    # DEVICE_LOST was true (and we set it false only after clearing context).
    # Nothing to clear: the ring, its cursor and the drained messages are all
    # fields of the context being retired, and the new one brings a fresh
    # `ValidationRing` that its own messenger writes into.
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

"""
    init_vulkan!(; select = pick_physical_device) -> VkContext

Build a context. **Does not install it** as the global — `vk_context()` is what
does that, and it is the only caller that should.

`select` receives the enumerated physical devices and returns one, so a caller
can ask for a device other than the one `pick_physical_device` prefers. That is
what makes a two-device test possible on a single-GPU machine: the loader
enumerates the real GPU *and* lavapipe from one instance, so

    gpu = vk_context()
    cpu = init_vulkan!(select = devs -> only(filter(islavapipe, devs)))

gives two live contexts with two distinct `id`s, which is the pair every
per-device cache key has to be checked against (`GUARDRAILS.md` §8). Before this
there was no way to ask for the second device at all, so the acceptance test the
briefs describe could not be written.
"""
function init_vulkan!(; select = pick_physical_device)
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
    # Before the messenger, because the messenger is given its address.
    validation = ValidationRing()
    debug_messenger = nothing
    if has_debug_utils
        debug_messenger = setup_debug_messenger(instance, validation)
    end

    # Pick physical device (prefer discrete GPU)
    phys_devs = unwrap(Vulkan.enumerate_physical_devices(instance))
    isempty(phys_devs) && throw(LavaError(
        "device initialization",
        "No Vulkan-capable GPU found",
        "Ensure Vulkan drivers are installed"))

    phys_dev = select(phys_devs)
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
    # 64-bit atomics. These were hardcoded `false` below while the emitter happily
    # declared the `Int64Atomics` SPIR-V capability, so any kernel using one built a
    # module whose capability had no enabled feature behind it — undefined, and
    # reported by the validation layer as
    # VUID-VkShaderModuleCreateInfo-pCode-08740. Ask the device instead of
    # assuming, in both directions: enabling an unsupported feature fails device
    # creation, and leaving a supported one off is what caused this.
    has_int64_atomics = let
        q = Vulkan.get_physical_device_features_2(phys_dev,
                Vulkan.PhysicalDeviceVulkan12Features).next
        (buffer = q.shader_buffer_int_64_atomics, shared = q.shader_shared_int_64_atomics)
    end
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
    # VK_NV_cooperative_matrix2 layers per-element operations, row reductions,
    # flexible dimensions and tensor addressing onto the KHR matrices. Which of
    # its seven sub-features are actually on is read back below, after device
    # creation, rather than assumed from the extension being present.
    # NVIDIA-only; kernels keep their KHR path for everyone else.
    has_coopmat2 = has_coopmat && has_extension(phys_dev, "VK_NV_cooperative_matrix2")
    # Matrix x vector at subgroup scope. Wanted for the shapes where a
    # cooperative matrix wastes most of its tile. NVIDIA-only.
    has_coopvec = has_extension(phys_dev, "VK_NV_cooperative_vector")
    # A defined reconvergence point (KHR, and RDNA3 has it). Needed before a
    # kernel may spin on a shared flag, i.e. before producer/consumer staging.
    has_max_reconv = has_extension(phys_dev, "VK_KHR_shader_maximal_reconvergence")
    has_subgroup_ucf = has_extension(phys_dev, "VK_KHR_shader_subgroup_uniform_control_flow")
    # OpGroupNonUniformRotateKHR: shuffle by a subgroup-uniform delta, which is
    # the cheap form of a butterfly reduction. KHR, promoted to Vulkan 1.4.
    has_subgroup_rotate = has_extension(phys_dev, "VK_KHR_shader_subgroup_rotate")

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
    if has_coopmat2
        push!(extensions, "VK_NV_cooperative_matrix2")
    end
    if has_coopvec
        push!(extensions, "VK_NV_cooperative_vector")
    end
    if has_max_reconv
        push!(extensions, "VK_KHR_shader_maximal_reconvergence")
    end
    if has_subgroup_ucf
        push!(extensions, "VK_KHR_shader_subgroup_uniform_control_flow")
    end
    if has_subgroup_rotate
        push!(extensions, "VK_KHR_shader_subgroup_rotate")
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
        has_int64_atomics.buffer,  # shader_buffer_int_64_atomics ← the emitter declares Int64Atomics
        has_int64_atomics.shared,  # shader_shared_int_64_atomics ← ditto, for @localmem
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
    # Ask the driver which sub-features it actually has before requesting them:
    # `vkCreateDevice` fails outright if a feature struct turns on a bit the
    # device does not support, and an advertised extension does not imply
    # every bit. Everything below follows that shape.
    coopmat2_caps = CoopMat2Caps()
    if has_coopmat2
        q = Vulkan.get_physical_device_features_2(phys_dev,
            Vulkan.PhysicalDeviceCooperativeMatrix2FeaturesNV).next
        coopmat2_caps = CoopMat2Caps(
            true,
            q.cooperative_matrix_workgroup_scope,
            q.cooperative_matrix_flexible_dimensions,
            q.cooperative_matrix_reductions,
            q.cooperative_matrix_conversions,
            q.cooperative_matrix_per_element_operations,
            q.cooperative_matrix_tensor_addressing,
            q.cooperative_matrix_block_loads,
        )
        feature_chain = Vulkan.PhysicalDeviceCooperativeMatrix2FeaturesNV(
            coopmat2_caps.workgroup_scope,
            coopmat2_caps.flexible_dimensions,
            coopmat2_caps.reductions,
            coopmat2_caps.conversions,
            coopmat2_caps.per_element_operations,
            coopmat2_caps.tensor_addressing,
            coopmat2_caps.block_loads;
            next=feature_chain
        )
    end
    if has_coopvec
        q = Vulkan.get_physical_device_features_2(phys_dev,
            Vulkan.PhysicalDeviceCooperativeVectorFeaturesNV).next
        has_coopvec = q.cooperative_vector
        if has_coopvec
            feature_chain = Vulkan.PhysicalDeviceCooperativeVectorFeaturesNV(
                true,                             # cooperative_vector
                q.cooperative_vector_training;    # (inference-only is legal)
                next=feature_chain
            )
        end
    end
    if has_max_reconv
        q = Vulkan.get_physical_device_features_2(phys_dev,
            Vulkan.PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR).next
        has_max_reconv = q.shader_maximal_reconvergence
        has_max_reconv && (feature_chain =
            Vulkan.PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR(true; next=feature_chain))
    end
    if has_subgroup_ucf
        q = Vulkan.get_physical_device_features_2(phys_dev,
            Vulkan.PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR).next
        has_subgroup_ucf = q.shader_subgroup_uniform_control_flow
        has_subgroup_ucf && (feature_chain =
            Vulkan.PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR(true; next=feature_chain))
    end
    if has_subgroup_rotate
        q = Vulkan.get_physical_device_features_2(phys_dev,
            Vulkan.PhysicalDeviceShaderSubgroupRotateFeaturesKHR).next
        has_subgroup_rotate = q.shader_subgroup_rotate
        has_subgroup_rotate && (feature_chain =
            Vulkan.PhysicalDeviceShaderSubgroupRotateFeaturesKHR(
                true,                             # shader_subgroup_rotate
                q.shader_subgroup_rotate_clustered;
                next=feature_chain))
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

    # Zero-alloc Vulkan function pointers for hot paths. Per device — see the
    # field's comment on `VkContext`; a global here crashed the first two-device
    # run. Kept in a local until the context exists, and assigned below.
    cmd_barrier_fptr = Vulkan.function_pointer(device, "vkCmdPipelineBarrier")

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
        rt_props, debug_messenger, validation,
        2, n_queues,  # next_queue_index=2 (0=primary, 1=compute), max=n_queues
        async_qf_idx, async_n_queues,
        false,        # device_lost (fresh context)
        mem_props, max_wg,
        as_scratch_align,
        has_ray_query,
        has_ser,
        has_coopmat,
        coopmat_shapes,
        coopmat2_caps,
        has_coopvec,
        has_max_reconv,
        has_subgroup_ucf,
        has_subgroup_rotate,
        has_memory_budget,
        has_external_memory,
        gpu_assisted,
        string(phys_props.driver_version),
        has_video_decode, video_decode_queue, video_qf_idx,
        query_device_compute(phys_dev, phys_props),
    )
    # After construction, because it is resolved from THIS device and the inner
    # constructor has no access to the local. Per device — a module-global one
    # sent the first device's command buffers through the second device's driver.
    ctx.cmd_pipeline_barrier_fptr = cmd_barrier_fptr
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

"""Whether this is Mesa's software rasteriser, which every machine here has."""
islavapipe(dev) = occursin("llvmpipe",
    String(filter(!=('\0'), collect(Vulkan.get_physical_device_properties(dev).device_name))))

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
    (p_callback_data == C_NULL || p_user_data == C_NULL) && return UInt32(0)
    # One isbits load, no allocation: which device's ring this messenger belongs
    # to is the pointer the driver hands back, not a global.
    r = unsafe_load(Ptr{ValidationRingRaw}(p_user_data))
    data = unsafe_load(p_callback_data)        # isbits C struct → stack value, no heap alloc
    msg_ptr = Ptr{UInt8}(data.pMessage)
    idx  = unsafe_load(r.write)
    slot = idx % VAL_RING_SLOTS
    dst  = r.buf + slot * VAL_RING_SLOT_BYTES
    n = 0
    if msg_ptr != C_NULL
        n = Int(ccall(:strlen, Csize_t, (Ptr{UInt8},), msg_ptr))
        n > VAL_RING_SLOT_BYTES - 1 && (n = VAL_RING_SLOT_BYTES - 1)
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, msg_ptr, n % Csize_t)
    end
    unsafe_store!(r.len, Cint(n), slot + 1)
    unsafe_store!(r.sev, severity,  slot + 1)
    unsafe_store!(r.typ, type,      slot + 1)
    unsafe_store!(r.write, idx + 1)
    # Return VK_FALSE — can't throw from a @cfunction callback (would corrupt
    # Vulkan state).  Errors are surfaced via drain_validation_messages!.
    return UInt32(0)
end

# Classify one drained message and, for hard validation errors, capture it.
# Runs ONLY on the main thread (called from drain), so logging/alloc is safe.
function handle_validation_message(r::ValidationRing, message::String, sev_u::UInt32, type_u::UInt32)
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
    # Capture just the user text into `r.printf`, separate from validation.
    if contains(message, "DebugPrintf")
        marker = findlast("DebugPrintf:", message)
        text = marker === nothing ? message :
               lstrip(message[last(marker) + 1 : end], ['\n', '\r', ' '])
        if length(r.printf) >= MAX_PRINTF_MESSAGES
            popfirst!(r.printf)
        end
        push!(r.printf, String(text))
        @info "lava_printf" text
        return nothing
    end
    if is_error && is_validation && !is_setup_noise
        if length(r.messages) >= MAX_VALIDATION_MESSAGES
            popfirst!(r.messages)
        end
        push!(r.messages, message)
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
capture hard validation errors in `ctx.validation.messages`, and log them. Must be
called on the main thread (it allocates + logs). Idempotent — only processes
slots written since the last drain. Call this before reading
`ctx.validation.messages` and while polling for a GPU-AV fault.
"""
function drain_validation_messages!(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[])
    ctx === nothing && return nothing
    r = ctx.validation
    write_idx = @inbounds r.write[1]
    read_idx  = r.read
    # If the callback lapped us, skip the slots it overwrote.
    if write_idx - read_idx > VAL_RING_SLOTS
        read_idx = write_idx - VAL_RING_SLOTS
    end
    while read_idx < write_idx
        slot = read_idx % VAL_RING_SLOTS
        n    = Int(@inbounds r.len[slot + 1])
        sev  = @inbounds r.sev[slot + 1]
        typ  = @inbounds r.typ[slot + 1]
        message = n == 0 ? "(no message)" :
            unsafe_string(pointer(r.buf) + slot * VAL_RING_SLOT_BYTES, n)
        handle_validation_message(r, message, sev, typ)
        read_idx += 1
    end
    r.read = write_idx
    return nothing
end

function setup_debug_messenger(instance::Vulkan.Instance, ring::ValidationRing)
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
    # `user_data` is what makes the ring per-device: the driver hands this
    # pointer back to every invocation, so the callback never has to guess which
    # context it is reporting for.
    index = findfirst(==(min_sev), Vulkan.message_severities)
    severity = |(Vulkan.message_severities[index:end]...)
    messenger = Vulkan.DebugUtilsMessengerEXT(
        instance, severity,
        Vulkan.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            Vulkan.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            Vulkan.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        callback_ptr;
        user_data = ring_user_data(ring),
    )
    return messenger
end

"""
    get_validation_messages() -> Vector{String}

Return recent validation layer messages. Useful for diagnosing DEVICE_LOST errors.
"""
get_validation_messages(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); copy(ctx.validation.messages))

"""
    clear_validation_messages!()

Clear the validation message buffer. Drains the async ring first (so captured
hard errors are logged), then empties the capture list.
"""
clear_validation_messages!(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); empty!(ctx.validation.messages))

"""
    get_printf_output() -> Vector{String}

Return captured `@lava_printf` output since the last clear. Drains the async
callback ring first. Requires `enable_debug_printf!()` to have been called.
"""
get_printf_output(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); copy(ctx.validation.printf))

"""
    clear_printf_output!()

Drop captured `@lava_printf` output (drains the ring first).
"""
clear_printf_output!(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); empty!(ctx.validation.printf))

"""
    check_validation_errors!(context::String)

Check if any validation errors were captured since the last check.
Throws `LavaError` with the error messages if any errors are found.
Call this after Vulkan operations that may trigger validation errors
(shader module creation, pipeline creation, dispatch recording).
"""
function check_validation_errors!(context::String,
                                 ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[])
    ctx === nothing && return
    drain_validation_messages!(ctx)
    msgs = ctx.validation.messages
    isempty(msgs) && return
    # Separate true errors from warnings using the severity recorded by the callback.
    # GPU-AV messages like "Unaligned pointer access" are errors, not warnings,
    # even though their text may contain strings like "WARNING-Validation".
    # We rely on the callback storing only ERROR-severity messages.
    errors = copy(msgs)
    isempty(errors) && return
    n = min(length(errors), 5)
    detail = join(["  [$i] $(first(errors[i], 1000))" for i in 1:n], "\n")
    # Clear after reporting to avoid re-triggering
    empty!(msgs)
    throw(LavaError(
        context,
        "Vulkan validation error(s):\n$detail",
        "Fix the validation errors above before proceeding."
    ))
end
