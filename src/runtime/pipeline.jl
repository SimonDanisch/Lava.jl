# Compute pipeline creation and caching for Lava.jl
#
# Creates VkShaderModule + VkComputePipeline from validated SPIR-V binary.
# Pipelines are cached by SPIR-V content hash.
#
# On Windows, vkCreateComputePipelines runs on a dedicated thread with a 256MB
# stack to work around AMD's proprietary Vulkan driver overflowing its internal
# compiler stack when processing certain SPIR-V patterns.

const LARGE_STACK_PIPELINE = Sys.iswindows()
const LARGE_STACK_SIZE = 256 * 1024 * 1024  # 256 MB

if LARGE_STACK_PIPELINE

struct _VkCreatePipelineArgs
    device::Ptr{Cvoid}
    PIPELINE_CACHE::Ptr{Cvoid}
    create_info_count::UInt32
    _pad::UInt32
    p_create_infos::Ptr{Cvoid}
    p_allocator::Ptr{Cvoid}
    p_pipelines::Ptr{Cvoid}
    result::Int32
    _pad2::Int32
end

function vk_pipeline_thread_callback(args_ptr::Ptr{Cvoid})::UInt32
    args = unsafe_load(Ptr{_VkCreatePipelineArgs}(args_ptr))
    result = ccall((:vkCreateComputePipelines, "vulkan-1"),
        Int32,
        (Ptr{Cvoid}, Ptr{Cvoid}, UInt32, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        args.device, args.PIPELINE_CACHE, args.create_info_count,
        args.p_create_infos, args.p_allocator, args.p_pipelines)
    unsafe_store!(Ptr{Int32}(args_ptr + 48), result)
    return UInt32(0)
end

end # if LARGE_STACK_PIPELINE

const PIPELINE_THREAD_CFUNC = Ref{Ptr{Nothing}}(C_NULL)

if LARGE_STACK_PIPELINE
function init_pipeline_thread!()
    PIPELINE_THREAD_CFUNC[] = @cfunction(vk_pipeline_thread_callback, UInt32, (Ptr{Cvoid},))
end
else
init_pipeline_thread!() = nothing
end

function create_compute_pipeline_large_stack(device::Ptr{Cvoid},
                                               pipeline_cache::Ptr{Cvoid},
                                               create_info_ptr::Ptr{Cvoid},
                                               p_pipelines::Ptr{Cvoid})
    args = Ref(_VkCreatePipelineArgs(
        device, pipeline_cache, UInt32(1), UInt32(0),
        create_info_ptr, C_NULL, p_pipelines,
        Int32(0), Int32(0),
    ))
    GC.@preserve args begin
        args_ptr = Ptr{Cvoid}(pointer_from_objref(args))
        thread_id = Ref(UInt32(0))
        handle = ccall((:CreateThread, "kernel32"), Ptr{Cvoid},
            (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, UInt32, Ptr{UInt32}),
            C_NULL, LARGE_STACK_SIZE, PIPELINE_THREAD_CFUNC[], args_ptr,
            UInt32(0), thread_id)
        handle == C_NULL && error("CreateThread failed for vkCreateComputePipelines")
        # 600 second timeout — AMD's Windows driver can be very slow under
        # accumulated session state (~thousands of prior compiles). Real hangs
        # are exceedingly rare; the failure mode is "slow but progressing".
        # If we timeout-and-CloseHandle while the thread is still inside
        # AMDVLK's compiler, the thread later crashes accessing freed
        # _VkCreatePipelineArgs (observed: access violation in vkResetEvent
        # after a Pkg.test Tier 4 broadcast Complex{Int32}). The crash kills
        # the whole process. To avoid that, we TerminateThread on timeout —
        # leaks the AMDVLK internal allocations but keeps Julia alive so
        # the test/user can recover gracefully (or call vk_reset_device!).
        wait_result = ccall((:WaitForSingleObject, "kernel32"), UInt32,
            (Ptr{Cvoid}, UInt32), handle, UInt32(600_000))
        if wait_result == 0x00000102  # WAIT_TIMEOUT
            # Force-kill the leaked AMDVLK thread so it can't crash the
            # process later when it accesses our freed args memory.
            ccall((:TerminateThread, "kernel32"), Cint,
                (Ptr{Cvoid}, UInt32), handle, UInt32(1))
            ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), handle)
            # Best-effort: mark the Lava context as device_lost so subsequent
            # work doesn't try to use a driver in unknown state.
            ctx = VK_CONTEXT_REF[]
            ctx === nothing || mark_device_lost!(ctx)
            error("vkCreateComputePipelines timed out after 600s — AMD Windows " *
                  "driver shader compiler hung. Device marked DEVICE_LOST; " *
                  "call Lava.vk_reset_device!() to recover.")
        end
        ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), handle)
        wait_result != 0 && error("WaitForSingleObject failed: $wait_result")
        return args[].result
    end
end

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

"""
    PIPELINE_NO_COMPILE

When set, compute pipelines are created with
`PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`: the driver REFUSES to
build a binary and returns `VK_PIPELINE_COMPILE_REQUIRED` instead. That turns
"the pipeline cache was warm" from a wall-clock inference into something the
driver asserts directly — a fast second run otherwise proves nothing, since it
could just as easily be Lava's in-memory `PIPELINE_CACHE` rather than the
on-disk blob.

Requires `pipelineCreationCacheControl`, which `vk_device!` enables as part of
the Vulkan 1.3 core feature set. Use via `no_pipeline_compilation`.
"""
const PIPELINE_NO_COMPILE = Ref(false)

"How many pipeline creations were refused because they would have compiled."
const PIPELINE_COMPILES_REFUSED = Ref(0)

"Entry names of pipelines that would have compiled while PIPELINE_NO_COMPILE was set."
const PIPELINE_COMPILE_MISSES = String[]

# VkResult VK_PIPELINE_COMPILE_REQUIRED. Positive (success class), so a bare
# `result != 0` check reports it as a generic failure — it needs naming.
const VK_PIPELINE_COMPILE_REQUIRED = 1000297000

"""
    no_pipeline_compilation(f) -> f()

Run `f` with the driver forbidden from compiling any compute pipeline: every
creation carries `PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`, so a
pipeline that is not already in the cache makes creation fail loudly instead of
quietly building a binary.

This is how to prove a cache is doing its job. Timing cannot: a fast second run
is equally explained by Lava's in-memory `PIPELINE_CACHE` dict, which never
touches disk. Under this wrapper, a run that completes did zero driver
compilation — the driver said so.

Clears Lava's in-memory pipeline cache first, so only the `VkPipelineCache`
(seeded from disk) can satisfy a creation. Restores the previous setting after.

    Lava.no_pipeline_compilation() do
        run_the_workload()          # throws if anything would compile
    end
"""
function no_pipeline_compilation(f)
    old = PIPELINE_NO_COMPILE[]
    old_refused = PIPELINE_COMPILES_REFUSED[]
    # Otherwise a hit on the Julia-side dict would mask a cold VkPipelineCache.
    empty!(PIPELINE_CACHE)
    empty!(PIPELINE_INSERTION_ORDER)
    PIPELINE_NO_COMPILE[] = true
    PIPELINE_COMPILES_REFUSED[] = 0
    empty!(PIPELINE_COMPILE_MISSES)
    try
        return f()
    finally
        PIPELINE_NO_COMPILE[] = old
        PIPELINE_COMPILES_REFUSED[] += old_refused
    end
end

const PIPELINE_CACHE = Dict{UInt64, LavaComputePipeline}()
const PIPELINE_INSERTION_ORDER = UInt64[]
const MAX_PIPELINE_CACHE_SIZE = Ref(1024)

# Lava's cooperative-matrix kernels are written against a 32-lane subgroup: they
# index their subgroup as `lane ÷ 32` and size their workgroups in multiples of
# it. Cooperative-matrix operations are subgroup-scoped, so on a device with a
# wider wave those lanes do not form a subgroup and the kernel writes only part
# of its output tile.
const COOPMAT_SUBGROUP = 32

struct SubgroupSizeControl
    min::Int
    max::Int
    compute::Bool   # COMPUTE present in requiredSubgroupSizeStages
end

const SUBGROUP_SIZE_CONTROL = Ref{Union{Nothing,SubgroupSizeControl}}(nothing)

# The device's DEFAULT subgroup width, as opposed to the min/max it can be pinned
# to. Queried once; cleared on device reset with the rest.
const DEVICE_SUBGROUP_SIZE = Ref(0)

function device_subgroup_size(ctx::VkContext = vk_context())
    DEVICE_SUBGROUP_SIZE[] != 0 && return DEVICE_SUBGROUP_SIZE[]
    props = Vulkan.get_physical_device_properties_2(ctx.physical_device,
                                                    Vulkan.PhysicalDeviceSubgroupProperties)
    DEVICE_SUBGROUP_SIZE[] = Int(props.next.subgroup_size)
end

"""
    subgroup_size_control(ctx) -> SubgroupSizeControl

The device's `VkPhysicalDeviceSubgroupSizeControlProperties`, queried once.
`compute` says whether a compute pipeline may pin its own subgroup size.
"""
function subgroup_size_control(ctx::VkContext = vk_context())
    cached = SUBGROUP_SIZE_CONTROL[]
    cached === nothing || return cached
    p = Vulkan.get_physical_device_properties_2(
            ctx.physical_device, Vulkan.PhysicalDeviceSubgroupSizeControlProperties).next
    c = SubgroupSizeControl(
        Int(p.min_subgroup_size), Int(p.max_subgroup_size),
        (UInt32(p.required_subgroup_size_stages) &
         UInt32(Vulkan.SHADER_STAGE_COMPUTE_BIT)) != 0)
    SUBGROUP_SIZE_CONTROL[] = c
    return c
end

"""Whether a compute pipeline on this device can be pinned to `n` lanes per subgroup."""
function can_require_subgroup_size(ctx::VkContext, n::Integer)
    c = subgroup_size_control(ctx)
    c.compute && c.min <= n <= c.max
end

# Whether the module declares `cap` via OpCapability. Capabilities are the first
# thing in a valid SPIR-V module, so this stops at OpMemoryModel rather than
# walking the whole binary.
function spirv_declares_capability(spirv::Vector{UInt8}, cap::UInt32)
    length(spirv) >= 20 && length(spirv) % 4 == 0 || return false
    w = reinterpret(UInt32, spirv)
    w[1] == 0x07230203 || return false          # magic, matching our byte order
    i = 6                                       # first word after the 5-word header
    @inbounds while i <= length(w)
        count = w[i] >> 16
        count == 0 && break                     # malformed; refuse to loop forever
        opcode = UInt16(w[i] & 0xffff)
        opcode == Op.OpMemoryModel && break      # past the capability section
        if opcode == Op.OpCapability && count >= 2 && i + 1 <= length(w)
            w[i+1] == cap && return true
        end
        i += count
    end
    return false
end

# Register cleanup callback for vk_reset_device!
push!(RESET_CALLBACKS, function()
    empty!(PIPELINE_CACHE)
    empty!(PIPELINE_INSERTION_ORDER)
    SUBGROUP_SIZE_CONTROL[] = nothing   # re-query: the next device may differ
    DEVICE_SUBGROUP_SIZE[]  = 0
end)

"""
    spirv_content_hash(bytes) -> UInt64

A hash over **every** byte of a SPIR-V module.

`Base.hash` on a large `Vector` deliberately *samples* elements instead of
reading all of them. For a dictionary of distinct arrays that is a sensible
trade; as a pipeline-cache key it is a silent-wrong-results bug, because two
instantiations of one kernel can differ in a handful of bytes and nothing else.

Measured: the 256- and 512-wide instantiations of the same kernel produce modules
that differ at **exactly one byte** — index 230, the `LocalSize` x operand — and
`hash` returns the identical value for both. The 512-wide launch therefore looked
up the 256-wide pipeline, dispatched a 256-thread shader over a grid computed for
512, and wrote exactly half its output. At 1024 it wrote a quarter. Whichever
size compiled first won, which is why it looked order- and body-dependent, and
why adding any unrelated store "fixed" it (different bytes, no collision).

That is the entire content of what was recorded for a long time as "above 256
this device silently runs fewer invocations than the shader declares". The device
runs every lane.

FNV-1a, because it reads every byte and needs no dependency. A 9 KB module costs
microseconds, and this runs once per pipeline creation, never per launch.
"""
function spirv_content_hash(bytes::AbstractVector{UInt8})
    h = 0xcbf29ce484222325
    @inbounds for b in bytes
        h = (h ⊻ b) * 0x00000100000001b3
    end
    return h
end

"""
    get_compute_pipeline(spirv_bytes::Vector{UInt8}, entry_name::String;
                         push_constant_size::Integer=8,
                         needs_tlas_descriptor::Bool=false) -> LavaComputePipeline

Get or create a compute pipeline from SPIR-V binary.
Validates SPIR-V before creating the shader module.
"""
function get_compute_pipeline(ctx::VkContext, spirv_bytes::Vector{UInt8}, entry_name::String;
                               push_constant_size::Integer=8,
                               needs_tlas_descriptor::Bool=false,
                               pipeline_cache=nothing)
    # `spirv_content_hash`, NOT `hash(spirv_bytes)` — see its docstring. The
    # length goes in too, so a truncation cannot alias a prefix.
    #
    # `ctx.id` leads, because a `LavaComputePipeline` owns a `VkPipeline`, a
    # `VkShaderModule`, a pipeline layout and a descriptor-set layout, and every
    # one of those belongs to the `VkDevice` that made it. Without the device in
    # the key, two devices compiling the same kernel produce the same key and the
    # second is handed the first's handles to bind into its own command buffer —
    # undefined behaviour, and the same class as the `hash(spirv_bytes)` collision
    # this line already carries a warning about (`GUARDRAILS.md` §8).
    cache_key = hash((ctx.id, spirv_content_hash(spirv_bytes), length(spirv_bytes),
                      entry_name, push_constant_size, needs_tlas_descriptor))
    cached = get(PIPELINE_CACHE, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    dev = ctx.device

    # Create shader module from SPIR-V
    @assert length(spirv_bytes) % 4 == 0 "SPIR-V binary must be 4-byte aligned"
    code_u32 = reinterpret(UInt32, spirv_bytes)
    shader_mod = Vulkan.ShaderModule(dev, length(spirv_bytes), code_u32)
    check_validation_errors!("vkCreateShaderModule (compute)")

    # Pipeline layout with push constant range
    push_ranges = if push_constant_size > 0
        [Vulkan.PushConstantRange(
            Vulkan.SHADER_STAGE_COMPUTE_BIT,
            UInt32(0),
            UInt32(push_constant_size)
        )]
    else
        Vulkan.PushConstantRange[]
    end

    # Descriptor set layout: binding 0 = TLAS (only when needs_tlas_descriptor)
    ds_layout = nothing
    ds_layouts = Vulkan.DescriptorSetLayout[]
    if needs_tlas_descriptor
        ds_layout = Vulkan.DescriptorSetLayout(
            dev,
            [Vulkan.DescriptorSetLayoutBinding(
                UInt32(0),
                Vulkan.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                Vulkan.SHADER_STAGE_COMPUTE_BIT;
                descriptor_count = UInt32(1)
            )]
        )
        push!(ds_layouts, ds_layout)
    end

    layout = Vulkan.PipelineLayout(dev, ds_layouts, push_ranges)

    # A cooperative-matrix module assumes a 32-lane subgroup (see COOPMAT_SUBGROUP).
    # Where the device lets a pipeline pin its subgroup size we do, which makes
    # those kernels correct on wave64 hardware rather than merely disabled. This
    # is a deterministic function of (SPIR-V, device), so `cache_key` still
    # identifies the pipeline uniquely without naming the size.
    #
    # Skipped entirely where the device is already COOPMAT_SUBGROUP wide: pinning
    # would be semantically identical, and not touching the create-info at all is
    # a stronger guarantee than "should be equivalent" for the wave32 hardware the
    # kernels were tuned on.
    stage_next = C_NULL
    if spirv_declares_capability(spirv_bytes, Cap.CooperativeMatrixKHR) &&
       device_subgroup_size(ctx) != COOPMAT_SUBGROUP &&
       can_require_subgroup_size(ctx, COOPMAT_SUBGROUP)
        stage_next = Vulkan.PipelineShaderStageRequiredSubgroupSizeCreateInfo(
            UInt32(COOPMAT_SUBGROUP))
    end

    # Create compute pipeline
    stage = Vulkan.PipelineShaderStageCreateInfo(
        Vulkan.SHADER_STAGE_COMPUTE_BIT,
        shader_mod,
        entry_name;
        next = stage_next
    )
    # Optional CAPTURE_STATISTICS bit so VK_KHR_pipeline_executable_properties
    # has a populated stats table to query later via pipeline_exec_stats.
    pipeline_flags = Vulkan.PipelineCreateFlag(Vulkan.PIPELINE_CREATE_DISPATCH_BASE_BIT)
    if PIPELINE_EXEC_PROPERTIES_REQUESTED[]
        pipeline_flags |= Vulkan.PipelineCreateFlag(Vulkan.PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR)
    end
    if PIPELINE_NO_COMPILE[]
        # EARLY_RETURN_ON_FAILURE alongside, as the spec pairs them: bail at the
        # first pipeline that would compile rather than continuing the batch.
        pipeline_flags |= Vulkan.PipelineCreateFlag(Vulkan.PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) |
                          Vulkan.PipelineCreateFlag(Vulkan.PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT)
    end
    ci = Vulkan.ComputePipelineCreateInfo(stage, layout, -1; flags=pipeline_flags)

    # `pipeline_cache` lets the frozen cache hand in a per-kernel one seeded from
    # that kernel's own `.bin`; without it the device-wide cache is used, which is
    # what every other caller wants.
    pcache = pipeline_cache === nothing ? ctx.pipeline_cache : pipeline_cache
    pipeline = try
        create_compute_pipeline(dev, ci; pipeline_cache=pcache)
    catch e
        # A cache miss is RECORDED, not fatal: retry without the flag so the
        # workload finishes and every miss is reported, rather than dying at the
        # first one. Aborting a pipeline creation mid-render leaves a partially
        # recorded batch behind and cost a DEVICE_LOST when this threw instead.
        if PIPELINE_NO_COMPILE[] && occursin("PIPELINE_COMPILE_REQUIRED", sprint(showerror, e))
            # Every Lava kernel's entry point is "main", so the name alone does not
            # identify anything. Record the SPIR-V hash and size, which do.
            # `spirv_content_hash`, NOT `hash`: `Base.hash` SAMPLES a large
            # Vector, so two modules differing in one byte report as the same
            # miss. That is the bug that read as a 256-lane device cap for
            # months (see `spirv_content_hash`), here in the one instrument
            # whose job is to say WHICH module was not in the cache.
            push!(PIPELINE_COMPILE_MISSES,
                  string(entry_name, " spirv=",
                         string(spirv_content_hash(spirv_bytes), base=16),
                         " (", length(spirv_bytes), " bytes)",
                         pipeline_cache === nothing ? "" : " [per-kernel cache]"))
            old_flag = PIPELINE_NO_COMPILE[]
            PIPELINE_NO_COMPILE[] = false
            try
                ci_retry = Vulkan.ComputePipelineCreateInfo(stage, layout, -1;
                    flags = Vulkan.PipelineCreateFlag(Vulkan.PIPELINE_CREATE_DISPATCH_BASE_BIT))
                create_compute_pipeline(dev, ci_retry; pipeline_cache=pcache)
            finally
                PIPELINE_NO_COMPILE[] = old_flag
            end
        else
            rethrow()
        end
    end

    result = LavaComputePipeline(shader_mod, layout, pipeline, UInt32(push_constant_size),
                                  needs_tlas_descriptor, ds_layout)
    PIPELINE_CACHE[cache_key] = result
    push!(PIPELINE_INSERTION_ORDER, cache_key)
    while length(PIPELINE_INSERTION_ORDER) > MAX_PIPELINE_CACHE_SIZE[]
        old_key = popfirst!(PIPELINE_INSERTION_ORDER)
        delete!(PIPELINE_CACHE, old_key)
        # No defensive pin needed: every dispatch pin!s its pipeline into
        # `batch.pinned`, so any in-flight batch that used this pipeline
        # holds a strong ref.  Once all such batches retire the ref drops
        # naturally and GC runs the destructor.
    end
    return result
end

"""
    alloc_compute_tlas_descriptor_set(dev, pipeline, tlas) -> (DescriptorPool, DescriptorSet)

Allocate a fresh descriptor pool and descriptor set that binds `tlas.accel`
to set=0, binding=0 for a compute pipeline compiled with
`needs_tlas_descriptor=true`. The caller MUST `pin!(batch, pool)` so the
pool stays alive until the dispatch retires; once the batch is reclaimed
the pool's destructor frees it.

We deliberately do NOT cache here. A previous version keyed a cache by
`(ds_layout, objectid(LavaTLAS))`, but `Raycore.sync!` produces a new
LavaTLAS each rebuild → cache misses → unbounded growth → eviction with
WeakRef-based TLAS-GC detection that lags Julia GC → pools destroyed
while their descriptor sets were still in flight on the GPU. Per-dispatch
allocation is microseconds and removes the entire class of bug.

`tlas` is a `LavaTLAS` — declared later in raytracing/acceleration.jl.
"""
function alloc_compute_tlas_descriptor_set(dev::Vulkan.Device,
                                            pipeline::LavaComputePipeline,
                                            tlas)  # LavaTLAS — declared later
    ds_layout = pipeline.descriptor_set_layout::Vulkan.DescriptorSetLayout
    desc_pool = Vulkan.DescriptorPool(dev, UInt32(1), [
        Vulkan.DescriptorPoolSize(
            Vulkan.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, UInt32(1)),
    ])
    desc_sets = @vk_checked "vkAllocateDescriptorSets (compute TLAS)" Vulkan.allocate_descriptor_sets(dev,
        Vulkan.DescriptorSetAllocateInfo(desc_pool, [ds_layout]))
    desc_set = desc_sets[1]

    as_write = Vulkan.WriteDescriptorSetAccelerationStructureKHR([tlas.accel])
    write_ds = Vulkan.WriteDescriptorSet(
        desc_set, UInt32(0), UInt32(0),
        Vulkan.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        Vulkan.DescriptorImageInfo[],
        Vulkan.DescriptorBufferInfo[],
        Vulkan.BufferView[];
        descriptor_count=UInt32(1),
        next=as_write,
    )
    Vulkan.update_descriptor_sets(dev, [write_ds], [])

    return desc_pool, desc_set
end

function create_compute_pipeline(dev::Vulkan.Device, ci::Vulkan.ComputePipelineCreateInfo;
                                   pipeline_cache=C_NULL)
    # Resolve the raw handle for the LARGE_STACK ccall path. Vulkan.jl high-
    # level wrappers carry the raw UInt64 (non-dispatchable handle) in `.vks`;
    # ccall accepts a Ptr{Cvoid} of that value on 64-bit systems.
    cache_handle = if pipeline_cache === C_NULL
        C_NULL
    else
        raw = pipeline_cache.vks
        raw isa Ptr ? raw : Ptr{Cvoid}(UInt(raw))
    end
    if LARGE_STACK_PIPELINE
        ci_low = convert(Vulkan._ComputePipelineCreateInfo, ci)
        vk_ci_ref = Ref(ci_low.vks)
        pipeline_out = Ref(Ptr{Cvoid}(C_NULL))

        GC.@preserve ci_low vk_ci_ref pipeline_out pipeline_cache begin
            vk_result = create_compute_pipeline_large_stack(
                dev.vks,
                cache_handle,
                Ptr{Cvoid}(pointer_from_objref(vk_ci_ref)),
                Ptr{Cvoid}(pointer_from_objref(pipeline_out)))
            if vk_result == VK_PIPELINE_COMPILE_REQUIRED
                PIPELINE_COMPILES_REFUSED[] += 1
                error("vkCreateComputePipelines returned VK_PIPELINE_COMPILE_REQUIRED: this " *
                      "pipeline is NOT in the cache and would have been compiled. " *
                      "(PIPELINE_NO_COMPILE is set; see no_pipeline_compilation.)")
            end
            vk_result != 0 && error("vkCreateComputePipelines failed with VkResult $vk_result")
        end

        raw_pipeline = pipeline_out[]
        parent = Vulkan.handle(dev)
        destructor = x -> Vulkan._destroy_pipeline(parent, x)
        return Vulkan.Pipeline(raw_pipeline, destructor, dev)
    else
        # The return code is NOT discardable here. VK_PIPELINE_COMPILE_REQUIRED is
        # a SUCCESS-class code, so `@check` inside Vulkan.jl does not raise and
        # `unwrap` returns normally — with pPipelines[1] left at VK_NULL_HANDLE.
        # Dropping the code therefore hands back a null pipeline that segfaults
        # later at vkCmdBindPipeline, a long way from the cause. This check used to
        # exist only in the LARGE_STACK (Windows) branch above.
        pipelines, code = @vk_checked "vkCreateComputePipelines" Vulkan.create_compute_pipelines(dev, [ci]; pipeline_cache)
        if code == Vulkan.PIPELINE_COMPILE_REQUIRED
            PIPELINE_COMPILES_REFUSED[] += 1
            error("vkCreateComputePipelines returned VK_PIPELINE_COMPILE_REQUIRED: this " *
                  "pipeline is NOT in the cache and would have been compiled. " *
                  "(PIPELINE_NO_COMPILE is set; see no_pipeline_compilation.)")
        end
        return pipelines[1]
    end
end
