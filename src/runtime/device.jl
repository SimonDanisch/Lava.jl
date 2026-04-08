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
    fence::Vulkan.Fence                 # Single fence for the whole batch
    recording::Bool
    dispatch_count::Int                 # Total dispatches across all segments (for barriers)
    segment_dispatches::Int             # Dispatches in current CB segment (for split threshold)
    last_was_rt::Bool
    data_refs::Vector{Any}
    dispatch_log::Vector{String}
    sealed_cmd_bufs::Vector{Vulkan.CommandBuffer}  # Completed CB segments awaiting submit
end

"""
    BatchQueue

An independent command submission channel owning a Vulkan queue, command pool,
and batch state. Multiple `BatchQueue`s can record and submit independently
(e.g., primary queue for graphics/present, compute queue for async RT).

Create with `BatchQueue(device, queue, queue_family_index)`.
"""
mutable struct BatchQueue
    queue::Vulkan.Queue
    cmd_pool::Vulkan.CommandPool
    active_batch::Union{Nothing, CommandBatch}
    in_flight::Vector{CommandBatch}
    free_batches::Vector{CommandBatch}
    free_cmd_bufs::Vector{Vulkan.CommandBuffer}
    # Dedicated transfer command buffer + fence (per-queue, thread-safe)
    xfer_cmd_buf::Vulkan.CommandBuffer
    xfer_fence::Vulkan.Fence
end

function BatchQueue(device::Vulkan.Device, queue::Vulkan.Queue, qf_idx::UInt32;
                    n_initial_batches::Int=2)
    cmd_pool = Vulkan.CommandPool(device, qf_idx;
        flags=Vulkan.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    batches = CommandBatch[]
    for _ in 1:n_initial_batches
        alloc_info = Vulkan.CommandBufferAllocateInfo(cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
        cb = unwrap(Vulkan.allocate_command_buffers(device, alloc_info))[1]
        fence = Vulkan.Fence(device)
        data_refs = Any[]
        sizehint!(data_refs, 128)
        push!(batches, CommandBatch(cb, fence, false, 0, 0, false, data_refs, String[], Vulkan.CommandBuffer[]))
    end
    # Dedicated transfer command buffer + fence
    xfer_alloc = Vulkan.CommandBufferAllocateInfo(cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    xfer_cmd_buf = unwrap(Vulkan.allocate_command_buffers(device, xfer_alloc))[1]
    xfer_fence = Vulkan.Fence(device)
    return BatchQueue(queue, cmd_pool, nothing, CommandBatch[], batches, Vulkan.CommandBuffer[],
                      xfer_cmd_buf, xfer_fence)
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
    # Primary batch queue — all global API functions delegate here
    default_bq::BatchQueue
    # Secondary compute queue (async RT) — same family, separate queue object
    compute_queue::Vulkan.Queue
    # Dedicated AS build command buffer + fence (always on primary queue)
    as_cmd_buf::Vulkan.CommandBuffer
    as_fence::Vulkan.Fence
    # Ray tracing (nothing if not available)
    rt_pipeline_properties::Union{Nothing, RTPipelineProperties}
    # Debug messenger (nothing if validation layers not available)
    debug_messenger::Any
    # Queue allocation: next available index + total requested from device
    next_queue_index::Int
    max_queue_count::Int
end

# Convenience accessors — keep existing code working with minimal changes.
# Batch-related fields and transfer resources delegate to default_bq.
Base.getproperty(ctx::VkContext, s::Symbol) = begin
    bq = getfield(ctx, :default_bq)
    if s === :queue; return bq.queue
    elseif s === :cmd_pool; return bq.cmd_pool
    elseif s === :active_batch; return bq.active_batch
    elseif s === :in_flight; return bq.in_flight
    elseif s === :free_batches; return bq.free_batches
    elseif s === :free_cmd_bufs; return bq.free_cmd_bufs
    elseif s === :xfer_cmd_buf; return bq.xfer_cmd_buf
    elseif s === :xfer_fence; return bq.xfer_fence
    else; return getfield(ctx, s)
    end
end

Base.setproperty!(ctx::VkContext, s::Symbol, v) = begin
    if s === :active_batch; getfield(ctx, :default_bq).active_batch = v
    else; setfield!(ctx, s, v)
    end
end

# Ring buffer of recent validation messages for context on DEVICE_LOST
const VALIDATION_MESSAGES = String[]
const MAX_VALIDATION_MESSAGES = 50

const VK_CONTEXT_REF = Ref{Union{Nothing, VkContext}}(nothing)

# Set to true after DEVICE_LOST — prevents finalizers from calling Vulkan on invalid handles
const DEVICE_LOST = Ref(false)

# Device generation counter — incremented on each vk_reset_device!().
# VkManagedBuffer records the generation at creation time. If a GC finalizer
# fires after a device reset, the buffer's generation won't match the current
# one and destruction is skipped (the old device cleaned up its own resources).
const DEVICE_GENERATION = Ref{UInt64}(0)

# Callbacks for vk_reset_device! — registered by later-included files (pipeline.jl,
# command.jl, launch.jl, memory.jl) to clear their module-level caches.
const RESET_CALLBACKS = Function[]

# Set of optional PhysicalDeviceFeatures enabled on the current device.
# Populated during init_vulkan!(), queried by compiler target and tests.
const ENABLED_OPTIONAL_FEATURES = Set{Symbol}()

"""
    has_device_feature(feature::Symbol) -> Bool

Check if an optional Vulkan device feature is enabled.
Triggers device initialization if not yet done.
Returns `true` during precompilation (safe default — the real check
happens at shader module creation time via Vulkan validation).

Example: `has_device_feature(:shader_float_64)`
"""
function has_device_feature(feature::Symbol)
    if ccall(:jl_generating_output, Cint, ()) != 0
        return true  # assume available during precompilation
    end
    vk_context()  # ensure initialized
    return feature in ENABLED_OPTIONAL_FEATURES
end

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
vk_queue() = vk_context().queue
vk_compute_queue() = vk_context().compute_queue

"""
    vk_reset_device!()

Reinitialize the Vulkan device after DEVICE_LOST or other unrecoverable errors.
Destroys the old context and creates a fresh one. Clears all caches (pipelines,
kernels, arg buffers).

**WARNING**: All existing `LavaArray`s become INVALID after reset — their backing
GPU buffers no longer exist. You must reallocate all GPU data.
"""
function vk_reset_device!()
    DEVICE_GENERATION[] += 1  # Invalidate old VkManagedBuffer handles
    DEVICE_LOST[] = false
    VK_CONTEXT_REF[] = nothing
    # Don't destroy old Vulkan handles — they're invalid after DEVICE_LOST.
    # GC will eventually try to destroy them; _destroy_buffer! skips when
    # DEVICE_LOST was true (and we set it false only after clearing context).
    empty!(VALIDATION_MESSAGES)
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

"""Check if a recording is active on the default batch queue."""
function has_active_recording(ctx::VkContext)
    batch = ctx.default_bq.active_batch
    return batch !== nothing && batch.recording
end

function init_vulkan!()
    # Create instance
    app_info = Vulkan.ApplicationInfo(
        v"0.1.0", v"0.1.0", v"1.3.0";
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

    # GPU-assisted validation: opt-in via LAVA_GPU_AV=1 (instruments shaders, very slow)
    gpu_assisted = false
    want_gpu_av = get(ENV, "LAVA_GPU_AV", "0") != "0"
    if want_gpu_av && has_validation && "VK_EXT_validation_features" in ext_names
        push!(inst_extensions, "VK_EXT_validation_features")
        validation_features = Vulkan.ValidationFeaturesEXT(
            [Vulkan.VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
             Vulkan.VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT],
            []
        )
        instance = Vulkan.Instance(
            layers,
            inst_extensions;
            application_info=app_info,
            next=validation_features
        )
        gpu_assisted = true
    else
        instance = Vulkan.Instance(
            layers,
            inst_extensions;
            application_info=app_info
        )
        if has_validation && want_gpu_av
            @warn "Vulkan validation layers active but VK_EXT_validation_features not available. GPU-assisted validation disabled."
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
    # Request up to 4 queues: primary, async compute, + 2 for per-screen graphics
    qf_props = Vulkan.get_physical_device_queue_family_properties(phys_dev)
    max_queues = qf_props[qf_idx + 1].queue_count
    n_queues = min(4, Int(max_queues))
    queue_priorities = ones(Float32, n_queues)
    queue_ci = [Vulkan.DeviceQueueCreateInfo(qf_idx, queue_priorities)]

    # Check for RT extension support
    has_rt = has_rt_extensions(phys_dev)

    # Check for workgroup memory explicit layout (needed for mixed-type shared memory structs)
    has_wg_explicit = has_extension(phys_dev, "VK_KHR_workgroup_memory_explicit_layout")

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
    if has_wg_explicit
        push!(extensions, "VK_KHR_workgroup_memory_explicit_layout")
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
        false,  # timeline_semaphore
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
    # Dynamic rendering (Vulkan 1.3 core) — no VkRenderPass/VkFramebuffer boilerplate
    dyn_rendering_features = Vulkan.PhysicalDeviceDynamicRenderingFeatures(
        true;   # dynamic_rendering
        next=vulkan12_features
    )

    # Chain workgroup explicit layout if available
    feature_chain = dyn_rendering_features
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

    # Query what the device actually supports before requesting features.
    # Some features (geometry_shader, tessellation_shader, wide_lines, etc.)
    # are unavailable on MoltenVK/macOS — requesting them causes VK_ERROR_FEATURE_NOT_PRESENT.
    supported_features = Vulkan.get_physical_device_features(phys_dev)
    required_features = Symbol[:shader_int_64, :shader_int_16]
    optional_features = Symbol[:shader_float_64, :geometry_shader, :tessellation_shader,
                               :fill_mode_non_solid, :wide_lines, :large_points]
    enabled = Symbol[]
    for f in required_features
        if !getproperty(supported_features, f)
            throw(LavaError("device initialization",
                "Required feature $f not supported by $(dev_name)",
                "Lava requires a GPU that supports $f"))
        end
        push!(enabled, f)
    end
    empty!(ENABLED_OPTIONAL_FEATURES)
    for f in optional_features
        if getproperty(supported_features, f)
            push!(enabled, f)
            push!(ENABLED_OPTIONAL_FEATURES, f)
        end
    end
    core_features = Vulkan.PhysicalDeviceFeatures(enabled...)

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

    # Default batch queue on the primary queue (includes transfer cmd buf/fence)
    default_bq = BatchQueue(device, queue, qf_idx)

    # Dedicated AS build command buffer + fence (separate from batch system, always primary queue)
    as_alloc = Vulkan.CommandBufferAllocateInfo(default_bq.cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    as_cmd_buf = unwrap(Vulkan.allocate_command_buffers(device, as_alloc))[1]
    as_fence = Vulkan.Fence(device)

    has_validation = !isempty(layers)
    if has_rt
        @info "Lava: initialized Vulkan device with RT" device=dev_name queue_family=qf_idx handle_size=rt_props.shader_group_handle_size max_recursion=rt_props.max_ray_recursion_depth validation=has_validation gpu_assisted=gpu_assisted debug_utils=has_debug_utils
    else
        @info "Lava: initialized Vulkan device (no RT)" device=dev_name queue_family=qf_idx validation=has_validation gpu_assisted=gpu_assisted debug_utils=has_debug_utils
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

    return VkContext(
        instance, phys_dev, device, qf_idx, dev_name,
        default_bq, compute_queue,
        as_cmd_buf, as_fence,
        rt_props,
        debug_messenger,
        2, n_queues  # next_queue_index=2 (0=primary, 1=compute), max=n_queues
    )
end

"""
    allocate_batch_queue!() -> BatchQueue

Create a new independent BatchQueue on a separate Vulkan queue (if available).
Falls back to a separate command pool on the primary queue if all queues are taken.
Used by Screen for isolated graphics rendering.
"""
function allocate_batch_queue!()
    ctx = vk_context()
    idx = ctx.next_queue_index
    if idx < ctx.max_queue_count
        queue = Vulkan.get_device_queue(ctx.device, ctx.queue_family_index, UInt32(idx))
        ctx.next_queue_index += 1
    else
        # All hardware queues taken — reuse primary queue with separate command pool
        queue = ctx.default_bq.queue
    end
    return BatchQueue(ctx.device, queue, ctx.queue_family_index)
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

function debug_callback(
    severity,
    type,
    p_callback_data::Ptr{Vulkan.VkCore.VkDebugUtilsMessengerCallbackDataEXT},
    p_user_data::Ptr{Cvoid},
)
    p_callback_data == C_NULL && return UInt32(0)
    data = unsafe_load(p_callback_data)
    msg_ptr = data.pMessage
    message = msg_ptr == C_NULL ? "(no message)" : unsafe_string(msg_ptr)

    # Store in ring buffer for context on DEVICE_LOST
    if length(VALIDATION_MESSAGES) >= MAX_VALIDATION_MESSAGES
        popfirst!(VALIDATION_MESSAGES)
    end
    push!(VALIDATION_MESSAGES, message)

    # Print based on severity — errors are always printed immediately
    is_error = (severity & Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0
    is_warning = (severity & Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0
    if is_error
        @error "Vulkan validation error" message
    elseif is_warning
        @warn "Vulkan validation warning" message
    end
    # Return VK_FALSE — can't throw from @cfunction callback (would corrupt Vulkan state).
    # Errors are collected in VALIDATION_MESSAGES and checked after Vulkan calls.
    return UInt32(0)
end

function setup_debug_messenger(instance::Vulkan.Instance)
    callback_ptr = @cfunction(
        debug_callback,
        UInt32,
        (Vulkan.DebugUtilsMessageSeverityFlagEXT,
         Vulkan.DebugUtilsMessageTypeFlagEXT,
         Ptr{Vulkan.VkCore.VkDebugUtilsMessengerCallbackDataEXT},
         Ptr{Cvoid})
    )

    messenger = Vulkan.DebugUtilsMessengerEXT(
        instance,
        callback_ptr;
        min_severity=Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
    )
    return messenger
end

"""
    get_validation_messages() -> Vector{String}

Return recent validation layer messages. Useful for diagnosing DEVICE_LOST errors.
"""
get_validation_messages() = copy(VALIDATION_MESSAGES)

"""
    clear_validation_messages!()

Clear the validation message buffer.
"""
clear_validation_messages!() = empty!(VALIDATION_MESSAGES)

"""
    check_validation_errors!(context::String)

Check if any validation errors were captured since the last check.
Throws `LavaError` with the error messages if any errors are found.
Call this after Vulkan operations that may trigger validation errors
(shader module creation, pipeline creation, dispatch recording).
"""
function check_validation_errors!(context::String)
    isempty(VALIDATION_MESSAGES) && return
    # Check for actual errors (not just warnings)
    # Validation messages containing "WARNING" are just warnings, not errors.
    errors = filter(m -> !contains(m, "WARNING"), VALIDATION_MESSAGES)
    isempty(errors) && return
    n = min(length(errors), 5)
    detail = join(["  [$i] $(first(errors[i], 300))" for i in 1:n], "\n")
    # Clear after reporting to avoid re-triggering
    empty!(VALIDATION_MESSAGES)
    throw(LavaError(
        context,
        "Vulkan validation error(s):\n$detail",
        "Fix the validation errors above before proceeding."
    ))
end
