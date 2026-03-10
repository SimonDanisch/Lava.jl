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
When the number of dispatches in the current CB segment exceeds `cb_split_threshold`,
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
    VkContext

Persistent Vulkan context holding device, queue, command pool, and batch-based
command buffer management for compute/graphics/RT dispatch.
"""
mutable struct VkContext
    instance::Vulkan.Instance
    physical_device::Vulkan.PhysicalDevice
    device::Vulkan.Device
    queue::Vulkan.Queue
    queue_family_index::UInt32
    cmd_pool::Vulkan.CommandPool
    device_name::String
    # Batch-based dispatch (replaces single cmd_buf/fence)
    active_batch::Union{Nothing, CommandBatch}   # Currently recording
    in_flight::Vector{CommandBatch}              # Submitted, not yet completed
    free_batches::Vector{CommandBatch}           # Completed, reusable
    free_cmd_bufs::Vector{Vulkan.CommandBuffer}  # Spare CBs for multi-CB splitting
    # Dedicated transfer command buffer + fence (separate from dispatch recording)
    # Prevents command buffer state corruption when _one_shot_copy runs between
    # dispatch recording and flush (NVIDIA validation: "active VkCommandBuffer")
    xfer_cmd_buf::Vulkan.CommandBuffer
    xfer_fence::Vulkan.Fence
    # Dedicated AS build command buffer + fence (separate from dispatch batches)
    # Prevents vkBeginCommandBuffer on active cmd_buf and vkQueueSubmit with in-use fence
    # when _build_as_on_gpu runs during dispatch recording.
    as_cmd_buf::Vulkan.CommandBuffer
    as_fence::Vulkan.Fence
    # Ray tracing (nothing if not available)
    rt_pipeline_properties::Union{Nothing, RTPipelineProperties}
    # Debug messenger (nothing if validation layers not available)
    debug_messenger::Any  # Union{Nothing, Vulkan.DebugUtilsMessengerEXT}
end

# Ring buffer of recent validation messages for context on DEVICE_LOST
const _validation_messages = String[]
const _max_validation_messages = 50

const _vk_context = Ref{Union{Nothing, VkContext}}(nothing)

# Set to true after DEVICE_LOST — prevents finalizers from calling Vulkan on invalid handles
const _device_lost = Ref(false)

# Callbacks for vk_reset_device! — registered by later-included files (pipeline.jl,
# command.jl, launch.jl, memory.jl) to clear their module-level caches.
const _reset_callbacks = Function[]

"""
    vk_context() -> VkContext

Get or create the global Vulkan context. Lazily initializes on first call.
"""
function vk_context()
    ctx = _vk_context[]
    if ctx === nothing
        ctx = _init_vulkan!()
        _vk_context[] = ctx
    end
    return ctx
end

vk_device() = vk_context().device
vk_queue() = vk_context().queue

"""
    vk_reset_device!()

Reinitialize the Vulkan device after DEVICE_LOST or other unrecoverable errors.
Destroys the old context and creates a fresh one. Clears all caches (pipelines,
kernels, arg buffers).

**WARNING**: All existing `LavaArray`s become INVALID after reset — their backing
GPU buffers no longer exist. You must reallocate all GPU data.
"""
function vk_reset_device!()
    _device_lost[] = false
    _vk_context[] = nothing
    # Don't destroy old Vulkan handles — they're invalid after DEVICE_LOST.
    # GC will eventually try to destroy them; _destroy_buffer! skips when
    # _device_lost was true (and we set it false only after clearing context).
    empty!(_validation_messages)
    # Run cleanup callbacks registered by other modules
    for cb in _reset_callbacks
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

"""Check if a recording is active (any batch is recording)."""
function has_active_recording(ctx::VkContext)
    batch = ctx.active_batch
    return batch !== nothing && batch.recording
end

function _init_vulkan!()
    # Create instance
    app_info = Vulkan.ApplicationInfo(
        v"0.1.0", v"0.1.0", v"1.3.0";
        application_name="Lava.jl",
        engine_name="Lava"
    )
    # Try with validation layers, fall back without
    layers = String[]
    available_layers = unwrap(Vulkan.enumerate_instance_layer_properties())
    for l in available_layers
        name = String(filter(!=('\0'), collect(l.layer_name)))
        if name == "VK_LAYER_KHRONOS_validation"
            push!(layers, "VK_LAYER_KHRONOS_validation")
            break
        end
    end

    # Instance extensions for surface/window support
    inst_extensions = String[
        "VK_KHR_surface",
    ]
    # Debug utils for validation message capture (works even without validation layers
    # for driver-level error reporting)
    available_ext = unwrap(Vulkan.enumerate_instance_extension_properties())
    ext_names = Set(String(filter(!=('\0'), collect(e.extension_name))) for e in available_ext)
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

    instance = Vulkan.Instance(
        layers,
        inst_extensions;
        application_info=app_info
    )

    # Set up debug messenger to capture validation/driver error messages
    debug_messenger = nothing
    if has_debug_utils
        debug_messenger = _setup_debug_messenger(instance)
    end

    # Pick physical device (prefer discrete GPU)
    phys_devs = unwrap(Vulkan.enumerate_physical_devices(instance))
    isempty(phys_devs) && throw(LavaError(
        "device initialization",
        "No Vulkan-capable GPU found",
        "Ensure Vulkan drivers are installed"))

    phys_dev = _pick_physical_device(phys_devs)
    props = Vulkan.get_physical_device_properties(phys_dev)
    dev_name = String(filter(!=('\0'), collect(props.device_name)))

    # Find queue family (prefer graphics+compute for graphics pipeline support)
    qf_idx = _find_graphics_compute_queue_family(phys_dev)

    # Create logical device with required features
    queue_ci = [Vulkan.DeviceQueueCreateInfo(qf_idx, [1.0f0])]

    # Check for RT extension support
    has_rt = _has_rt_extensions(phys_dev)

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
        false,  # shader_float_16
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

    # Chain RT features if available
    feature_chain = dyn_rendering_features
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

    # Enable shader int64, float64, geometry/tessellation shaders, wide lines
    core_features = Vulkan.PhysicalDeviceFeatures(
        :shader_int_64, :shader_float_64,
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

    # Command pool (resettable command buffers)
    cmd_pool = Vulkan.CommandPool(
        device, qf_idx;
        flags=Vulkan.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    )

    # Pre-allocate command buffers + fences:
    # [1]: initial dispatch batch (compute/graphics/RT)
    # [2]: transfer operations (_one_shot_copy staging downloads)
    # [3]: dedicated AS builds (separate from dispatch batches — THE FIX)
    # [4]: spare for free_batches pool
    alloc_info = Vulkan.CommandBufferAllocateInfo(
        cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 4
    )
    cmd_bufs = unwrap(Vulkan.allocate_command_buffers(device, alloc_info))
    initial_cmd_buf = cmd_bufs[1]
    xfer_cmd_buf = cmd_bufs[2]
    as_cmd_buf = cmd_bufs[3]
    spare_cmd_buf = cmd_bufs[4]

    initial_fence = Vulkan.Fence(device)
    xfer_fence = Vulkan.Fence(device)
    as_fence = Vulkan.Fence(device)
    spare_fence = Vulkan.Fence(device)

    # Create initial batch in free pool
    initial_batch = CommandBatch(initial_cmd_buf, initial_fence, false, 0, 0, false, Any[], String[], Vulkan.CommandBuffer[])
    spare_batch = CommandBatch(spare_cmd_buf, spare_fence, false, 0, 0, false, Any[], String[], Vulkan.CommandBuffer[])

    has_validation = !isempty(layers)
    if has_rt
        @info "Lava: initialized Vulkan device with RT" device=dev_name queue_family=qf_idx handle_size=rt_props.shader_group_handle_size max_recursion=rt_props.max_ray_recursion_depth validation=has_validation debug_utils=has_debug_utils
    else
        @info "Lava: initialized Vulkan device (no RT)" device=dev_name queue_family=qf_idx validation=has_validation debug_utils=has_debug_utils
    end
    if !has_validation
        @warn "Vulkan validation layers not found. Install vulkan-validationlayers for GPU error diagnostics."
    end

    # NVIDIA TDR workaround: use split-indirect path (downloads group count, dispatches
    # directly) instead of true vkCmdDispatchIndirect. Batching multiple heavy indirect
    # dispatches in a single command buffer triggers Xid 109 CTX SWITCH TIMEOUT on NVIDIA.
    # The split path adds a trivial sync per indirect dispatch (~4 bytes download) but
    # prevents TDR by flushing between dispatches.
    vendor_id = props.vendor_id
    is_nvidia = vendor_id == 0x10DE
    if is_nvidia
        _spirv_opt_enabled[] = true
        # Auto-flush was removed after Fix 9 (universal misaligned PSB detection).
        # TDR timeout was caused by misaligned PSB loads, not command buffer size.
        # auto_flush_threshold remains 0 (unlimited batching) for best performance.
        @debug "NVIDIA detected: spirv-opt=on, auto-flush=$(auto_flush_threshold[])"
    end

    # Initialize zero-alloc Vulkan function pointers for hot paths
    _cmd_pipeline_barrier_fptr[] = Vulkan.function_pointer(device, "vkCmdPipelineBarrier")

    return VkContext(
        instance, phys_dev, device, queue, qf_idx,
        cmd_pool, dev_name,
        nothing,  # active_batch
        CommandBatch[],  # in_flight
        CommandBatch[initial_batch, spare_batch],  # free_batches
        Vulkan.CommandBuffer[],  # free_cmd_bufs
        xfer_cmd_buf, xfer_fence,
        as_cmd_buf, as_fence,
        rt_props,
        debug_messenger
    )
end

function _pick_physical_device(devs)
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

function _find_graphics_compute_queue_family(phys_dev)
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
function _has_rt_extensions(phys_dev)
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

function _debug_callback(
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
    if length(_validation_messages) >= _max_validation_messages
        popfirst!(_validation_messages)
    end
    push!(_validation_messages, message)

    # Print based on severity — errors are always printed immediately
    is_error = (severity & Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0
    is_warning = (severity & Vulkan.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0
    if is_error
        @error "Vulkan validation error" message
    elseif is_warning
        @warn "Vulkan validation warning" message
    end
    # Return VK_FALSE — can't throw from @cfunction callback (would corrupt Vulkan state).
    # Errors are collected in _validation_messages and checked after Vulkan calls.
    return UInt32(0)
end

function _setup_debug_messenger(instance::Vulkan.Instance)
    callback_ptr = @cfunction(
        _debug_callback,
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
get_validation_messages() = copy(_validation_messages)

"""
    clear_validation_messages!()

Clear the validation message buffer.
"""
clear_validation_messages!() = empty!(_validation_messages)

"""
    check_validation_errors!(context::String)

Check if any validation errors were captured since the last check.
Throws `LavaError` with the error messages if any errors are found.
Call this after Vulkan operations that may trigger validation errors
(shader module creation, pipeline creation, dispatch recording).
"""
function check_validation_errors!(context::String)
    isempty(_validation_messages) && return
    # Check for actual errors (not just warnings)
    errors = filter(m -> !startswith(m, "(Warning"), _validation_messages)
    isempty(errors) && return
    n = min(length(errors), 5)
    detail = join(["  [$i] $(first(errors[i], 300))" for i in 1:n], "\n")
    # Clear after reporting to avoid re-triggering
    empty!(_validation_messages)
    throw(LavaError(
        context,
        "Vulkan validation error(s):\n$detail",
        "Fix the validation errors above before proceeding."
    ))
end
