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
    VkContext

Persistent Vulkan context holding device, queue, command pool, and reusable
command buffer + fence for compute dispatch.
"""
mutable struct VkContext
    instance::Vulkan.Instance
    physical_device::Vulkan.PhysicalDevice
    device::Vulkan.Device
    queue::Vulkan.Queue
    queue_family_index::UInt32
    cmd_pool::Vulkan.CommandPool
    cmd_buf::Vulkan.CommandBuffer
    fence::Vulkan.Fence
    device_name::String
    # Dispatch batching state
    recording::Bool
    dispatch_count::Int
    last_was_rt::Bool   # Track last dispatch type for barriers
    # Ray tracing (nothing if not available)
    rt_pipeline_properties::Union{Nothing, RTPipelineProperties}
end

const _vk_context = Ref{Union{Nothing, VkContext}}(nothing)

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
    # Platform-specific surface extension
    if Sys.islinux()
        # Try XCB first (most common on modern Linux), fall back to Xlib
        available_ext = unwrap(Vulkan.enumerate_instance_extension_properties())
        ext_names = Set(String(filter(!=('\0'), collect(e.extension_name))) for e in available_ext)
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

    # Chain required features
    bda_features = Vulkan.PhysicalDeviceBufferDeviceAddressFeatures(
        true,   # buffer_device_address
        false,  # buffer_device_address_capture_replay
        false   # buffer_device_address_multi_device
    )
    var_ptr_features = Vulkan.PhysicalDeviceVariablePointersFeatures(
        true,   # variable_pointers_storage_buffer
        true;   # variable_pointers
        next=bda_features
    )
    # Dynamic rendering (Vulkan 1.3 core) — no VkRenderPass/VkFramebuffer boilerplate
    dyn_rendering_features = Vulkan.PhysicalDeviceDynamicRenderingFeatures(
        true;   # dynamic_rendering
        next=var_ptr_features
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

    # Pre-allocate command buffer + fence
    alloc_info = Vulkan.CommandBufferAllocateInfo(
        cmd_pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1
    )
    cmd_bufs = unwrap(Vulkan.allocate_command_buffers(device, alloc_info))
    cmd_buf = cmd_bufs[1]

    fence = Vulkan.Fence(device)

    if has_rt
        @info "Lava: initialized Vulkan device with RT" device=dev_name queue_family=qf_idx handle_size=rt_props.shader_group_handle_size max_recursion=rt_props.max_ray_recursion_depth
    else
        @info "Lava: initialized Vulkan device (no RT)" device=dev_name queue_family=qf_idx
    end

    return VkContext(
        instance, phys_dev, device, queue, qf_idx,
        cmd_pool, cmd_buf, fence, dev_name,
        false, 0, false,
        rt_props
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
