# Vulkan device initialization for Lava.jl
#
# Singleton VkContext holds all persistent Vulkan state.
# Lazy initialization: first use triggers device creation.
#
# Required features: BufferDeviceAddress, VariablePointers, Int64, Float64

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

    instance = Vulkan.Instance(
        layers,
        String[];
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

    # Find compute queue family
    qf_idx = _find_compute_queue_family(phys_dev)

    # Create logical device with required features
    queue_ci = [Vulkan.DeviceQueueCreateInfo(qf_idx, [1.0f0])]

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
    # Enable shader int64, float64
    core_features = Vulkan.PhysicalDeviceFeatures(
        :shader_int_64, :shader_float_64
    )

    device = Vulkan.Device(
        phys_dev,
        queue_ci,
        [],     # layers
        [];     # extensions
        enabled_features=core_features,
        next=var_ptr_features
    )

    queue = Vulkan.get_device_queue(device, qf_idx, 0)

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

    @info "Lava: initialized Vulkan device" device=dev_name queue_family=qf_idx

    return VkContext(
        instance, phys_dev, device, queue, qf_idx,
        cmd_pool, cmd_buf, fence, dev_name,
        false, 0
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

function _find_compute_queue_family(phys_dev)
    qf_props = Vulkan.get_physical_device_queue_family_properties(phys_dev)
    # Prefer dedicated compute (no graphics)
    for (i, qfp) in enumerate(qf_props)
        if (qfp.queue_flags & Vulkan.QUEUE_COMPUTE_BIT) != 0 &&
           (qfp.queue_flags & Vulkan.QUEUE_GRAPHICS_BIT) == 0
            return UInt32(i - 1)
        end
    end
    # Fall back to any compute-capable queue
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
