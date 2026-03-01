# Vulkan memory management for Lava.jl
#
# Device-local buffers with BDA (Buffer Device Address) for kernel arguments.
# Staging buffer for CPU↔GPU transfers.

"""
    VkManagedBuffer

A GPU buffer with a known device address (BDA).
"""
mutable struct VkManagedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64     # BDA for PhysicalStorageBuffer access
    size::Int
end

# Strong references to keep Vulkan handles alive until explicit free
const _live_buffers = Set{VkManagedBuffer}()

"""
    vk_alloc(nbytes::Integer) -> VkManagedBuffer

Allocate a device-local buffer with BDA support.
"""
function vk_alloc(nbytes::Integer)
    dev = vk_device()
    nbytes = max(nbytes, 16)  # Vulkan requires non-zero size

    buf = Vulkan.Buffer(
        dev, nbytes,
        Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = _find_memory_type(
        mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    )

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(
        UInt32(0);  # device_mask (0 = all devices)
        flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    )
    memory = Vulkan.DeviceMemory(
        dev, mem_reqs.size, mem_type_idx;
        next=alloc_flags
    )
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    # Query BDA
    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)

    result = VkManagedBuffer(buf, memory, address, Int(nbytes))
    push!(_live_buffers, result)
    return result
end

"""
    vk_free!(buf::VkManagedBuffer)

Free a managed buffer's Vulkan resources.
"""
function vk_free!(buf::VkManagedBuffer)
    delete!(_live_buffers, buf)
    # Explicitly destroy in correct order: buffer before memory.
    # Vulkan.jl handles are refcounted — calling destructor() sets refcount to 0,
    # so the GC finalizer's second call will be a no-op (refcount underflows, iszero → false).
    buf.buffer.destructor()
    buf.memory.destructor()
    buf.size = 0
end

# ── Staging buffer for CPU↔GPU transfers ──

const _staging_buf = Ref{Union{Nothing, Tuple{Vulkan.Buffer, Vulkan.DeviceMemory, Ptr{Nothing}, Int}}}(nothing)

"""Get or grow the staging buffer to at least `nbytes`."""
function _get_staging(nbytes::Integer)
    existing = _staging_buf[]
    if existing !== nothing && existing[4] >= nbytes
        return existing
    end

    dev = vk_device()

    # Free old staging buffer — use handle destructors to avoid double-free from GC
    if existing !== nothing
        old_buf, old_mem, _, _ = existing
        Vulkan.unmap_memory(dev, old_mem)
        old_buf.destructor()
        old_mem.destructor()
    end

    # Allocate new (round up to power of 2)
    alloc_size = max(65536, nextpow(2, nbytes))
    buf = Vulkan.Buffer(
        dev, alloc_size,
        Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = _find_memory_type(
        mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
    )
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    # Map permanently
    mapped_ptr = unwrap(Vulkan.map_memory(dev, memory, 0, alloc_size))

    result = (buf, memory, mapped_ptr, Int(alloc_size))
    _staging_buf[] = result
    return result
end

"""
    upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)

Upload host data to a device-local buffer via staging.
"""
function upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)
    nbytes = length(host_data)
    nbytes == 0 && return

    staging_buf, _, mapped_ptr, _ = _get_staging(nbytes)
    unsafe_copyto!(Ptr{UInt8}(mapped_ptr), pointer(host_data), nbytes)

    _one_shot_copy(staging_buf, 0, dst.buffer, offset, nbytes)
end

"""
    download!(host_data::Vector{UInt8}, src::VkManagedBuffer; offset::Int=0)

Download data from a device-local buffer to host via staging.
"""
function download!(host_data::Vector{UInt8}, src::VkManagedBuffer; offset::Int=0)
    nbytes = length(host_data)
    nbytes == 0 && return

    staging_buf, _, mapped_ptr, _ = _get_staging(nbytes)
    _one_shot_copy(src.buffer, offset, staging_buf, 0, nbytes)
    unsafe_copyto!(pointer(host_data), Ptr{UInt8}(mapped_ptr), nbytes)
end

"""
    upload_typed!(dst::VkManagedBuffer, data::AbstractVector{T}; offset::Int=0)

Upload a typed array to a device-local buffer.
"""
function upload_typed!(dst::VkManagedBuffer, data::AbstractVector{T}; offset::Int=0) where T
    bytes = Vector{UInt8}(reinterpret(UInt8, vec(collect(data))))
    upload!(dst, bytes; offset)
end

"""
    download_typed!(data::AbstractVector{T}, src::VkManagedBuffer; offset::Int=0)

Download data from a device-local buffer into a typed array.
"""
function download_typed!(data::AbstractVector{T}, src::VkManagedBuffer; offset::Int=0) where T
    nbytes = length(data) * sizeof(T)
    bytes = Vector{UInt8}(undef, nbytes)
    download!(bytes, src; offset)
    GC.@preserve data bytes begin
        unsafe_copyto!(Ptr{UInt8}(pointer(data)), Ptr{UInt8}(pointer(bytes)), nbytes)
    end
end

# ── Internal helpers ──

function _one_shot_copy(src::Vulkan.Buffer, src_offset::Integer,
                        dst::Vulkan.Buffer, dst_offset::Integer,
                        nbytes::Integer)
    ctx = vk_context()
    dev = ctx.device
    cmd = ctx.cmd_buf

    # Auto-flush pending dispatches before memory transfer
    if ctx.recording
        vk_flush!()
    end

    unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    )))

    region = Vulkan.BufferCopy(UInt64(src_offset), UInt64(dst_offset), UInt64(nbytes))
    Vulkan.cmd_copy_buffer(cmd, src, dst, [region])

    unwrap(Vulkan.end_command_buffer(cmd))

    # Submit and wait using the context's reusable fence
    submit_info = Vulkan.SubmitInfo([], [], [cmd], [])
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=ctx.fence))
    unwrap(Vulkan.wait_for_fences(dev, [ctx.fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [ctx.fence]))
end

"""
    VkMappedBuffer

A host-visible, device-accessible buffer with a permanently mapped pointer.
Used for kernel argument buffers — write directly via memcpy, no staging needed.
"""
mutable struct VkMappedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64     # BDA
    mapped_ptr::Ptr{UInt8}
    size::Int
end

"""
    vk_alloc_mapped(nbytes::Integer) -> VkMappedBuffer

Allocate a host-visible, device-accessible buffer with BDA support.
Permanently mapped — write directly to `mapped_ptr`.
On AMD RDNA, uses resizable BAR memory (device-local + host-visible).
Falls back to host-visible only if BAR is unavailable.
"""
function vk_alloc_mapped(nbytes::Integer)
    dev = vk_device()
    nbytes = max(nbytes, 16)

    buf = Vulkan.Buffer(
        dev, nbytes,
        Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)

    # Try device-local + host-visible (BAR memory, fastest)
    preferred_flags = Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                      Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
    mem_type_idx = _find_memory_type_optional(mem_reqs.memory_type_bits, preferred_flags)

    # Fall back to host-visible + coherent only
    if mem_type_idx === nothing
        fallback_flags = Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
        mem_type_idx = _find_memory_type(mem_reqs.memory_type_bits, fallback_flags)
    end

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(
        UInt32(0);
        flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    )
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)

    mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, nbytes)))

    result = VkMappedBuffer(buf, memory, address, mapped_ptr, Int(nbytes))
    return result
end

function _find_memory_type_optional(type_bits::UInt32, required_flags)
    ctx = vk_context()
    mem_props = Vulkan.get_physical_device_memory_properties(ctx.physical_device)
    for i in 0:(mem_props.memory_type_count - 1)
        if (type_bits & (UInt32(1) << i)) != 0
            mt = mem_props.memory_types[i + 1]
            if (mt.property_flags & required_flags) == required_flags
                return UInt32(i)
            end
        end
    end
    return nothing
end

function _find_memory_type(type_bits::UInt32, required_flags)
    ctx = vk_context()
    mem_props = Vulkan.get_physical_device_memory_properties(ctx.physical_device)

    for i in 0:(mem_props.memory_type_count - 1)
        if (type_bits & (UInt32(1) << i)) != 0
            mt = mem_props.memory_types[i + 1]
            if (mt.property_flags & required_flags) == required_flags
                return UInt32(i)
            end
        end
    end
    throw(LavaError(
        "memory allocation",
        "No suitable memory type found for flags $required_flags",
        "Check GPU memory capabilities"))
end
