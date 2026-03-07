# Vulkan memory management for Lava.jl
#
# Device-local buffers with BDA (Buffer Device Address) for kernel arguments.
# Staging buffer for CPU↔GPU transfers.

"""
    VkManagedBuffer

A GPU buffer with a known device address (BDA).
When `mapped_ptr` is non-null, the buffer is in BAR memory (host-visible + device-local)
and can be read/written directly from the CPU without staging copies.
"""
mutable struct VkManagedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64     # BDA for PhysicalStorageBuffer access
    mapped_ptr::Ptr{UInt8}  # Non-null for unified/BAR memory
    size::Int
end

# Strong references to keep Vulkan handles alive until explicit free
const _live_buffers = Set{VkManagedBuffer}()

# Deferred free list: buffers whose GC finalizer fired while a command buffer
# was recording or executing. Destroying them immediately would free GPU memory
# that the in-flight command buffer still references via BDA → DEVICE_LOST.
# Processed after vk_flush!() completes (GPU is idle, safe to destroy).
const DEFERRED_FREES = VkManagedBuffer[]

# ── GPU memory pressure tracking ──
# Julia's GC doesn't know about GPU memory. LavaArray wrappers are ~50 bytes on
# the CPU heap, but back 100+ MB of VRAM each. Without pressure signals, GC
# never fires and dead GPU buffers accumulate until OOM.
# Solution: track live GPU bytes and trigger GC.gc(false) proactively,
# matching AMDGPU.jl's maybe_collect() pattern.
const GPU_LIVE_BYTES = Threads.Atomic{Int}(0)
const GPU_LAST_INCR_GC_TIME = Ref(0.0)
const GPU_LAST_FULL_GC_TIME = Ref(0.0)

"""
    maybe_collect()

Trigger GC if GPU memory pressure is high. Julia's GC doesn't know about
VRAM — LavaArray wrappers are ~50 bytes on the CPU heap but back hundreds of
MB of GPU memory. Without this, dead GPU buffers accumulate until OOM.

Two tiers (with separate timers so incremental GC doesn't starve full GC):
- >256 MiB tracked: incremental GC (rate-limited to every 100ms)
- >512 MiB tracked: full GC (rate-limited to every 2s)

Called from `vk_alloc` / `vk_alloc_unified` before each allocation.
"""
function maybe_collect()
    live = GPU_LIVE_BYTES[]
    live < 256 * 1024 * 1024 && return  # <256 MiB: no pressure
    t = time()
    # Full GC has its own timer — incremental GC must not starve it
    if live > 512 * 1024 * 1024 && (t - GPU_LAST_FULL_GC_TIME[]) > 2.0
        GPU_LAST_FULL_GC_TIME[] = t
        GPU_LAST_INCR_GC_TIME[] = t
        GC.gc(true)
    elseif (t - GPU_LAST_INCR_GC_TIME[]) > 0.1
        GPU_LAST_INCR_GC_TIME[] = t
        GC.gc(false)
    end
    return
end

"""
    vk_alloc(nbytes::Integer) -> VkManagedBuffer

Allocate a device-local buffer with BDA support.
"""
function vk_alloc(nbytes::Integer; extra_usage::UInt32=UInt32(0))
    maybe_collect()
    dev = vk_device()
    nbytes = max(nbytes, 16)  # Vulkan requires non-zero size

    usage = Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
            Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT
    if extra_usage != UInt32(0)
        usage |= Vulkan.BufferUsageFlag(extra_usage)
    end

    buf = Vulkan.Buffer(
        dev, nbytes,
        usage,
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

    result = VkManagedBuffer(buf, memory, address, Ptr{UInt8}(0), Int(nbytes))
    push!(_live_buffers, result)
    Threads.atomic_add!(GPU_LIVE_BYTES, nbytes)
    return result
end

"""
    vk_free!(buf::VkManagedBuffer)

Free a managed buffer's Vulkan resources.

If a command buffer is currently recording or executing (ctx.recording == true),
the destruction is deferred until after vk_flush!() completes. This prevents
DEVICE_LOST from GC finalizers freeing GPU memory that the in-flight command
buffer still references via BDA addresses.
"""
function vk_free!(buf::VkManagedBuffer)
    buf.size == 0 && return  # Already freed

    delete!(_live_buffers, buf)

    # Defer destruction while GPU may be using this buffer.
    # GC finalizers can fire at any point — if a command buffer is recording,
    # the buffer's BDA address may be embedded in a dispatch's arg buffer.
    # Destroying it now would make the GPU read freed memory → page fault.
    ctx = _vk_context[]
    if ctx !== nothing && ctx.recording
        push!(DEFERRED_FREES, buf)
        return
    end

    _destroy_buffer!(buf)
end

"""Actually destroy a buffer's Vulkan resources. Called from vk_free! or flush_deferred_frees!."""
function _destroy_buffer!(buf::VkManagedBuffer)
    # Unmap if this was a unified/BAR buffer
    if buf.mapped_ptr != Ptr{UInt8}(0)
        Vulkan.unmap_memory(vk_device(), buf.memory)
        buf.mapped_ptr = Ptr{UInt8}(0)
    end
    # Explicitly destroy in correct order: buffer before memory.
    # Vulkan.jl handles are refcounted — calling destructor() sets refcount to 0,
    # so the GC finalizer's second call will be a no-op (refcount underflows, iszero → false).
    buf.buffer.destructor()
    buf.memory.destructor()
    Threads.atomic_sub!(GPU_LIVE_BYTES, buf.size)
    buf.size = 0
end

"""Process deferred buffer frees after GPU is idle. Called from vk_flush!()."""
function flush_deferred_frees!()
    isempty(DEFERRED_FREES) && return
    n = length(DEFERRED_FREES)
    # Debug warning: deferred frees indicate a GC finalizer fired during recording.
    # This is safely handled (Layer 2), but frequent occurrences suggest Layer 1
    # (keep_data_alive!) is missing on a dispatch path. Investigate if this fires often.
    @debug "Lava: flushing $n deferred buffer frees (GC fired during recording)"
    for buf in DEFERRED_FREES
        _destroy_buffer!(buf)
    end
    empty!(DEFERRED_FREES)
end

# ── Staging buffer for CPU↔GPU transfers ──

const STAGING_BUF = Ref{Union{Nothing, Tuple{Vulkan.Buffer, Vulkan.DeviceMemory, Ptr{Nothing}, Int}}}(nothing)

"""Get or grow the staging buffer to at least `nbytes`."""
function get_staging(nbytes::Integer)
    existing = STAGING_BUF[]
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
    STAGING_BUF[] = result
    return result
end

"""
    upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)

Upload host data to a device-local buffer via staging.
"""
function upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)
    nbytes = length(host_data)
    nbytes == 0 && return

    # Fast path: if buffer is in BAR memory (mapped), write directly via mapped ptr.
    if dst.mapped_ptr != Ptr{UInt8}(0)
        ctx = vk_context()
        if ctx.recording
            vk_flush!()
        end
        unsafe_copyto!(dst.mapped_ptr + offset, pointer(host_data), nbytes)
        return
    end

    staging_buf, _, mapped_ptr, _ = get_staging(nbytes)
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

    # Fast path: if buffer is in BAR memory (mapped), just flush and read directly.
    # Saves one fence wait by avoiding the staging copy command.
    if src.mapped_ptr != Ptr{UInt8}(0)
        ctx = vk_context()
        if ctx.recording
            vk_flush!()
        end
        unsafe_copyto!(pointer(host_data), src.mapped_ptr + offset, nbytes)
        return
    end

    staging_buf, _, mapped_ptr, _ = get_staging(nbytes)
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

"""
    vk_alloc_unified(nbytes::Integer) -> VkManagedBuffer

Allocate a unified (BAR) buffer as VkManagedBuffer with `mapped_ptr` set.
Used by `KA.allocate(; unified=true)` for host-readable GPU buffers.
"""
function vk_alloc_unified(nbytes::Integer)
    maybe_collect()
    mapped = vk_alloc_mapped(max(nbytes, 16))
    managed = VkManagedBuffer(mapped.buffer, mapped.memory, mapped.address,
                               mapped.mapped_ptr, Int(mapped.size))
    push!(_live_buffers, managed)
    Threads.atomic_add!(GPU_LIVE_BYTES, managed.size)
    return managed
end

# ── Indirect dispatch buffer pool ──

"""
    VkIndirectBuffer — A small mapped buffer for VkDispatchIndirectCommand (3×UInt32).
    Has INDIRECT_BUFFER_BIT + STORAGE_BUFFER_BIT + SHADER_DEVICE_ADDRESS_BIT.
"""
mutable struct VkIndirectBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64
    mapped_ptr::Ptr{UInt8}
    size::Int
end

const _indirect_buffers = VkIndirectBuffer[]
const _indirect_buffer_idx = Ref(0)

"""Get or create an indirect dispatch buffer from the pool."""
function _get_indirect_buffer()
    _indirect_buffer_idx[] += 1
    idx = _indirect_buffer_idx[]

    while length(_indirect_buffers) < idx
        push!(_indirect_buffers, _alloc_indirect_buffer())
    end

    return _indirect_buffers[idx]
end

"""Reset indirect buffer pool index after flush."""
function reset_indirect_buffer_pool!()
    _indirect_buffer_idx[] = 0
end

function _alloc_indirect_buffer()
    dev = vk_device()
    nbytes = 16  # 3×UInt32 = 12 bytes, round to 16

    buf = Vulkan.Buffer(
        dev, nbytes,
        Vulkan.BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)

    # Prefer device-local + host-visible (BAR), fall back to host-visible
    preferred = Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
    mem_type_idx = _find_memory_type_optional(mem_reqs.memory_type_bits, preferred)
    if mem_type_idx === nothing
        fallback = Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT
        mem_type_idx = _find_memory_type(mem_reqs.memory_type_bits, fallback)
    end

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(
        UInt32(0); flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    addr_info = Vulkan.BufferDeviceAddressInfo(buf)
    address = Vulkan.get_buffer_device_address(dev, addr_info)
    mapped_ptr = Ptr{UInt8}(unwrap(Vulkan.map_memory(dev, memory, 0, nbytes)))

    return VkIndirectBuffer(buf, memory, address, mapped_ptr, Int(nbytes))
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
