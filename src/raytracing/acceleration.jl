# Acceleration structure build for Lava.jl
#
# BLAS (bottom-level): triangle geometry
# TLAS (top-level): instances referencing BLASes
#
# Buffers need ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR for vertex/index data
# and ACCELERATION_STRUCTURE_STORAGE_BIT_KHR for AS backing storage.

"""
    LavaBLAS

Bottom-level acceleration structure wrapping a VkAccelerationStructureKHR.
"""
struct LavaBLAS
    accel::Vulkan.AccelerationStructureKHR
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64  # AS device address (for TLAS instances)
end

"""
    LavaTLAS

Top-level acceleration structure wrapping a VkAccelerationStructureKHR.
"""
struct LavaTLAS
    accel::Vulkan.AccelerationStructureKHR
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    _blas_refs::Vector{LavaBLAS}  # prevent GC of BLAS backing buffers
end

"""
    build_blas(vertices::Vector{NTuple{3,Float32}}, indices::Vector{UInt32}) -> LavaBLAS

Build a bottom-level acceleration structure from triangle vertices and indices.
`indices` are triplets (0-based) defining triangles.
"""
function build_blas(vertices::Vector{NTuple{3,Float32}}, indices::Vector{UInt32};
                    opaque::Bool=true)
    ctx = vk_context()
    dev = ctx.device

    # Upload vertex data to device buffer
    vertex_bytes = reinterpret(UInt8, vertices)
    vertex_buf, vertex_mem, vertex_addr = _create_as_input_buffer(vertex_bytes)

    # Upload index data
    index_bytes = reinterpret(UInt8, indices)
    index_buf, index_mem, index_addr = _create_as_input_buffer(index_bytes)

    n_triangles = UInt32(length(indices) ÷ 3)
    max_vertex = UInt32(length(vertices) - 1)

    vfmt = UInt32(Vulkan.FORMAT_R32G32B32_SFLOAT)
    vstride = UInt64(sizeof(NTuple{3,Float32}))
    itype = UInt32(Vulkan.INDEX_TYPE_UINT32)
    build_flags = UInt32(Vulkan.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    geo_flags = opaque ? UInt32(Vulkan.GEOMETRY_OPAQUE_BIT_KHR) : UInt32(0)

    # Query build sizes using correctly-packed geometry
    sizes = _query_as_build_sizes(dev;
        as_type=UInt32(1),  # BOTTOM_LEVEL
        build_flags,
        geometry_type=:triangles,
        vertex_format=vfmt, vertex_addr, vertex_stride=vstride,
        max_vertex, index_type=itype, index_addr,
        geo_flags,
        max_primitive_count=n_triangles)

    # Create AS backing buffer + AS object
    as_buf, as_mem = _create_as_storage_buffer(sizes.acceleration_structure_size)
    as_ci = Vulkan.AccelerationStructureCreateInfoKHR(
        as_buf, UInt64(0), sizes.acceleration_structure_size,
        Vulkan.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    )
    accel = Vulkan.AccelerationStructureKHR(dev, as_ci)

    # Create scratch buffer
    scratch_buf, scratch_mem, scratch_addr = _create_scratch_buffer(sizes.build_scratch_size)

    # Build on GPU — input buffers must stay alive during the GPU build.
    # In batched mode (_as_batching), stash refs so they survive until submit.
    # In non-batched mode, GC.@preserve covers the synchronous submit+wait.
    input_refs = (vertex_buf, vertex_mem, index_buf, index_mem, scratch_buf, scratch_mem)
    if _as_batching[]
        push!(_as_batch_preserves, input_refs)
    end
    GC.@preserve vertex_buf vertex_mem index_buf index_mem scratch_buf scratch_mem begin
        _build_as_on_gpu(ctx, accel, scratch_addr;
            as_type=UInt32(1), build_flags,
            geometry_type=:triangles,
            vertex_format=vfmt, vertex_addr, vertex_stride=vstride,
            max_vertex, index_type=itype, index_addr,
            geo_flags, primitive_count=n_triangles)
    end

    # Eagerly free temporary buffers to reclaim VRAM — but only when NOT inside
    # as_build() batching, where the command buffer hasn't been submitted yet.
    if !_as_batching[]
        finalize(scratch_buf); finalize(scratch_mem)
        finalize(index_buf); finalize(index_mem)
        finalize(vertex_buf); finalize(vertex_mem)
    end

    # Get AS device address
    addr_info = Vulkan.AccelerationStructureDeviceAddressInfoKHR(accel)
    as_addr = Vulkan.get_acceleration_structure_device_address_khr(dev, addr_info)

    return LavaBLAS(accel, as_buf, as_mem, as_addr)
end

"""
    build_tlas(blas_list::Vector{LavaBLAS};
               transforms=nothing, custom_indices=nothing) -> LavaTLAS

Build a top-level acceleration structure from a list of BLASes.
Each BLAS becomes one instance. Default transform is identity.
"""
function build_tlas(blas_list::Vector{LavaBLAS};
                    transforms::Union{Nothing, Vector{NTuple{12,Float32}}}=nothing,
                    custom_indices::Union{Nothing, Vector{UInt32}}=nothing)
    ctx = vk_context()
    dev = ctx.device
    n_instances = length(blas_list)

    # Build instance buffer (64 bytes per instance, manually packed)
    instance_data = Vector{UInt8}(undef, 64 * n_instances)
    for i in 1:n_instances
        offset = (i - 1) * 64
        _pack_as_instance!(instance_data, offset, blas_list[i].address;
            transform=transforms === nothing ? nothing : transforms[i],
            custom_index=custom_indices === nothing ? UInt32(0) : custom_indices[i],
        )
    end

    inst_buf, inst_mem, inst_addr = _create_as_input_buffer(instance_data)

    build_flags = UInt32(Vulkan.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)

    # Query sizes using correctly-packed geometry
    sizes = _query_as_build_sizes(dev;
        as_type=UInt32(0),  # TOP_LEVEL
        build_flags,
        geometry_type=:instances,
        instance_addr=inst_addr,
        max_primitive_count=UInt32(n_instances))

    # Create AS backing buffer + AS
    as_buf, as_mem = _create_as_storage_buffer(sizes.acceleration_structure_size)
    as_ci = Vulkan.AccelerationStructureCreateInfoKHR(
        as_buf, UInt64(0), sizes.acceleration_structure_size,
        Vulkan.ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    )
    accel = Vulkan.AccelerationStructureKHR(dev, as_ci)

    # Scratch buffer
    scratch_buf, scratch_mem, scratch_addr = _create_scratch_buffer(sizes.build_scratch_size)

    # Build on GPU — preserve buffers from GC during GPU build
    input_refs_tlas = (inst_buf, inst_mem, scratch_buf, scratch_mem)
    if _as_batching[]
        push!(_as_batch_preserves, input_refs_tlas)
    end
    GC.@preserve inst_buf inst_mem scratch_buf scratch_mem begin
        _build_as_on_gpu(ctx, accel, scratch_addr;
            as_type=UInt32(0), build_flags,
            geometry_type=:instances,
            instance_addr=inst_addr,
            primitive_count=UInt32(n_instances))
    end

    # Eagerly free temporary buffers to reclaim VRAM — but only when NOT inside
    # as_build() batching, where the command buffer hasn't been submitted yet.
    if !_as_batching[]
        finalize(scratch_buf); finalize(scratch_mem)
        finalize(inst_buf); finalize(inst_mem)
    end

    # Keep unique BLASes alive — TLAS references them by device address.
    unique_blas = unique(blas_list)

    return LavaTLAS(accel, as_buf, as_mem, unique_blas)
end

# ── Internal helpers ──

"""Pack a VkAccelerationStructureInstanceKHR into `buf` at byte `offset`.
Layout (64 bytes):
  - [0:47]  transform: VkTransformMatrixKHR (3×4 float32, row-major)
  - [48:50] instanceCustomIndex (24 bits)
  - [51]    mask (8 bits)
  - [52:54] instanceShaderBindingTableRecordOffset (24 bits)
  - [55]    flags (8 bits)
  - [56:63] accelerationStructureReference (uint64)
"""
function _pack_as_instance!(buf::Vector{UInt8}, offset::Int, blas_addr::UInt64;
                            transform::Union{Nothing, NTuple{12,Float32}}=nothing,
                            custom_index::UInt32=UInt32(0),
                            mask::UInt8=0xff,
                            sbt_offset::UInt32=UInt32(0),
                            flags::UInt8=0x00)
    # Identity transform if not specified (row-major 3×4)
    if transform === nothing
        transform = (1f0, 0f0, 0f0, 0f0,
                     0f0, 1f0, 0f0, 0f0,
                     0f0, 0f0, 1f0, 0f0)
    end

    # Transform: 12 × Float32 = 48 bytes
    for j in 1:12
        unsafe_store!(Ptr{Float32}(pointer(buf, offset + 1 + (j-1)*4)), transform[j])
    end

    # instanceCustomIndex (24 bits) + mask (8 bits) = 4 bytes at offset 48
    val_48 = (custom_index & 0x00FFFFFF) | (UInt32(mask) << 24)
    unsafe_store!(Ptr{UInt32}(pointer(buf, offset + 49)), val_48)

    # instanceShaderBindingTableRecordOffset (24 bits) + flags (8 bits) = 4 bytes at offset 52
    val_52 = (sbt_offset & 0x00FFFFFF) | (UInt32(flags) << 24)
    unsafe_store!(Ptr{UInt32}(pointer(buf, offset + 53)), val_52)

    # accelerationStructureReference = BLAS device address (8 bytes at offset 56)
    unsafe_store!(Ptr{UInt64}(pointer(buf, offset + 57)), blas_addr)

    return nothing
end

"""Create a HOST_VISIBLE buffer pool for AS input data (vertices/indices).

Unlike `_create_as_input_buffer` which uploads specific data, this allocates
an empty mappable buffer of `nbytes` for the caller to fill via map/memcpy/unmap.
Returns (buffer, memory, base_device_address).
"""
function _create_as_input_pool(nbytes::UInt64)
    ctx = vk_context()
    dev = ctx.device

    buf = Vulkan.Buffer(
        dev, max(nbytes, 16),
        Vulkan.BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[],
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = _find_memory_type(
        mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(UInt32(0);
        flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    addr = Vulkan.get_buffer_device_address(dev, Vulkan.BufferDeviceAddressInfo(buf))

    return buf, memory, addr
end

"""Create a device-local buffer for AS input (vertex/index/instance data)."""
function _create_as_input_buffer(data::Union{Vector{UInt8}, Base.ReinterpretArray})
    ctx = vk_context()
    dev = ctx.device
    nbytes = length(data)

    buf = Vulkan.Buffer(
        dev, max(nbytes, 16),
        Vulkan.BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[],
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = _find_memory_type(
        mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(UInt32(0);
        flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    # Upload
    ptr = unwrap(Vulkan.map_memory(dev, memory, 0, nbytes))
    unsafe_copyto!(Ptr{UInt8}(ptr), pointer(data isa Base.ReinterpretArray ? collect(data) : data), nbytes)
    Vulkan.unmap_memory(dev, memory)

    # Get BDA
    addr = Vulkan.get_buffer_device_address(dev, Vulkan.BufferDeviceAddressInfo(buf))

    return buf, memory, addr
end

"""Create a device-local buffer for AS storage."""
function _create_as_storage_buffer(nbytes)
    ctx = vk_context()
    dev = ctx.device

    buf = Vulkan.Buffer(
        dev, max(nbytes, 16),
        Vulkan.BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[],
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = _find_memory_type(
        mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(UInt32(0);
        flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    return buf, memory
end

"""Create a scratch buffer for AS build."""
function _create_scratch_buffer(nbytes)
    ctx = vk_context()
    dev = ctx.device
    phys = ctx.physical_device

    # Vulkan requires scratch device addresses to satisfy
    # minAccelerationStructureScratchOffsetAlignment.
    as_props2 = Vulkan.get_physical_device_properties_2(
        phys, Vulkan.PhysicalDeviceAccelerationStructurePropertiesKHR)
    scratch_align = UInt64(as_props2.next.min_acceleration_structure_scratch_offset_alignment)
    scratch_align = max(scratch_align, UInt64(1))
    aligned_size = max(UInt64(nbytes) + (scratch_align - UInt64(1)), scratch_align)

    buf = Vulkan.Buffer(
        dev, aligned_size,
        Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[],
    )

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = _find_memory_type(
        mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(UInt32(0);
        flags=Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = Vulkan.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    base_addr = Vulkan.get_buffer_device_address(dev, Vulkan.BufferDeviceAddressInfo(buf))
    addr = ((base_addr + scratch_align - UInt64(1)) ÷ scratch_align) * scratch_align

    return buf, memory, addr
end

const _VkBRI = Vulkan.VulkanCore.LibVulkan.VkAccelerationStructureBuildRangeInfoKHR

# VulkanCore.jl alignment bug: VkDeviceOrHostAddressConstKHR is NTuple{8,UInt8} (alignment 1)
# but in C it's a union of uint64_t/void* (alignment 8). This causes misaligned fields in:
#   VkAccelerationStructureGeometryKHR:       Julia sizeof=88 vs C sizeof=96
#   VkAccelerationStructureGeometryDataKHR:   geometry union at offset 20 vs C offset 24
# We construct all AS-related C structs manually with correct field offsets.

const _C_SIZEOF_AS_GEOMETRY_KHR = 96

# VkStructureType values
const _VK_STYPE_BGI = Int32(1000150000)   # BUILD_GEOMETRY_INFO
const _VK_STYPE_SIZES = Int32(1000150020) # BUILD_SIZES_INFO (VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR)
const _VK_STYPE_GEO = Int32(1000150006)   # GEOMETRY
const _VK_STYPE_TRI = Int32(1000150005)   # GEOMETRY_TRIANGLES_DATA
const _VK_STYPE_INST = Int32(1000150004)  # GEOMETRY_INSTANCES_DATA

"""Pack VkAccelerationStructureGeometryKHR (96 bytes, correct C layout) into `buf`.

C layout:
  0:  sType (4)  |  4: pad (4)  |  8: pNext (8)
  16: geometryType (4)  |  20: pad (4)
  24: geometry union (64 bytes)
  88: flags (4)  |  92: pad (4)
"""
function _pack_geometry!(buf::Vector{UInt8}, offset::Int;
        geometry_type::Symbol,
        # For triangles:
        vertex_format::UInt32=UInt32(0), vertex_addr::UInt64=UInt64(0),
        vertex_stride::UInt64=UInt64(0), max_vertex::UInt32=UInt32(0),
        index_type::UInt32=UInt32(0), index_addr::UInt64=UInt64(0),
        transform_addr::UInt64=UInt64(0),
        # For instances:
        instance_addr::UInt64=UInt64(0),
        geo_flags::UInt32=UInt32(0))
    p = pointer(buf, offset + 1)
    unsafe_store!(Ptr{Int32}(p), _VK_STYPE_GEO)           # sType @ 0
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 8), C_NULL)         # pNext @ 8

    q = p + 24  # geometry union start
    if geometry_type == :triangles
        unsafe_store!(Ptr{UInt32}(p + 16), UInt32(0))       # geometryType = TRIANGLES @ 16
        # Triangles sub-struct within union
        unsafe_store!(Ptr{Int32}(q), _VK_STYPE_TRI)                  # sType @ 0
        unsafe_store!(Ptr{Ptr{Nothing}}(q + 8), C_NULL)              # pNext @ 8
        unsafe_store!(Ptr{UInt32}(q + 16), vertex_format)            # vertexFormat @ 16
        unsafe_store!(Ptr{UInt64}(q + 24), vertex_addr)              # vertexData @ 24
        unsafe_store!(Ptr{UInt64}(q + 32), vertex_stride)            # vertexStride @ 32
        unsafe_store!(Ptr{UInt32}(q + 40), max_vertex)               # maxVertex @ 40
        unsafe_store!(Ptr{UInt32}(q + 44), index_type)               # indexType @ 44
        unsafe_store!(Ptr{UInt64}(q + 48), index_addr)               # indexData @ 48
        unsafe_store!(Ptr{UInt64}(q + 56), transform_addr)           # transformData @ 56
    elseif geometry_type == :instances
        unsafe_store!(Ptr{UInt32}(p + 16), UInt32(2))       # geometryType = INSTANCES (VK_GEOMETRY_TYPE_INSTANCES_KHR=2) @ 16
        # Instances sub-struct within union
        unsafe_store!(Ptr{Int32}(q), _VK_STYPE_INST)                 # sType @ 0
        unsafe_store!(Ptr{Ptr{Nothing}}(q + 8), C_NULL)              # pNext @ 8
        unsafe_store!(Ptr{UInt32}(q + 16), UInt32(0))                # arrayOfPointers @ 16
        unsafe_store!(Ptr{UInt64}(q + 24), instance_addr)            # data @ 24
    else
        error("Unknown geometry type: $geometry_type")
    end
    unsafe_store!(Ptr{UInt32}(p + 88), geo_flags)           # flags @ 88
    return nothing
end

"""Pack VkAccelerationStructureBuildGeometryInfoKHR (80 bytes, C layout matches Julia)."""
function _pack_build_geometry_info!(buf::Vector{UInt8}, offset::Int;
        as_type::UInt32, mode::UInt32=UInt32(0),
        build_flags::UInt32=UInt32(0),
        src_as::Ptr{Nothing}=C_NULL, dst_as::Ptr{Nothing}=C_NULL,
        geometry_count::UInt32=UInt32(1),
        p_geometries::Ptr{Nothing}=C_NULL,
        scratch_addr::UInt64=UInt64(0))
    p = pointer(buf, offset + 1)
    unsafe_store!(Ptr{Int32}(p), _VK_STYPE_BGI)              # sType @ 0
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 8), C_NULL)           # pNext @ 8
    unsafe_store!(Ptr{UInt32}(p + 16), as_type)               # type @ 16
    unsafe_store!(Ptr{UInt32}(p + 20), build_flags)           # flags @ 20
    unsafe_store!(Ptr{UInt32}(p + 24), mode)                  # mode @ 24
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 32), src_as)          # srcAccelerationStructure @ 32
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 40), dst_as)          # dstAccelerationStructure @ 40
    unsafe_store!(Ptr{UInt32}(p + 48), geometry_count)        # geometryCount @ 48
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 56), p_geometries)    # pGeometries @ 56
    unsafe_store!(Ptr{Ptr{Nothing}}(p + 64), C_NULL)          # ppGeometries @ 64
    unsafe_store!(Ptr{UInt64}(p + 72), scratch_addr)          # scratchData @ 72
    return nothing
end

"""Query AS build sizes using correctly-packed C structs.

Calls vkGetAccelerationStructureBuildSizesKHR directly via ccall to work around
VulkanCore.jl alignment bug in VkAccelerationStructureGeometryKHR.
"""
function _query_as_build_sizes(dev::Vulkan.Device; as_type::UInt32,
        build_flags::UInt32=UInt32(0), max_primitive_count::UInt32=UInt32(0),
        kwargs...)
    geo_buf = zeros(UInt8, _C_SIZEOF_AS_GEOMETRY_KHR)
    _pack_geometry!(geo_buf, 0; kwargs...)

    bgi_buf = zeros(UInt8, 80)

    # Output struct: VkAccelerationStructureBuildSizesInfoKHR
    # sType(4) + pad(4) + pNext(8) + accelerationStructureSize(8) + updateScratchSize(8) + buildScratchSize(8) = 40
    sizes_buf = zeros(UInt8, 40)
    unsafe_store!(Ptr{Int32}(pointer(sizes_buf)), _VK_STYPE_SIZES)

    fptr = Vulkan.function_pointer(dev, "vkGetAccelerationStructureBuildSizesKHR")

    GC.@preserve geo_buf bgi_buf sizes_buf begin
        geo_ptr = pointer(geo_buf)
        _pack_build_geometry_info!(bgi_buf, 0;
            as_type, build_flags,
            geometry_count=UInt32(1),
            p_geometries=Ptr{Nothing}(geo_ptr))

        max_counts = Ref(max_primitive_count)

        GC.@preserve max_counts begin
            ccall(fptr, Cvoid,
                (Ptr{Nothing}, UInt32,     # device, buildType
                 Ptr{Nothing},              # pBuildInfo
                 Ptr{UInt32},               # pMaxPrimitiveCounts
                 Ptr{Nothing}),             # pSizeInfo
                dev.vks,
                UInt32(1),                  # VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR
                pointer(bgi_buf),
                max_counts,
                pointer(sizes_buf))
        end
    end

    # Read back sizes
    as_size = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 16))
    update_scratch = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 24))
    build_scratch = unsafe_load(Ptr{UInt64}(pointer(sizes_buf) + 32))

    return (acceleration_structure_size=as_size,
            update_scratch_size=update_scratch,
            build_scratch_size=build_scratch)
end

# ── Batched AS Build API ──

# When true, _build_as_on_gpu records into an already-open command buffer
# instead of managing begin/end/submit/wait itself.
const _as_batching = Ref(false)

# GC preservation list for batched builds — keeps input/scratch buffers alive
# until the single submit+wait completes.
const _as_batch_preserves = Any[]

"""
    as_build(f)

Batch multiple acceleration structure builds into a single GPU submission.

Without `as_build`, each `build_blas`/`build_tlas` call submits its own command
buffer and waits for completion — N builds = N submit+wait roundtrips. With
`as_build`, all builds share one command buffer with inter-build barriers,
submitted once at the end.

# Example
```julia
blases, tlas = as_build() do
    bs = [build_blas(verts, idxs) for (verts, idxs) in meshes]
    tlas = build_tlas(bs)
    return (bs, tlas)
end
```

BLAS device addresses are available immediately after `build_blas` returns
(even before the GPU build executes), so `build_tlas` can reference them.
"""
function as_build(f)
    _as_batching[] && error("as_build() cannot be nested")

    ctx = vk_context()

    # Flush any pending compute dispatches before AS builds
    if has_active_recording(ctx)
        vk_flush!()
    end

    cmd = ctx.as_cmd_buf
    unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    )))

    _as_batching[] = true
    empty!(_as_batch_preserves)
    result = try
        f()
    catch
        _as_batching[] = false
        empty!(_as_batch_preserves)
        # Reset command buffer so it's reusable
        unwrap(Vulkan.end_command_buffer(cmd))
        rethrow()
    end
    _as_batching[] = false

    # Final barrier: make all AS writes visible to RT shader reads
    post_barrier = Vulkan.MemoryBarrier(
        C_NULL,
        Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        Vulkan.ACCESS_SHADER_READ_BIT,
    )
    Vulkan.cmd_pipeline_barrier(
        cmd, [post_barrier], [], [];
        src_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        dst_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                       Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    )

    unwrap(Vulkan.end_command_buffer(cmd))

    submit_info = Vulkan.SubmitInfo([], [], [cmd], [])
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=ctx.as_fence))
    unwrap(Vulkan.wait_for_fences(ctx.device, [ctx.as_fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(ctx.device, [ctx.as_fence]))

    empty!(_as_batch_preserves)
    return result
end

"""Build an acceleration structure on the GPU using correctly-packed C structs.

Records vkCmdBuildAccelerationStructuresKHR into a command buffer. When called
inside `as_build()`, records into the shared batch command buffer with only an
inter-build barrier. Otherwise, manages its own command buffer lifecycle
(begin → barrier → build → barrier → end → submit → wait).
"""
function _build_as_on_gpu(ctx::VkContext, accel::Vulkan.AccelerationStructureKHR,
                          scratch_addr::UInt64;
                          as_type::UInt32, build_flags::UInt32=UInt32(0),
                          primitive_count::UInt32=UInt32(0),
                          kwargs...)
    dev = ctx.device
    batching = _as_batching[]

    if !batching
        # Flush any pending compute dispatches before AS build
        if has_active_recording(ctx)
            vk_flush!()
        end
    end

    # Use dedicated AS command buffer + fence (never touches dispatch batches)
    cmd = ctx.as_cmd_buf

    # Pack geometry (96 bytes, correct C layout)
    geo_buf = zeros(UInt8, _C_SIZEOF_AS_GEOMETRY_KHR)
    _pack_geometry!(geo_buf, 0; kwargs...)

    # Pack build geometry info (80 bytes)
    bgi_buf = zeros(UInt8, 80)

    # Pack range info (16 bytes)
    c_range = _VkBRI(primitive_count, UInt32(0), UInt32(0), UInt32(0))

    fptr = Vulkan.function_pointer(dev, "vkCmdBuildAccelerationStructuresKHR")

    if !batching
        unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
            flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )))
    end

    # Synchronize prior AS builds before this build command.
    # Required when TLAS reads BLAS built in earlier submissions,
    # or between batched builds sharing one command buffer.
    pre_barrier = Vulkan.MemoryBarrier(
        C_NULL,
        Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    )
    Vulkan.cmd_pipeline_barrier(
        cmd, [pre_barrier], [], [];
        src_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        dst_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    )

    GC.@preserve geo_buf bgi_buf c_range begin
        geo_ptr = pointer(geo_buf)
        dst_ptr = accel.vks

        _pack_build_geometry_info!(bgi_buf, 0;
            as_type, build_flags, dst_as=dst_ptr,
            geometry_count=UInt32(1),
            p_geometries=Ptr{Nothing}(geo_ptr),
            scratch_addr)

        c_ranges = [c_range]
        pp_ranges = Ref(pointer(c_ranges))

        GC.@preserve c_ranges pp_ranges begin
            ccall(fptr, Cvoid,
                (Ptr{Nothing}, UInt32,
                 Ptr{Nothing},
                 Ptr{Ptr{_VkBRI}}),
                cmd.vks, UInt32(1),
                pointer(bgi_buf),
                pp_ranges)
        end
    end

    if batching
        # Keep buffers alive until as_build() submits
        push!(_as_batch_preserves, (geo_buf, bgi_buf, c_range))
    else
        # Non-batched: finish and submit immediately
        post_barrier = Vulkan.MemoryBarrier(
            C_NULL,
            Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            Vulkan.ACCESS_SHADER_READ_BIT,
        )
        Vulkan.cmd_pipeline_barrier(
            cmd, [post_barrier], [], [];
            src_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            dst_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                           Vulkan.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        )

        unwrap(Vulkan.end_command_buffer(cmd))

        submit_info = Vulkan.SubmitInfo([], [], [cmd], [])
        unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=ctx.as_fence))
        unwrap(Vulkan.wait_for_fences(dev, [ctx.as_fence], true, typemax(UInt64)))
        unwrap(Vulkan.reset_fences(dev, [ctx.as_fence]))
    end
end

"""
    build_blas_pooled(all_vertices, all_indices) -> Vector{LavaBLAS}

Build multiple BLASes using pooled memory allocation. All vertex/index data
goes into a single HOST_VISIBLE buffer, all AS storage into a single
DEVICE_LOCAL buffer, and one scratch buffer is reused. This reduces thousands
of Vulkan allocations to ~6 regardless of mesh count.

`all_vertices[i]::Vector{NTuple{3,Float32}}` and `all_indices[i]::Vector{UInt32}`
provide per-BLAS geometry. Returns one `LavaBLAS` per input.
"""
function build_blas_pooled(all_vertices::Vector{Vector{NTuple{3,Float32}}},
                           all_indices::Vector{Vector{UInt32}})
    n_blas = length(all_vertices)
    n_blas == 0 && return LavaBLAS[]
    @assert length(all_indices) == n_blas

    ctx = vk_context()
    dev = ctx.device

    vfmt = UInt32(Vulkan.FORMAT_R32G32B32_SFLOAT)
    vstride = UInt64(sizeof(NTuple{3,Float32}))
    itype = UInt32(Vulkan.INDEX_TYPE_UINT32)
    build_flags = UInt32(Vulkan.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    geo_flags = UInt32(Vulkan.GEOMETRY_OPAQUE_BIT_KHR)

    # Pass 1: Query build sizes and compute pool layout
    vertex_offsets = Vector{UInt64}(undef, n_blas)
    index_offsets = Vector{UInt64}(undef, n_blas)
    as_offsets = Vector{UInt64}(undef, n_blas)
    as_sizes_list = Vector{UInt64}(undef, n_blas)
    max_scratch_size = UInt64(0)
    input_cursor = UInt64(0)
    as_cursor = UInt64(0)

    for i in 1:n_blas
        n_tris = UInt32(length(all_indices[i]) ÷ 3)
        max_vertex = UInt32(length(all_vertices[i]) - 1)

        sizes = _query_as_build_sizes(dev;
            as_type=UInt32(1), build_flags,
            geometry_type=:triangles,
            vertex_format=vfmt, vertex_addr=UInt64(0), vertex_stride=vstride,
            max_vertex, index_type=itype, index_addr=UInt64(0),
            geo_flags, max_primitive_count=n_tris)

        vertex_offsets[i] = input_cursor
        vbytes = UInt64(length(all_vertices[i]) * sizeof(NTuple{3,Float32}))
        input_cursor += (vbytes + 15) & ~UInt64(15)

        index_offsets[i] = input_cursor
        ibytes = UInt64(length(all_indices[i]) * sizeof(UInt32))
        input_cursor += (ibytes + 15) & ~UInt64(15)

        as_offsets[i] = as_cursor
        as_sizes_list[i] = sizes.acceleration_structure_size
        as_cursor += (sizes.acceleration_structure_size + 255) & ~UInt64(255)

        max_scratch_size = max(max_scratch_size, sizes.build_scratch_size)
    end

    total_input_bytes = input_cursor
    total_as_bytes = as_cursor

    # Pass 2: Allocate pooled buffers (3 allocations total)
    input_buf, input_mem, input_base_addr = _create_as_input_pool(max(total_input_bytes, 16))
    as_pool_buf, as_pool_mem = _create_as_storage_buffer(max(total_as_bytes, 16))
    scratch_buf, scratch_mem, scratch_addr = _create_scratch_buffer(max(max_scratch_size, 16))

    # Pass 3: Upload all vertex/index data into the input pool
    mapped_ptr = unwrap(Vulkan.map_memory(dev, input_mem, 0, total_input_bytes))
    for i in 1:n_blas
        vdata = all_vertices[i]
        vbytes = length(vdata) * sizeof(NTuple{3,Float32})
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + vertex_offsets[i]),
                       Ptr{UInt8}(pointer(vdata)), vbytes)
        idata = all_indices[i]
        ibytes = length(idata) * sizeof(UInt32)
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + index_offsets[i]),
                       Ptr{UInt8}(pointer(idata)), ibytes)
    end
    Vulkan.unmap_memory(dev, input_mem)

    # Pass 4: Create AS objects and build all BLASes
    hw_blas_list = Vector{LavaBLAS}(undef, n_blas)
    for i in 1:n_blas
        as_ci = Vulkan.AccelerationStructureCreateInfoKHR(
            as_pool_buf, as_offsets[i], as_sizes_list[i],
            Vulkan.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        )
        accel = Vulkan.AccelerationStructureKHR(dev, as_ci)
        addr_info = Vulkan.AccelerationStructureDeviceAddressInfoKHR(accel)
        as_addr = Vulkan.get_acceleration_structure_device_address_khr(dev, addr_info)
        hw_blas_list[i] = LavaBLAS(accel, as_pool_buf, as_pool_mem, as_addr)
    end

    GC.@preserve input_buf input_mem scratch_buf scratch_mem as_pool_buf as_pool_mem begin
        as_build() do
            cmd = ctx.as_cmd_buf
            for i in 1:n_blas
                n_tris = UInt32(length(all_indices[i]) ÷ 3)
                max_vertex = UInt32(length(all_vertices[i]) - 1)
                _build_as_on_gpu(ctx, hw_blas_list[i].accel, scratch_addr;
                    as_type=UInt32(1), build_flags,
                    geometry_type=:triangles,
                    vertex_format=vfmt,
                    vertex_addr=input_base_addr + vertex_offsets[i],
                    vertex_stride=vstride,
                    max_vertex, index_type=itype,
                    index_addr=input_base_addr + index_offsets[i],
                    geo_flags, primitive_count=n_tris)
                # Barrier between builds: shared scratch buffer must be drained before reuse.
                # Without this, concurrent AS builds corrupt each other's scratch data.
                if i < n_blas
                    scratch_barrier = Vulkan.MemoryBarrier(
                        C_NULL,
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                    )
                    Vulkan.cmd_pipeline_barrier(
                        cmd, [scratch_barrier], [], [];
                        src_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        dst_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    )
                end
            end
        end
    end

    # Eagerly free temporary buffers to reclaim VRAM immediately.
    # The AS build is complete (as_build waits for GPU), so these are no longer needed.
    # Without explicit cleanup, they linger until GC runs, wasting hundreds of MB on
    # large scenes (e.g., crown scene: ~170MB input + ~50MB scratch).
    finalize(scratch_buf); finalize(scratch_mem)
    finalize(input_buf); finalize(input_mem)

    return hw_blas_list
end

# ── Raycore-compatible bridge functions ──

"""
    build_blas_from_primitives(primitives) -> LavaBLAS

Build a hardware BLAS from an array of triangle primitives.
Each primitive must have a `.vertices` field with 3 vertex positions
(e.g., `SVector{3, Point3f}`). Works with Raycore's `Triangle{T}` type.

Primitives on GPU (LavaArray, etc.) are downloaded to CPU automatically.
"""
function build_blas_from_primitives(primitives; opaque::Bool=true)
    cpu_prims = _to_cpu_vector(primitives)
    n_tris = length(cpu_prims)

    # Extract vertex positions: 3 vertices per triangle
    vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        verts = cpu_prims[i].vertices
        for j in 1:3
            v = verts[j]
            vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end

    # Sequential indices (each triangle has 3 unique vertices)
    indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        indices[i+1] = UInt32(i)
    end

    return build_blas(vertices, indices; opaque)
end

"""
    build_hw_accel_from_tlas(tlas) -> (hw_tlas, triangle_data, blas_offsets)

Build hardware acceleration structures from a Raycore-compatible TLAS.

Uses pooled memory allocation: all vertex/index data goes into a single
HOST_VISIBLE buffer, all BLAS AS storage into a single DEVICE_LOCAL buffer,
and one scratch buffer is reused across all builds. This reduces ~3000+
individual Vulkan allocations to ~6 regardless of mesh count.

The TLAS must have:
- `.blas_array`: indexable collection of BLAS objects, each with `.primitives`
- `.instances`: array of instance descriptors with `.blas_index` (1-based),
  `.transform` (4×4 matrix, local-to-world)

Returns:
- `hw_tlas::LavaTLAS` — Vulkan top-level acceleration structure
- `triangle_data::Vector` — CPU vector of all primitives (caller uploads to GPU)
- `blas_offsets::Vector{UInt32}` — per-BLAS offset into triangle_data (0-based)

After a ray trace, look up the hit triangle:
```
tri = triangle_data[blas_offsets[instance_custom_index + 1] + primitive_id + 1]
```
where `instance_custom_index` = BLAS index (0-based) set by this function.
"""
function build_hw_accel_from_tlas(tlas)
    instances = _to_cpu_vector(tlas.instances)
    blas_array = _to_cpu_vector(tlas.blas_array)

    n_blas = length(blas_array)
    n_instances = length(instances)
    ctx = vk_context()
    dev = ctx.device

    # ── Pass 1: Collect all geometry on CPU ──
    cpu_prims_list = Vector{Any}(undef, n_blas)
    blas_offsets = Vector{UInt32}(undef, n_blas)
    all_primitives = []
    prim_offset = UInt32(0)

    # Per-BLAS vertex/index arrays
    all_vertices = Vector{Vector{NTuple{3,Float32}}}(undef, n_blas)
    all_indices = Vector{Vector{UInt32}}(undef, n_blas)

    for i in 1:n_blas
        cpu_prims_list[i] = _to_cpu_vector(blas_array[i].primitives)
        blas_offsets[i] = prim_offset
        prim_offset += UInt32(length(cpu_prims_list[i]))
        append!(all_primitives, cpu_prims_list[i])

        # Extract vertices/indices for this BLAS
        n_tris = length(cpu_prims_list[i])
        verts = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
        idxs = Vector{UInt32}(undef, n_tris * 3)
        for j in 1:n_tris
            vs = cpu_prims_list[i][j].vertices
            for k in 1:3
                v = vs[k]
                verts[(j-1)*3 + k] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
                idxs[(j-1)*3 + k] = UInt32((j-1)*3 + k - 1)
            end
        end
        all_vertices[i] = verts
        all_indices[i] = idxs
    end

    # ── Pass 2: Query build sizes and compute pool layout ──
    vfmt = UInt32(Vulkan.FORMAT_R32G32B32_SFLOAT)
    vstride = UInt64(sizeof(NTuple{3,Float32}))
    itype = UInt32(Vulkan.INDEX_TYPE_UINT32)
    build_flags = UInt32(Vulkan.BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
    geo_flags = UInt32(Vulkan.GEOMETRY_OPAQUE_BIT_KHR)

    # Layout tracking
    vertex_offsets = Vector{UInt64}(undef, n_blas)  # byte offset into input pool
    index_offsets = Vector{UInt64}(undef, n_blas)
    as_offsets = Vector{UInt64}(undef, n_blas)       # byte offset into AS storage pool
    as_sizes_list = Vector{UInt64}(undef, n_blas)
    max_scratch_size = UInt64(0)

    input_cursor = UInt64(0)  # cursor into input pool (vertex then index, interleaved per BLAS)
    as_cursor = UInt64(0)     # cursor into AS storage pool (256-byte aligned)

    for i in 1:n_blas
        n_tris = UInt32(length(all_indices[i]) ÷ 3)
        max_vertex = UInt32(length(all_vertices[i]) - 1)

        # Query sizes — addresses don't matter for size queries
        sizes = _query_as_build_sizes(dev;
            as_type=UInt32(1), build_flags,
            geometry_type=:triangles,
            vertex_format=vfmt, vertex_addr=UInt64(0), vertex_stride=vstride,
            max_vertex, index_type=itype, index_addr=UInt64(0),
            geo_flags, max_primitive_count=n_tris)

        # Input pool layout: vertex data then index data, 16-byte aligned
        vertex_offsets[i] = input_cursor
        vbytes = UInt64(length(all_vertices[i]) * sizeof(NTuple{3,Float32}))
        input_cursor += (vbytes + 15) & ~UInt64(15)  # 16-byte align

        index_offsets[i] = input_cursor
        ibytes = UInt64(length(all_indices[i]) * sizeof(UInt32))
        input_cursor += (ibytes + 15) & ~UInt64(15)  # 16-byte align

        # AS storage pool layout: 256-byte aligned
        as_offsets[i] = as_cursor
        as_sizes_list[i] = sizes.acceleration_structure_size
        as_cursor += (sizes.acceleration_structure_size + 255) & ~UInt64(255)

        max_scratch_size = max(max_scratch_size, sizes.build_scratch_size)
    end

    total_input_bytes = input_cursor
    total_as_bytes = as_cursor

    # ── Pass 3: Allocate pooled buffers ──
    # Input pool: HOST_VISIBLE for direct upload
    input_buf, input_mem, input_base_addr = _create_as_input_pool(max(total_input_bytes, 16))

    # AS storage pool: DEVICE_LOCAL
    as_pool_buf, as_pool_mem = _create_as_storage_buffer(max(total_as_bytes, 16))

    # Scratch buffer: single allocation, reused for all builds
    scratch_buf, scratch_mem, scratch_addr = _create_scratch_buffer(max(max_scratch_size, 16))

    # ── Pass 4: Upload vertex/index data into input pool ──
    mapped_ptr = unwrap(Vulkan.map_memory(dev, input_mem, 0, total_input_bytes))
    for i in 1:n_blas
        # Copy vertex data
        vdata = all_vertices[i]
        vbytes = length(vdata) * sizeof(NTuple{3,Float32})
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + vertex_offsets[i]),
                       Ptr{UInt8}(pointer(vdata)), vbytes)
        # Copy index data
        idata = all_indices[i]
        ibytes = length(idata) * sizeof(UInt32)
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr + index_offsets[i]),
                       Ptr{UInt8}(pointer(idata)), ibytes)
    end
    Vulkan.unmap_memory(dev, input_mem)

    # ── Pass 5: Create AS objects and build all BLASes ──
    hw_blas_list = Vector{LavaBLAS}(undef, n_blas)

    # Create AccelerationStructureKHR objects (one per BLAS, all referencing the pool)
    for i in 1:n_blas
        as_ci = Vulkan.AccelerationStructureCreateInfoKHR(
            as_pool_buf, as_offsets[i], as_sizes_list[i],
            Vulkan.ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        )
        accel = Vulkan.AccelerationStructureKHR(dev, as_ci)
        addr_info = Vulkan.AccelerationStructureDeviceAddressInfoKHR(accel)
        as_addr = Vulkan.get_acceleration_structure_device_address_khr(dev, addr_info)
        hw_blas_list[i] = LavaBLAS(accel, as_pool_buf, as_pool_mem, as_addr)
    end

    # Batch all BLAS + TLAS builds into a single GPU submission
    hw_tlas = GC.@preserve input_buf input_mem scratch_buf scratch_mem as_pool_buf as_pool_mem begin
        as_build() do
            cmd = ctx.as_cmd_buf
            for i in 1:n_blas
                n_tris = UInt32(length(all_indices[i]) ÷ 3)
                max_vertex = UInt32(length(all_vertices[i]) - 1)

                _build_as_on_gpu(ctx, hw_blas_list[i].accel, scratch_addr;
                    as_type=UInt32(1), build_flags,
                    geometry_type=:triangles,
                    vertex_format=vfmt,
                    vertex_addr=input_base_addr + vertex_offsets[i],
                    vertex_stride=vstride,
                    max_vertex, index_type=itype,
                    index_addr=input_base_addr + index_offsets[i],
                    geo_flags, primitive_count=n_tris)
                # Barrier between builds: shared scratch buffer must be drained before reuse.
                if i < n_blas
                    scratch_barrier = Vulkan.MemoryBarrier(
                        C_NULL,
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                        Vulkan.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                    )
                    Vulkan.cmd_pipeline_barrier(
                        cmd, [scratch_barrier], [], [];
                        src_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        dst_stage_mask=Vulkan.PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    )
                end
            end

            # Build instance list for TLAS
            hw_blas_refs = Vector{LavaBLAS}(undef, n_instances)
            transforms = Vector{NTuple{12,Float32}}(undef, n_instances)
            custom_indices = Vector{UInt32}(undef, n_instances)

            for i in 1:n_instances
                inst = instances[i]
                blas_idx = Int(inst.blas_index)
                hw_blas_refs[i] = hw_blas_list[blas_idx]
                transforms[i] = _mat4_to_vk_transform(inst.transform)
                custom_indices[i] = UInt32(blas_idx - 1)
            end

            return build_tlas(hw_blas_refs; transforms, custom_indices)
        end
    end

    # Eagerly free temporary buffers to reclaim VRAM immediately.
    # GPU build is complete (as_build waits for fences). AS storage pool
    # stays alive via LavaBLAS.buffer references.
    finalize(scratch_buf); finalize(scratch_mem)
    finalize(input_buf); finalize(input_mem)

    # Convert all_primitives to typed vector
    if !isempty(all_primitives)
        T = typeof(all_primitives[1])
        typed_prims = T[p for p in all_primitives]
    else
        typed_prims = Any[]
    end

    return (hw_tlas, typed_prims, blas_offsets)
end

"""Convert a 4×4 matrix to VkTransformMatrixKHR (3×4 row-major) NTuple{12,Float32}.
Works with SMatrix{4,4}, Mat4f, or any indexable 4×4 matrix."""
function _mat4_to_vk_transform(m)
    # VkTransformMatrixKHR = 3 rows × 4 cols, row-major
    # Row 0: m[1,1], m[1,2], m[1,3], m[1,4]
    # Row 1: m[2,1], m[2,2], m[2,3], m[2,4]
    # Row 2: m[3,1], m[3,2], m[3,3], m[3,4]
    return (Float32(m[1,1]), Float32(m[1,2]), Float32(m[1,3]), Float32(m[1,4]),
            Float32(m[2,1]), Float32(m[2,2]), Float32(m[2,3]), Float32(m[2,4]),
            Float32(m[3,1]), Float32(m[3,2]), Float32(m[3,3]), Float32(m[3,4]))
end

"""Download GPU array to CPU Vector, or return as-is if already a CPU collection."""
function _to_cpu_vector(x)
    x isa Vector && return x
    x isa Tuple && return collect(x)
    # Try Array() for GPU arrays (LavaArray, CuArray, ROCArray, etc.)
    try
        return Array(x)
    catch
        return collect(x)
    end
end
