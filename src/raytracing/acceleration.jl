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

    # Build on GPU
    _build_as_on_gpu(ctx, accel, scratch_addr;
        as_type=UInt32(1), build_flags,
        geometry_type=:triangles,
        vertex_format=vfmt, vertex_addr, vertex_stride=vstride,
        max_vertex, index_type=itype, index_addr,
        geo_flags, primitive_count=n_triangles)

    # Get AS device address
    addr_info = Vulkan.AccelerationStructureDeviceAddressInfoKHR(accel)
    as_addr = Vulkan.get_acceleration_structure_device_address_khr(dev, addr_info)

    # Temporary buffers (vertex, index, scratch) are GC-managed via Vulkan.jl finalizers.
    # No manual destroy needed — they'll be freed when GC collects them.

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

    # Build on GPU
    _build_as_on_gpu(ctx, accel, scratch_addr;
        as_type=UInt32(0), build_flags,
        geometry_type=:instances,
        instance_addr=inst_addr,
        primitive_count=UInt32(n_instances))

    # Temporary buffers (instance, scratch) are GC-managed via Vulkan.jl finalizers.
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

"""Build an acceleration structure on the GPU using correctly-packed C structs.

Records vkCmdBuildAccelerationStructuresKHR into a one-shot command buffer and
submits it, waiting for completion. All C structs are manually packed to work
around VulkanCore.jl alignment bugs.
"""
function _build_as_on_gpu(ctx::VkContext, accel::Vulkan.AccelerationStructureKHR,
                          scratch_addr::UInt64;
                          as_type::UInt32, build_flags::UInt32=UInt32(0),
                          primitive_count::UInt32=UInt32(0),
                          kwargs...)
    dev = ctx.device

    # Flush any pending compute dispatches before AS build
    if has_active_recording(ctx)
        vk_flush!()
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

    unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    )))

    # Synchronize prior AS builds before this build command.
    # Required when TLAS reads BLAS built in earlier submissions.
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

    # Make AS writes visible to subsequent AS builds and RT shader reads.
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

    # Build hardware BLAS for each unique BLAS + collect primitives
    hw_blas_list = Vector{LavaBLAS}(undef, n_blas)
    blas_offsets = Vector{UInt32}(undef, n_blas)
    all_primitives = []
    offset = UInt32(0)

    for i in 1:n_blas
        blas = blas_array[i]
        cpu_prims = _to_cpu_vector(blas.primitives)

        hw_blas_list[i] = build_blas_from_primitives(cpu_prims)
        blas_offsets[i] = offset

        append!(all_primitives, cpu_prims)
        offset += UInt32(length(cpu_prims))
    end

    # Build instance list for TLAS
    # Each Raycore instance → one Vulkan instance
    hw_blas_refs = Vector{LavaBLAS}(undef, n_instances)
    transforms = Vector{NTuple{12,Float32}}(undef, n_instances)
    custom_indices = Vector{UInt32}(undef, n_instances)

    for i in 1:n_instances
        inst = instances[i]
        blas_idx = Int(inst.blas_index)
        hw_blas_refs[i] = hw_blas_list[blas_idx]

        # Extract 3×4 row-major transform from 4×4 column-major matrix
        m = inst.transform
        transforms[i] = _mat4_to_vk_transform(m)

        # Store BLAS index (0-based) as instance custom index
        custom_indices[i] = UInt32(blas_idx - 1)
    end

    # Build hardware TLAS
    hw_tlas = build_tlas(hw_blas_refs; transforms, custom_indices)

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
