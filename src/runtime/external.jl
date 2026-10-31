# External-memory interop: share Lava-produced images with other APIs.
#
# An `ExternalImage` owns a dedicated, exportable Vulkan image allocation on
# Lava's device. Its memory can be handed to another API as an opaque fd
# (`memoryfd`) — e.g. imported into OpenGL via GL_EXT_memory_object_fd — so
# a frame computed in a Lava kernel reaches the other API with **zero
# copies across PCIe** (one device-local blit, no host roundtrip).
#
# Everything here is additive and opt-in: nothing runs unless an
# ExternalImage is created. The allocation is deliberately dedicated and
# outside the pooled allocator — external memory must not be sub-allocated
# (importers see the whole allocation), and NVIDIA requires dedicated
# allocations for exported images anyway.
#
# See VideoEdit/interop_mwe.jl for a full Vulkan↔GL validation including
# the GL import side.

"""
    ExternalImage(width, height; format = Vulkan.FORMAT_R8G8B8A8_UNORM)

A GPU image whose memory can be exported to other APIs (see [`memoryfd`](@ref)).
OPTIMAL tiling, `TRANSFER_DST | SAMPLED` usage. Fill it from a `LavaArray`
with `copyto!(img, array)`. Requires `VK_KHR_external_memory_fd` (enabled
automatically at device creation when the driver offers it; check
`vk_context().external_memory_available`).
"""
mutable struct ExternalImage
    image::Vulkan.Image
    memory::Vulkan.DeviceMemory
    width::Int
    height::Int
    allocation_size::Int   # importers must import exactly this many bytes
    layout_initialized::Bool
end

function ExternalImage(width::Integer, height::Integer;
                       format::Vulkan.Format = Vulkan.FORMAT_R8G8B8A8_UNORM)
    ctx = vk_context()
    ctx.external_memory_available ||
        error("this device was created without VK_KHR_external_memory_fd support")
    handle_types = Vulkan.EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    image = @vk_checked "external_image_create" Vulkan.create_image(ctx.device, Vulkan.IMAGE_TYPE_2D, format,
        Vulkan.Extent3D(width, height, 1), 1, 1, Vulkan.SAMPLE_COUNT_1_BIT,
        Vulkan.IMAGE_TILING_OPTIMAL,
        Vulkan.IMAGE_USAGE_TRANSFER_DST_BIT | Vulkan.IMAGE_USAGE_SAMPLED_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[], Vulkan.IMAGE_LAYOUT_UNDEFINED;
        next = Vulkan.ExternalMemoryImageCreateInfo(; handle_types))
    req = Vulkan.get_image_memory_requirements(ctx.device, image)
    mtype = findfirst(0:(length(ctx.memory_properties.memory_types) - 1)) do i
        (req.memory_type_bits >> i) & 1 == 1 &&
            ctx.memory_properties.memory_types[i + 1].property_flags &
            Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT != Vulkan.MemoryPropertyFlag(0)
    end
    mtype === nothing && error("no device-local memory type for external image")
    memory = @vk_checked "external_image_alloc" Vulkan.allocate_memory(ctx.device, req.size, mtype - 1;
        next = Vulkan.ExportMemoryAllocateInfo(;
            next = Vulkan.MemoryDedicatedAllocateInfo(; image),
            handle_types))
    @vk_checked "external_image_bind" Vulkan.bind_image_memory(ctx.device, image, memory, 0)
    return ExternalImage(image, memory, Int(width), Int(height), Int(req.size), false)
end

"""
    memoryfd(img::ExternalImage) -> Int

Export the image's memory as an opaque file descriptor
(`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`). Each call duplicates a
new fd; the importer takes ownership (e.g. `glImportMemoryFdEXT` consumes
it). Importers must import exactly `img.allocation_size` bytes and, for
GL, mark the memory object dedicated and the texture OPTIMAL-tiled.
"""
function memoryfd(img::ExternalImage)
    ctx = vk_context()
    # Vulkan.jl's high-level get_memory_fd_khr passes a Ref{Int64} where the
    # C signature wants int*, so call the low-level entry point directly.
    fptr = Vulkan.function_pointer(ctx.device, "vkGetMemoryFdKHR")
    pfd = Ref{Int32}(-1)
    info = Vulkan._MemoryGetFdInfoKHR(img.memory,
        Vulkan.EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
    res = Vulkan.vkGetMemoryFdKHR(ctx.device, info, pfd, fptr)
    res == Vulkan.VkCore.VK_SUCCESS || error("vkGetMemoryFdKHR failed: $res")
    return Int(pfd[])
end

"""
    copyto!(img::ExternalImage, a::LavaArray) -> img

Blit the array's bytes into the external image (device-local copy — the
data never leaves the GPU). The array must hold at least
`4 * width * height` bytes of pixel data in x-contiguous row order matching
the image format. Waits for pending Lava kernels writing `a`, then blocks
until the copy has landed, so the importer may sample immediately.

Runs as a one-shot submission on the context's secondary compute queue —
it never interleaves with the BatchQueue's batched submissions.
"""
function Base.copyto!(img::ExternalImage, a::LavaArray)
    nbytes = length(a) * sizeof(eltype(a))
    needed = 4 * img.width * img.height
    nbytes >= needed ||
        error("array holds $nbytes bytes; image needs $needed")
    ctx = (a.buf[].ctx)::VkContext
    KA.synchronize(LavaBackend(ctx))  # pending kernel writes must land first

    managed = a.buf[]
    src_offset = UInt64(managed.pool_offset + a.offset)
    # Vulkan.jl handles are refcounted and destroy themselves via finalizers —
    # no manual destroy (a second vkDestroy* on the same handle segfaults).
    pool = @vk_checked "external_copy_pool" Vulkan.create_command_pool(ctx.device, ctx.queue_family_index)
    let
        cb = first(@vk_checked "external_copy_cb" Vulkan.allocate_command_buffers(ctx.device,
            Vulkan.CommandBufferAllocateInfo(pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))
        @vk_checked "external_copy_begin" Vulkan.begin_command_buffer(cb, Vulkan.CommandBufferBeginInfo())
        range = Vulkan.ImageSubresourceRange(Vulkan.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        oldlayout = img.layout_initialized ? Vulkan.IMAGE_LAYOUT_GENERAL :
                                             Vulkan.IMAGE_LAYOUT_UNDEFINED
        Vulkan.cmd_pipeline_barrier(cb, [], [],
            [Vulkan.ImageMemoryBarrier(Vulkan.AccessFlag(0), Vulkan.ACCESS_TRANSFER_WRITE_BIT,
                oldlayout, Vulkan.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                Vulkan.QUEUE_FAMILY_IGNORED, Vulkan.QUEUE_FAMILY_IGNORED, img.image, range)];
            src_stage_mask = Vulkan.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            dst_stage_mask = Vulkan.PIPELINE_STAGE_TRANSFER_BIT)
        Vulkan.cmd_copy_buffer_to_image(cb, managed.buffer, img.image,
            Vulkan.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            [Vulkan.BufferImageCopy(src_offset, 0, 0,
                Vulkan.ImageSubresourceLayers(Vulkan.IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
                Vulkan.Offset3D(0, 0, 0), Vulkan.Extent3D(img.width, img.height, 1))])
        Vulkan.cmd_pipeline_barrier(cb, [], [],
            [Vulkan.ImageMemoryBarrier(Vulkan.ACCESS_TRANSFER_WRITE_BIT, Vulkan.AccessFlag(0),
                Vulkan.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, Vulkan.IMAGE_LAYOUT_GENERAL,
                Vulkan.QUEUE_FAMILY_IGNORED, Vulkan.QUEUE_FAMILY_IGNORED, img.image, range)];
            src_stage_mask = Vulkan.PIPELINE_STAGE_TRANSFER_BIT,
            dst_stage_mask = Vulkan.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
        @vk_checked "external_copy_end" Vulkan.end_command_buffer(cb)
        @vk_checked "external_copy_submit" Vulkan.queue_submit(ctx.compute_queue, [Vulkan.SubmitInfo([], [], [cb], [])])
        @vk_checked "external_copy_wait" Vulkan.queue_wait_idle(ctx.compute_queue)
        img.layout_initialized = true
    end
    return img
end
