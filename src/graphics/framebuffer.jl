# Offscreen framebuffer and render targets for Lava.jl
#
# Uses VK_KHR_dynamic_rendering — no VkRenderPass/VkFramebuffer needed.
# Just images + views as render targets.

"""
    LavaFramebuffer

Offscreen render target with color and optional depth images.
Used for render-to-texture or offscreen rendering.
"""
mutable struct LavaFramebuffer
    width::Int
    height::Int
    # Color attachment
    color_image::Vulkan.Image
    color_memory::Vulkan.DeviceMemory
    color_view::Vulkan.ImageView
    color_format::Vulkan.Format
    # Depth attachment (optional)
    depth_image::Union{Nothing, Vulkan.Image}
    depth_memory::Union{Nothing, Vulkan.DeviceMemory}
    depth_view::Union{Nothing, Vulkan.ImageView}
    depth_format::Vulkan.Format
end

"""
    LavaFramebuffer(width, height; depth=true, color_format=Vulkan.FORMAT_B8G8R8A8_SRGB)

Create an offscreen framebuffer with color and optional depth attachments.
"""
function LavaFramebuffer(width::Integer, height::Integer;
                          ctx::VkContext=vk_context(),
                          depth::Bool=true,
                          color_format::Vulkan.Format=Vulkan.FORMAT_B8G8R8A8_SRGB)
    dev = ctx.device
    phys = ctx.physical_device

    # Create color image
    color_image = Vulkan.Image(dev,
        Vulkan.IMAGE_TYPE_2D, color_format,
        Vulkan.Extent3D(UInt32(width), UInt32(height), UInt32(1)),
        UInt32(1), UInt32(1),
        Vulkan.SAMPLE_COUNT_1_BIT,
        Vulkan.IMAGE_TILING_OPTIMAL,
        Vulkan.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Vulkan.IMAGE_USAGE_TRANSFER_SRC_BIT | Vulkan.IMAGE_USAGE_SAMPLED_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[],
        Vulkan.IMAGE_LAYOUT_UNDEFINED,
    )
    color_memory = alloc_image_memory(dev, phys, color_image)
    color_view = Vulkan.ImageView(dev, color_image, Vulkan.IMAGE_VIEW_TYPE_2D, color_format,
        Vulkan.ComponentMapping(
            Vulkan.COMPONENT_SWIZZLE_IDENTITY, Vulkan.COMPONENT_SWIZZLE_IDENTITY,
            Vulkan.COMPONENT_SWIZZLE_IDENTITY, Vulkan.COMPONENT_SWIZZLE_IDENTITY,
        ),
        Vulkan.ImageSubresourceRange(Vulkan.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
    )

    # Create depth image (optional)
    depth_format = Vulkan.FORMAT_D32_SFLOAT
    depth_img = nothing
    depth_mem = nothing
    depth_vw = nothing
    if depth
        depth_img = Vulkan.Image(dev,
            Vulkan.IMAGE_TYPE_2D, depth_format,
            Vulkan.Extent3D(UInt32(width), UInt32(height), UInt32(1)),
            UInt32(1), UInt32(1),
            Vulkan.SAMPLE_COUNT_1_BIT,
            Vulkan.IMAGE_TILING_OPTIMAL,
            Vulkan.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[],
            Vulkan.IMAGE_LAYOUT_UNDEFINED,
        )
        depth_mem = alloc_image_memory(dev, phys, depth_img)
        depth_vw = Vulkan.ImageView(dev, depth_img, Vulkan.IMAGE_VIEW_TYPE_2D, depth_format,
            Vulkan.ComponentMapping(
                Vulkan.COMPONENT_SWIZZLE_IDENTITY, Vulkan.COMPONENT_SWIZZLE_IDENTITY,
                Vulkan.COMPONENT_SWIZZLE_IDENTITY, Vulkan.COMPONENT_SWIZZLE_IDENTITY,
            ),
            Vulkan.ImageSubresourceRange(Vulkan.IMAGE_ASPECT_DEPTH_BIT,
                UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
        )
    end

    LavaFramebuffer(Int(width), Int(height),
        color_image, color_memory, color_view, color_format,
        depth_img, depth_mem, depth_vw, depth_format)
end

"""Allocate device-local memory for an image."""
function alloc_image_memory(dev::Vulkan.Device, phys::Vulkan.PhysicalDevice, image::Vulkan.Image)
    mem_reqs = Vulkan.get_image_memory_requirements(dev, image)
    mem_props = Vulkan.get_physical_device_memory_properties(phys)

    type_idx = find_memory_type(mem_props, mem_reqs.memory_type_bits,
                                  Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    mem = Vulkan.DeviceMemory(dev, mem_reqs.size, type_idx)
    unwrap(Vulkan.bind_image_memory(dev, image, mem, UInt64(0)))
    return mem
end

"""Find a memory type index matching required bits and properties."""
function find_memory_type(mem_props, type_bits::UInt32, required_flags)
    for i in 0:Int(mem_props.memory_type_count) - 1
        if (type_bits & (1 << i)) != 0
            flags = mem_props.memory_types[i + 1].property_flags
            if (flags & required_flags) == required_flags
                return UInt32(i)
            end
        end
    end
    error("No suitable memory type found for image allocation")
end

Base.size(fb::LavaFramebuffer) = (fb.width, fb.height)

"""Bytes per pixel for a Vulkan format."""
function format_pixel_size(fmt::Vulkan.Format)
    fmt == Vulkan.FORMAT_B8G8R8A8_SRGB    && return 4
    fmt == Vulkan.FORMAT_B8G8R8A8_UNORM   && return 4
    fmt == Vulkan.FORMAT_R8G8B8A8_SRGB    && return 4
    fmt == Vulkan.FORMAT_R8G8B8A8_UNORM   && return 4
    fmt == Vulkan.FORMAT_R32G32B32A32_SFLOAT && return 16
    fmt == Vulkan.FORMAT_R16G16B16A16_SFLOAT && return 8
    error("Unknown pixel size for format $fmt")
end

"""Julia element type for readback of a Vulkan format."""
function format_element_type(fmt::Vulkan.Format)
    fmt == Vulkan.FORMAT_B8G8R8A8_SRGB      && return NTuple{4, UInt8}
    fmt == Vulkan.FORMAT_B8G8R8A8_UNORM     && return NTuple{4, UInt8}
    fmt == Vulkan.FORMAT_R8G8B8A8_SRGB      && return NTuple{4, UInt8}
    fmt == Vulkan.FORMAT_R8G8B8A8_UNORM     && return NTuple{4, UInt8}
    fmt == Vulkan.FORMAT_R32G32B32A32_SFLOAT && return NTuple{4, Float32}
    fmt == Vulkan.FORMAT_R16G16B16A16_SFLOAT && return NTuple{4, Float16}
    error("Unknown element type for format $fmt")
end

"""
    readback_framebuffer(fb::LavaFramebuffer) -> Matrix

Read back the color attachment pixels to CPU memory.
Returns a width x height matrix with element type matching the framebuffer format:
- `FORMAT_B8G8R8A8_SRGB` / `_UNORM`: `NTuple{4, UInt8}` (BGRA bytes)
- `FORMAT_R32G32B32A32_SFLOAT`: `NTuple{4, Float32}` (RGBA float)
- `FORMAT_R16G16B16A16_SFLOAT`: `NTuple{4, Float16}` (RGBA half)
"""
function readback_framebuffer(fb::LavaFramebuffer; ctx::VkContext=vk_context())
    dev = ctx.device

    if has_active_recording(ctx.default_bq)
        flush!(ctx.default_bq, ctx.device)
    end

    bpp = format_pixel_size(fb.color_format)
    T = format_element_type(fb.color_format)
    nbytes = fb.width * fb.height * bpp
    staging_buf, _, mapped_ptr, _ = get_staging(ctx.default_bq, nbytes)

    cmd = ctx.xfer_cmd_buf
    fence = ctx.xfer_fence
    unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(;
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))

    transition_image!(cmd, fb.color_image,
        Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, Vulkan.PIPELINE_STAGE_TRANSFER_BIT,
        Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, Vulkan.ACCESS_TRANSFER_READ_BIT)

    region = Vulkan.BufferImageCopy(
        UInt64(0), UInt32(0), UInt32(0),
        Vulkan.ImageSubresourceLayers(Vulkan.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        Vulkan.Offset3D(0, 0, 0),
        Vulkan.Extent3D(UInt32(fb.width), UInt32(fb.height), UInt32(1)),
    )
    Vulkan.cmd_copy_image_to_buffer(cmd, fb.color_image,
        Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buf, [region])

    unwrap(Vulkan.end_command_buffer(cmd))

    submit_info = Vulkan.SubmitInfo([], [], [cmd], [])
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=fence))
    unwrap(Vulkan.wait_for_fences(dev, [fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [fence]))
    drain_deferred_frees!(ctx.default_bq)
    drain_deferred_as_frees!(ctx.default_bq)

    pixels = Matrix{T}(undef, fb.width, fb.height)
    unsafe_copyto!(Ptr{UInt8}(pointer(pixels)), Ptr{UInt8}(mapped_ptr), nbytes)
    return pixels
end

"""
    readback_window(win::RenderWindow) -> Matrix{NTuple{4, UInt8}}

Read back the current swapchain image to CPU memory.
Must be called after rendering but BEFORE present_frame!.
Returns a width x height matrix of BGRA byte tuples.
"""
function readback_window(win::RenderWindow; ctx::VkContext=vk_context())
    dev = ctx.device

    if has_active_recording(ctx.default_bq)
        flush!(ctx.default_bq, ctx.device)
    end

    w, h = size(win)
    bpp = format_pixel_size(win.format)
    T = format_element_type(win.format)
    nbytes = w * h * bpp
    staging_buf, _, mapped_ptr, _ = get_staging(ctx.default_bq, nbytes)

    image = win.images[win.current_image_idx + 1]

    cmd = ctx.xfer_cmd_buf
    fence = ctx.xfer_fence
    unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(;
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))

    transition_image!(cmd, image,
        Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, Vulkan.PIPELINE_STAGE_TRANSFER_BIT,
        Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, Vulkan.ACCESS_TRANSFER_READ_BIT)

    region = Vulkan.BufferImageCopy(
        UInt64(0), UInt32(0), UInt32(0),
        Vulkan.ImageSubresourceLayers(Vulkan.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        Vulkan.Offset3D(0, 0, 0),
        Vulkan.Extent3D(UInt32(w), UInt32(h), UInt32(1)),
    )
    Vulkan.cmd_copy_image_to_buffer(cmd, image,
        Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buf, [region])

    # Transition back to COLOR_ATTACHMENT_OPTIMAL so present_frame! can transition to PRESENT_SRC
    transition_image!(cmd, image,
        Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        Vulkan.PIPELINE_STAGE_TRANSFER_BIT, Vulkan.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        Vulkan.ACCESS_TRANSFER_READ_BIT, Vulkan.AccessFlag(0))

    unwrap(Vulkan.end_command_buffer(cmd))

    submit_info = Vulkan.SubmitInfo([], [], [cmd], [])
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=fence))
    unwrap(Vulkan.wait_for_fences(dev, [fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [fence]))
    drain_deferred_frees!(ctx.default_bq)
    drain_deferred_as_frees!(ctx.default_bq)

    pixels = Matrix{T}(undef, w, h)
    unsafe_copyto!(Ptr{UInt8}(pointer(pixels)), Ptr{UInt8}(mapped_ptr), nbytes)
    return pixels
end

# ── Render Target Subtypes ──

"""Render to a swapchain window image."""
struct WindowTarget <: RenderTarget
    window::RenderWindow
end

"""Render to an offscreen framebuffer."""
struct OffscreenTarget <: RenderTarget
    fb::LavaFramebuffer
end
