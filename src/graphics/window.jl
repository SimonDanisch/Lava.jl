# Window and swapchain management for Lava.jl
#
# Uses GLFW for window creation, Vulkan for surface/swapchain.
# Requires VK_KHR_surface + platform-specific surface extension.

import GLFW

"""
    RenderWindow

A window with Vulkan surface and swapchain for presenting rendered frames.
Uses GLFW for cross-platform window management.
"""
mutable struct RenderWindow
    handle::GLFW.Window
    surface::Vulkan.SurfaceKHR
    swapchain::Union{Nothing, Vulkan.SwapchainKHR}
    images::Vector{Vulkan.Image}
    views::Vector{Vulkan.ImageView}
    format::Vulkan.Format
    extent::Vulkan.Extent2D
    # Per-frame-in-flight sync (rotating ring buffer)
    image_available::Vector{Vulkan.Semaphore}
    render_finished::Vector{Vulkan.Semaphore}
    in_flight::Vector{Vulkan.Fence}
    current_frame::Int  # index into sync arrays (1-based, wraps)
    # Per-frame batch: reclaimed after fence wait in acquire_next_image!
    frame_batches::Vector{Union{Nothing, CommandBatch}}
    # Current frame state
    current_image_idx::UInt32
    acquired::Bool
    # Owning context — swapchain/surface are bound to a specific device.
    ctx::VkContext
end

"""
    RenderWindow(width, height; title="Lava", vsync=true)

Create a new window with Vulkan surface and swapchain.
"""
function RenderWindow(width::Integer, height::Integer;
                      ctx::VkContext=vk_context(),
                      title::String="Lava", vsync::Bool=true)

    # Initialize GLFW (no OpenGL context — we use Vulkan)
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.RESIZABLE, true)

    handle = GLFW.CreateWindow(width, height, title)

    # Create Vulkan surface via GLFW
    surface_ptr = GLFW.CreateWindowSurface(ctx.instance.vks, handle)
    surface_ptr == C_NULL && throw(LavaError("window creation", "GLFW CreateWindowSurface failed",
                                "Ensure Vulkan drivers support window surfaces"))
    surface = Vulkan.SurfaceKHR(surface_ptr, ctx.instance, ctx.instance.refcount)

    win = RenderWindow(
        handle, surface, nothing,
        Vulkan.Image[], Vulkan.ImageView[],
        Vulkan.FORMAT_B8G8R8A8_SRGB,
        Vulkan.Extent2D(width, height),
        Vulkan.Semaphore[], Vulkan.Semaphore[], Vulkan.Fence[],
        1,  # current_frame
        Union{Nothing, CommandBatch}[],  # frame_batches
        UInt32(0), false,
        ctx,
    )

    create_swapchain!(win; vsync)
    return win
end

"""
    create_swapchain!(win::RenderWindow; vsync=true)

Create or recreate the swapchain for the window.
"""
function create_swapchain!(win::RenderWindow; vsync::Bool=true)
    ctx = win.ctx
    dev = ctx.device
    phys = ctx.physical_device

    # Query surface capabilities
    caps = unwrap(Vulkan.get_physical_device_surface_capabilities_khr(phys, win.surface))

    # Pick format (prefer B8G8R8A8_SRGB)
    formats = unwrap(Vulkan.get_physical_device_surface_formats_khr(phys; surface=win.surface))
    chosen_format = formats[1]
    for f in formats
        if f.format == Vulkan.FORMAT_B8G8R8A8_SRGB &&
           f.color_space == Vulkan.COLOR_SPACE_SRGB_NONLINEAR_KHR
            chosen_format = f
            break
        end
    end
    win.format = chosen_format.format

    # Pick present mode
    present_modes = unwrap(Vulkan.get_physical_device_surface_present_modes_khr(phys; surface=win.surface))
    present_mode = Vulkan.PRESENT_MODE_FIFO_KHR  # vsync, always supported
    if !vsync
        for pm in present_modes
            if pm == Vulkan.PRESENT_MODE_MAILBOX_KHR
                present_mode = pm
                break
            elseif pm == Vulkan.PRESENT_MODE_IMMEDIATE_KHR
                present_mode = pm
            end
        end
    end

    # Determine extent
    if caps.current_extent.width != typemax(UInt32)
        win.extent = caps.current_extent
    else
        w, h = GLFW.GetFramebufferSize(win.handle)
        win.extent = Vulkan.Extent2D(
            clamp(UInt32(w), caps.min_image_extent.width, caps.max_image_extent.width),
            clamp(UInt32(h), caps.min_image_extent.height, caps.max_image_extent.height),
        )
    end

    # Image count
    image_count = caps.min_image_count + 1
    if caps.max_image_count > 0
        image_count = min(image_count, caps.max_image_count)
    end

    old_swapchain = win.swapchain

    kw = Dict{Symbol,Any}()
    if old_swapchain !== nothing
        kw[:old_swapchain] = old_swapchain
    end

    swapchain = Vulkan.SwapchainKHR(
        dev, win.surface,
        image_count, chosen_format.format, chosen_format.color_space,
        win.extent, UInt32(1),
        Vulkan.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Vulkan.IMAGE_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[],
        caps.current_transform,
        Vulkan.COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        present_mode, true;
        kw...,
    )

    win.swapchain = swapchain
    win.images = unwrap(Vulkan.get_swapchain_images_khr(dev, swapchain))

    # Create image views
    win.views = Vulkan.ImageView[]
    for img in win.images
        view = Vulkan.ImageView(dev, img, Vulkan.IMAGE_VIEW_TYPE_2D, win.format,
            Vulkan.ComponentMapping(
                Vulkan.COMPONENT_SWIZZLE_IDENTITY,
                Vulkan.COMPONENT_SWIZZLE_IDENTITY,
                Vulkan.COMPONENT_SWIZZLE_IDENTITY,
                Vulkan.COMPONENT_SWIZZLE_IDENTITY,
            ),
            Vulkan.ImageSubresourceRange(
                Vulkan.IMAGE_ASPECT_COLOR_BIT,
                UInt32(0), UInt32(1), UInt32(0), UInt32(1),
            ),
        )
        push!(win.views, view)
    end

    # Create per-frame sync primitives (one set per swapchain image)
    n = length(win.images)
    win.image_available = [Vulkan.Semaphore(dev) for _ in 1:n]
    win.render_finished = [Vulkan.Semaphore(dev) for _ in 1:n]
    win.in_flight = [Vulkan.Fence(dev; flags=Vulkan.FENCE_CREATE_SIGNALED_BIT) for _ in 1:n]
    win.frame_batches = Union{Nothing, CommandBatch}[nothing for _ in 1:n]
    win.current_frame = 1
end

"""
    acquire_next_image!(win::RenderWindow) -> UInt32

Acquire the next swapchain image. Returns the image index.
Must be called before recording rendering commands.
"""
function acquire_next_image!(win::RenderWindow)
    ctx = win.ctx
    dev = ctx.device
    fi = win.current_frame

    wait_for_fences!(ctx.default_bq, [win.in_flight[fi]])
    unwrap(Vulkan.reset_fences(dev, [win.in_flight[fi]]))

    # Reclaim batch from previous frame in this slot — GPU is done (fence waited above)
    old_batch = win.frame_batches[fi]
    if old_batch !== nothing
        old_batch.recording = false
        old_batch.dispatch_count = 0
        old_batch.last_was_rt = false
        empty!(old_batch.pinned)
        empty!(old_batch.wait_semaphores)
        push!(ctx.default_bq.free_batches, old_batch)
        win.frame_batches[fi] = nothing
    end

    idx, _ = throw_if_error(ctx, "vkAcquireNextImageKHR",
        Vulkan.acquire_next_image_khr(dev, win.swapchain,
            typemax(UInt64); semaphore=win.image_available[fi]))

    win.current_image_idx = idx
    win.acquired = true
    return idx
end

"""
    present!(win::RenderWindow)

Present the rendered frame to the screen.
Must be called after recording and submitting rendering commands.
"""
function present!(win::RenderWindow)
    win.acquired || error("Cannot present: no image acquired (call acquire_next_image! first)")
    ctx = win.ctx
    fi = win.current_frame
    present_info = Vulkan.PresentInfoKHR(
        [win.render_finished[fi]],
        [win.swapchain],
        [win.current_image_idx],
    )
    throw_if_error(ctx, "vkQueuePresentKHR",
        Vulkan.queue_present_khr(ctx.default_bq.queue, present_info))

    win.acquired = false
    # Advance to next frame-in-flight slot
    win.current_frame = mod1(fi + 1, length(win.in_flight))
end

"""
    resize!(win::RenderWindow)

Handle window resize by recreating the swapchain.
"""
function Base.resize!(win::RenderWindow)
    ctx = win.ctx
    Vulkan.device_wait_idle(ctx.device)
    # Reclaim in-flight frame batches before recreating swapchain
    for i in eachindex(win.frame_batches)
        batch = win.frame_batches[i]
        if batch !== nothing
            batch.recording = false
            batch.dispatch_count = 0
            batch.last_was_rt = false
            empty!(batch.pinned)
            empty!(batch.wait_semaphores)
            push!(ctx.default_bq.free_batches, batch)
            win.frame_batches[i] = nothing
        end
    end
    create_swapchain!(win)
end

function Base.isopen(win::RenderWindow)
    win.handle.handle != C_NULL && !GLFW.WindowShouldClose(win.handle)
end

function Base.close(win::RenderWindow)
    # Idempotent -- safe to call multiple times
    win.handle.handle == C_NULL && return
    ctx = win.ctx
    Vulkan.device_wait_idle(ctx.device)
    # Reclaim any in-flight frame batches (GPU is idle after device_wait_idle)
    for i in eachindex(win.frame_batches)
        batch = win.frame_batches[i]
        if batch !== nothing
            batch.recording = false
            batch.dispatch_count = 0
            batch.last_was_rt = false
            empty!(batch.pinned)
            empty!(batch.wait_semaphores)
            push!(ctx.default_bq.free_batches, batch)
            win.frame_batches[i] = nothing
        end
    end
    GLFW.DestroyWindow(win.handle)
    win.handle = GLFW.Window(C_NULL)
end

Base.size(win::RenderWindow) = (Int(win.extent.width), Int(win.extent.height))
