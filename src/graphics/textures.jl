# Texture types and management for Lava.jl
#
# VkImage + VkSampler + descriptor set management for texture sampling in shaders.

abstract type AbstractLavaTexture{T, N} end

"""2D texture backed by VkImage."""
struct LavaTexture2D{T} <: AbstractLavaTexture{T, 2}
    image::Vulkan.Image
    memory::Vulkan.DeviceMemory
    view::Vulkan.ImageView
    width::Int
    height::Int
    format::Vulkan.Format
end

"""1D texture backed by VkImage."""
struct LavaTexture1D{T} <: AbstractLavaTexture{T, 1}
    image::Vulkan.Image
    memory::Vulkan.DeviceMemory
    view::Vulkan.ImageView
    width::Int
    format::Vulkan.Format
end

"""Reusable sampler configuration."""
struct LavaSampler
    handle::Vulkan.Sampler
    filter::Symbol
    wrap::Symbol
    anisotropy::Float32
end

"""Combined texture + sampler, ready for binding."""
struct SampledTexture{T, N}
    texture::AbstractLavaTexture{T, N}
    sampler::LavaSampler
end

# ── Sampler Construction ──

function LavaSampler(; filter::Symbol=:linear, wrap::Symbol=:repeat, anisotropy::Real=0.0f0)
    ctx = vk_context()
    dev = ctx.device

    vk_filter = filter == :nearest ? Vulkan.FILTER_NEAREST :
                filter == :linear  ? Vulkan.FILTER_LINEAR :
                error("Unknown filter: $filter")

    vk_wrap = wrap == :repeat      ? Vulkan.SAMPLER_ADDRESS_MODE_REPEAT :
              wrap == :clamp       ? Vulkan.SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE :
              wrap == :mirror      ? Vulkan.SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT :
              wrap == :clamp_border ? Vulkan.SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER :
              error("Unknown wrap mode: $wrap")

    sampler = Vulkan.Sampler(dev,
        vk_filter, vk_filter,
        Vulkan.SAMPLER_MIPMAP_MODE_LINEAR,
        vk_wrap, vk_wrap, vk_wrap,
        0.0f0,  # mip LOD bias
        anisotropy > 0,
        Float32(anisotropy),
        false, Vulkan.COMPARE_OP_ALWAYS,
        0.0f0, 0.0f0,
        Vulkan.BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        false,
    )

    LavaSampler(sampler, filter, wrap, Float32(anisotropy))
end

# ── Texture Construction ──

"""Create a 2D texture from a matrix of data."""
function LavaTexture2D(data::Matrix{T}; filter=:linear, wrap=:repeat) where T
    ctx = vk_context()
    dev = ctx.device
    phys = ctx.physical_device

    h, w = size(data)
    format = _julia_to_vk_format(T)

    image = Vulkan.Image(dev,
        Vulkan.IMAGE_TYPE_2D, format,
        Vulkan.Extent3D(UInt32(w), UInt32(h), UInt32(1)),
        UInt32(1), UInt32(1),
        Vulkan.SAMPLE_COUNT_1_BIT,
        Vulkan.IMAGE_TILING_OPTIMAL,
        Vulkan.IMAGE_USAGE_SAMPLED_BIT | Vulkan.IMAGE_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[],
        Vulkan.IMAGE_LAYOUT_UNDEFINED,
    )

    memory = _alloc_image_memory(dev, phys, image)

    view = Vulkan.ImageView(dev, image, Vulkan.IMAGE_VIEW_TYPE_2D, format,
        Vulkan.ComponentMapping(
            Vulkan.COMPONENT_SWIZZLE_IDENTITY, Vulkan.COMPONENT_SWIZZLE_IDENTITY,
            Vulkan.COMPONENT_SWIZZLE_IDENTITY, Vulkan.COMPONENT_SWIZZLE_IDENTITY),
        Vulkan.ImageSubresourceRange(Vulkan.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(1), UInt32(0), UInt32(1)))

    tex = LavaTexture2D{T}(image, memory, view, w, h, format)

    # Upload data
    _upload_texture_data!(tex, data)

    return tex
end

"""Upload pixel data to a texture via staging buffer."""
function _upload_texture_data!(tex::LavaTexture2D{T}, data::Matrix{T}) where T
    ctx = vk_context()
    dev = ctx.device

    # Create staging buffer
    bytes = reinterpret(UInt8, vec(collect(data)))
    staging = vk_alloc(length(bytes);
        usage=Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT,
        host_visible=true)
    upload!(staging, bytes)

    # Flush pending commands first
    vk_flush!()

    # One-shot command buffer for transfer
    cmd = ctx.cmd_buf
    unwrap(Vulkan.begin_command_buffer(cmd, Vulkan.CommandBufferBeginInfo(;
        flags=Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))

    # Transition to TRANSFER_DST
    _transition_image!(cmd, tex.image,
        Vulkan.IMAGE_LAYOUT_UNDEFINED, Vulkan.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        Vulkan.PIPELINE_STAGE_TOP_OF_PIPE_BIT, Vulkan.PIPELINE_STAGE_TRANSFER_BIT,
        Vulkan.AccessFlag(0), Vulkan.ACCESS_TRANSFER_WRITE_BIT)

    # Copy buffer to image
    region = Vulkan.BufferImageCopy(
        UInt64(0), UInt32(0), UInt32(0),
        Vulkan.ImageSubresourceLayers(Vulkan.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        Vulkan.Offset3D(0, 0, 0),
        Vulkan.Extent3D(UInt32(tex.width), UInt32(tex.height), UInt32(1)),
    )
    Vulkan.cmd_copy_buffer_to_image(cmd, staging.buffer, tex.image,
        Vulkan.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])

    # Transition to SHADER_READ_ONLY
    _transition_image!(cmd, tex.image,
        Vulkan.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, Vulkan.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        Vulkan.PIPELINE_STAGE_TRANSFER_BIT, Vulkan.PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        Vulkan.ACCESS_TRANSFER_WRITE_BIT, Vulkan.ACCESS_SHADER_READ_BIT)

    # Submit
    unwrap(Vulkan.end_command_buffer(cmd))
    submit_info = Vulkan.SubmitInfo([], [], [cmd], [])
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=ctx.fence))
    unwrap(Vulkan.wait_for_fences(dev, [ctx.fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [ctx.fence]))
    ctx.recording = false
    ctx.dispatch_count = 0
end

# ── Format Mapping ──

function _julia_to_vk_format(::Type{T}) where T
    T == Float32      ? Vulkan.FORMAT_R32_SFLOAT :
    T == NTuple{4, Float32} ? Vulkan.FORMAT_R32G32B32A32_SFLOAT :
    T == NTuple{3, Float32} ? Vulkan.FORMAT_R32G32B32_SFLOAT :
    T == NTuple{2, Float32} ? Vulkan.FORMAT_R32G32_SFLOAT :
    T == UInt8        ? Vulkan.FORMAT_R8_UNORM :
    T == NTuple{4, UInt8} ? Vulkan.FORMAT_R8G8B8A8_UNORM :
    error("No VkFormat mapping for Julia type: $T")
end

# ── Descriptor Set for Textures ──

struct TextureBindings
    layout::Vulkan.DescriptorSetLayout
    pool::Vulkan.DescriptorPool
    set::Vulkan.DescriptorSet
end

"""Create a descriptor set binding combined image samplers."""
function bind_textures(textures::Vector{<:SampledTexture})
    ctx = vk_context()
    dev = ctx.device

    n = length(textures)
    bindings = [Vulkan.DescriptorSetLayoutBinding(
        UInt32(i - 1),
        Vulkan.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        UInt32(1),
        Vulkan.SHADER_STAGE_FRAGMENT_BIT | Vulkan.SHADER_STAGE_VERTEX_BIT,
    ) for i in 1:n]

    layout = Vulkan.DescriptorSetLayout(dev, bindings)

    pool = Vulkan.DescriptorPool(dev, UInt32(1),
        [Vulkan.DescriptorPoolSize(Vulkan.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, UInt32(n))])

    alloc_info = Vulkan.DescriptorSetAllocateInfo(pool, [layout])
    sets = unwrap(Vulkan.allocate_descriptor_sets(dev, alloc_info))
    dset = sets[1]

    # Write descriptors
    writes = Vulkan.WriteDescriptorSet[]
    for (i, st) in enumerate(textures)
        img_info = Vulkan.DescriptorImageInfo(
            st.sampler.handle,
            st.texture.view,
            Vulkan.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        push!(writes, Vulkan.WriteDescriptorSet(
            dset, UInt32(i - 1), UInt32(0), UInt32(1),
            Vulkan.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            [img_info], [], [],
        ))
    end
    Vulkan.update_descriptor_sets(dev, writes, [])

    TextureBindings(layout, pool, dset)
end

# Convenience
Base.:*(tex::AbstractLavaTexture, sam::LavaSampler) = SampledTexture(tex, sam)
LavaTexture(data::Matrix{T}; kw...) where T = LavaTexture2D(data; kw...)
