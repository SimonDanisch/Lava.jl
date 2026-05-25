# Graphics pipeline creation for Lava.jl
#
# Creates VkGraphicsPipeline using VK_KHR_dynamic_rendering (no VkRenderPass).
# Pipeline state (blend, cull, depth, topology) configured via dispatch on config types.

"""
    CompiledGraphicsPipeline

A compiled graphics pipeline ready for draw commands.
"""
struct CompiledGraphicsPipeline
    pipeline::Vulkan.Pipeline
    pipeline_layout::Vulkan.PipelineLayout
    modules::Vector{Vulkan.ShaderModule}
    push_constant_size::UInt32
    descriptor_set_layout::Union{Nothing, Vulkan.DescriptorSetLayout}
    push_stage_flags::Vulkan.ShaderStageFlag
    # Pipeline state (for debug/inspection)
    color_format::Vulkan.Format
    has_depth::Bool
end

const GFX_PIPELINE_CACHE = Dict{UInt64, CompiledGraphicsPipeline}()

# Clear graphics pipeline cache on vk_reset_device!
push!(RESET_CALLBACKS, function()
    empty!(GFX_PIPELINE_CACHE)
end)

"""
    create_graphics_pipeline(vertex_spirv, fragment_spirv;
        blend=Opaque(), cull=CullBack(), topology=TriangleList(), depth=DepthLess(),
        color_format=Vulkan.FORMAT_B8G8R8A8_SRGB, push_constant_size=8,
        geometry_spirv=nothing, geometry_config=nothing,
        tess_ctrl_spirv=nothing, tess_eval_spirv=nothing, tess_config=nothing,
        descriptor_set_layout=nothing) -> CompiledGraphicsPipeline

Create a graphics pipeline from SPIR-V shader binaries.
Uses VK_KHR_dynamic_rendering — no VkRenderPass needed.
"""
function create_graphics_pipeline(vertex_spirv::Vector{UInt8},
                                    fragment_spirv::Vector{UInt8};
                                    ctx::VkContext=vk_context(),
                                    blend::BlendMode=Opaque(),
                                    cull::CullFace=CullBack(),
                                    topology::Topology=TriangleList(),
                                    depth::DepthMode=DepthLess(),
                                    color_format::Vulkan.Format=Vulkan.FORMAT_B8G8R8A8_SRGB,
                                    depth_format::Vulkan.Format=Vulkan.FORMAT_D32_SFLOAT,
                                    push_constant_size::Integer=8,
                                    geometry_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                                    tess_ctrl_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                                    tess_eval_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                                    tess_config::Union{Nothing, TessConfig}=nothing,
                                    descriptor_set_layout::Union{Nothing, Vulkan.DescriptorSetLayout}=nothing)
    dev = ctx.device

    # Create shader modules
    vert_mod = create_gfx_shader_module(dev, vertex_spirv)
    frag_mod = create_gfx_shader_module(dev, fragment_spirv)
    modules = Vulkan.ShaderModule[vert_mod, frag_mod]

    stages = Vulkan.PipelineShaderStageCreateInfo[
        Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_VERTEX_BIT, vert_mod, "main"),
        Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_FRAGMENT_BIT, frag_mod, "main"),
    ]

    # Optional: geometry shader
    if geometry_spirv !== nothing
        geom_mod = create_gfx_shader_module(dev, geometry_spirv)
        push!(modules, geom_mod)
        push!(stages, Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_GEOMETRY_BIT, geom_mod, "main"))
    end

    # Optional: tessellation shaders
    if tess_ctrl_spirv !== nothing && tess_eval_spirv !== nothing
        tc_mod = create_gfx_shader_module(dev, tess_ctrl_spirv)
        te_mod = create_gfx_shader_module(dev, tess_eval_spirv)
        push!(modules, tc_mod, te_mod)
        push!(stages, Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_TESSELLATION_CONTROL_BIT, tc_mod, "main"))
        push!(stages, Vulkan.PipelineShaderStageCreateInfo(
            Vulkan.SHADER_STAGE_TESSELLATION_EVALUATION_BIT, te_mod, "main"))
    end

    # Vertex input: EMPTY (BDA vertex pulling)
    vertex_input = Vulkan.PipelineVertexInputStateCreateInfo([], [])

    # Input assembly
    vk_topo = vk_topology(topology)
    input_assembly = Vulkan.PipelineInputAssemblyStateCreateInfo(vk_topo, false)

    # Tessellation state (if applicable)
    tess_state = C_NULL
    if tess_config !== nothing
        tess_state = Vulkan.PipelineTessellationStateCreateInfo(UInt32(tess_config.patch_vertices))
    end

    # Viewport/scissor: dynamic (set via cmd_set_viewport/cmd_set_scissor)
    # Pass dummy viewport/scissor — will be overridden by dynamic state
    dummy_viewport = Vulkan.Viewport(0.0f0, 0.0f0, 1.0f0, 1.0f0, 0.0f0, 1.0f0)
    dummy_scissor = Vulkan.Rect2D(Vulkan.Offset2D(0, 0), Vulkan.Extent2D(1, 1))
    viewport_state = Vulkan.PipelineViewportStateCreateInfo(;
        viewports=[dummy_viewport], scissors=[dummy_scissor])

    # Rasterization
    cull_mode = vk_cull(cull)
    rasterization = Vulkan.PipelineRasterizationStateCreateInfo(
        false, false,
        Vulkan.POLYGON_MODE_FILL,
        Vulkan.FRONT_FACE_COUNTER_CLOCKWISE,
        false, 0.0f0, 0.0f0, 0.0f0,
        1.0f0;  # line width
        cull_mode=cull_mode,
    )

    # Multisampling: no MSAA
    multisample = Vulkan.PipelineMultisampleStateCreateInfo(
        Vulkan.SAMPLE_COUNT_1_BIT, false, 1.0f0, false, false)

    # Depth/stencil
    has_depth = !(depth isa DepthOff)
    depth_enable, depth_compare = vk_depth(depth)
    depth_stencil = Vulkan.PipelineDepthStencilStateCreateInfo(
        depth_enable, depth_enable,
        depth_compare,
        false,    # depth bounds test
        false,    # stencil test
        Vulkan.StencilOpState(Vulkan.STENCIL_OP_KEEP, Vulkan.STENCIL_OP_KEEP,
            Vulkan.STENCIL_OP_KEEP, Vulkan.COMPARE_OP_ALWAYS, 0, 0, 0),
        Vulkan.StencilOpState(Vulkan.STENCIL_OP_KEEP, Vulkan.STENCIL_OP_KEEP,
            Vulkan.STENCIL_OP_KEEP, Vulkan.COMPARE_OP_ALWAYS, 0, 0, 0),
        0.0f0, 1.0f0,
    )

    # Color blend
    blend_attachment = vk_blend(blend)
    color_blend = Vulkan.PipelineColorBlendStateCreateInfo(
        false, Vulkan.LOGIC_OP_COPY,
        [blend_attachment],
        (0.0f0, 0.0f0, 0.0f0, 0.0f0),
    )

    # Dynamic state
    dynamic_states = [Vulkan.DYNAMIC_STATE_VIEWPORT, Vulkan.DYNAMIC_STATE_SCISSOR]
    dynamic_state = Vulkan.PipelineDynamicStateCreateInfo(dynamic_states)

    # Pipeline layout
    all_stage_flags = Vulkan.SHADER_STAGE_VERTEX_BIT | Vulkan.SHADER_STAGE_FRAGMENT_BIT
    if geometry_spirv !== nothing
        all_stage_flags |= Vulkan.SHADER_STAGE_GEOMETRY_BIT
    end
    if tess_ctrl_spirv !== nothing
        all_stage_flags |= Vulkan.SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                            Vulkan.SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    end

    push_ranges = Vulkan.PushConstantRange[]
    if push_constant_size > 0
        push!(push_ranges, Vulkan.PushConstantRange(
            all_stage_flags, UInt32(0), UInt32(push_constant_size)))
    end

    ds_layouts = Vulkan.DescriptorSetLayout[]
    if descriptor_set_layout !== nothing
        push!(ds_layouts, descriptor_set_layout)
    end

    layout = Vulkan.PipelineLayout(dev, ds_layouts, push_ranges)

    # Dynamic rendering info (Vulkan 1.3+)
    rendering_info = Vulkan.PipelineRenderingCreateInfo(
        UInt32(0),           # view mask
        [color_format],      # color attachment formats
        has_depth ? depth_format : Vulkan.FORMAT_UNDEFINED,
        Vulkan.FORMAT_UNDEFINED,  # stencil format
    )

    # Create graphics pipeline (dynamic rendering — null render pass)
    ci = Vulkan.GraphicsPipelineCreateInfo(
        stages,
        rasterization,
        layout, UInt32(0), Int32(-1);
        vertex_input_state=vertex_input,
        input_assembly_state=input_assembly,
        tessellation_state=tess_state,
        viewport_state=viewport_state,
        multisample_state=multisample,
        depth_stencil_state=depth_stencil,
        color_blend_state=color_blend,
        dynamic_state=dynamic_state,
        next=rendering_info,
    )

    pipelines, _ = @vk_checked "vkCreateGraphicsPipelines" Vulkan.create_graphics_pipelines(
        dev, [ci]; pipeline_cache=vk_context().pipeline_cache)
    pipeline = pipelines[1]

    return CompiledGraphicsPipeline(
        pipeline, layout, modules,
        UInt32(push_constant_size),
        descriptor_set_layout,
        all_stage_flags,
        color_format, has_depth,
    )
end

function create_gfx_shader_module(dev::Vulkan.Device, spirv_bytes::Vector{UInt8})
    @assert length(spirv_bytes) % 4 == 0 "SPIR-V binary must be 4-byte aligned"
    code_u32 = reinterpret(UInt32, spirv_bytes)
    Vulkan.ShaderModule(dev, length(spirv_bytes), code_u32)
end

# ── Pipeline State Dispatch ──

vk_topology(::TriangleList)  = Vulkan.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
vk_topology(::TriangleStrip) = Vulkan.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
vk_topology(::LineList)      = Vulkan.PRIMITIVE_TOPOLOGY_LINE_LIST
vk_topology(::LineStrip)     = Vulkan.PRIMITIVE_TOPOLOGY_LINE_STRIP
vk_topology(::PointList)     = Vulkan.PRIMITIVE_TOPOLOGY_POINT_LIST
vk_topology(::PatchList)              = Vulkan.PRIMITIVE_TOPOLOGY_PATCH_LIST
vk_topology(::LineListAdjacency)      = Vulkan.PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY

vk_cull(::NoCull)    = Vulkan.CULL_MODE_NONE
vk_cull(::CullBack)  = Vulkan.CULL_MODE_BACK_BIT
vk_cull(::CullFront) = Vulkan.CULL_MODE_FRONT_BIT

function vk_depth(::DepthLess)
    (true, Vulkan.COMPARE_OP_LESS)
end
function vk_depth(::DepthLessEq)
    (true, Vulkan.COMPARE_OP_LESS_OR_EQUAL)
end
function vk_depth(::DepthGreater)
    (true, Vulkan.COMPARE_OP_GREATER)
end
function vk_depth(::DepthAlways)
    (true, Vulkan.COMPARE_OP_ALWAYS)
end
function vk_depth(::DepthOff)
    (false, Vulkan.COMPARE_OP_ALWAYS)
end

function vk_blend(::Opaque)
    Vulkan.PipelineColorBlendAttachmentState(
        false,
        Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_FACTOR_ZERO, Vulkan.BLEND_OP_ADD,
        Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_FACTOR_ZERO, Vulkan.BLEND_OP_ADD,
        Vulkan.COLOR_COMPONENT_R_BIT | Vulkan.COLOR_COMPONENT_G_BIT |
        Vulkan.COLOR_COMPONENT_B_BIT | Vulkan.COLOR_COMPONENT_A_BIT,
    )
end

function vk_blend(::AlphaBlend)
    Vulkan.PipelineColorBlendAttachmentState(
        true,
        Vulkan.BLEND_FACTOR_SRC_ALPHA, Vulkan.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, Vulkan.BLEND_OP_ADD,
        Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_FACTOR_ZERO, Vulkan.BLEND_OP_ADD,
        Vulkan.COLOR_COMPONENT_R_BIT | Vulkan.COLOR_COMPONENT_G_BIT |
        Vulkan.COLOR_COMPONENT_B_BIT | Vulkan.COLOR_COMPONENT_A_BIT,
    )
end

function vk_blend(::Additive)
    Vulkan.PipelineColorBlendAttachmentState(
        true,
        Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_OP_ADD,
        Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_OP_ADD,
        Vulkan.COLOR_COMPONENT_R_BIT | Vulkan.COLOR_COMPONENT_G_BIT |
        Vulkan.COLOR_COMPONENT_B_BIT | Vulkan.COLOR_COMPONENT_A_BIT,
    )
end

function vk_blend(::Premultiplied)
    Vulkan.PipelineColorBlendAttachmentState(
        true,
        Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, Vulkan.BLEND_OP_ADD,
        Vulkan.BLEND_FACTOR_ONE, Vulkan.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, Vulkan.BLEND_OP_ADD,
        Vulkan.COLOR_COMPONENT_R_BIT | Vulkan.COLOR_COMPONENT_G_BIT |
        Vulkan.COLOR_COMPONENT_B_BIT | Vulkan.COLOR_COMPONENT_A_BIT,
    )
end

# ── Draw Recording ──

"""
    vk_draw!(pipeline::CompiledGraphicsPipeline, color_view::Vulkan.ImageView,
             color_image::Vulkan.Image, extent::Vulkan.Extent2D,
             vertex_count::Integer; push_data=UInt8[], instances=1,
             depth_view=nothing, clear_color=nothing,
             indices_buffer=nothing, index_count=0)

Record a draw command using dynamic rendering.
"""
function vk_draw!(bq::BatchQueue,
                   pipeline::CompiledGraphicsPipeline,
                   color_view::Vulkan.ImageView,
                   color_image::Vulkan.Image,
                   extent::Vulkan.Extent2D,
                   vertex_count::Integer;
                   push_data::Vector{UInt8}=UInt8[],
                   instances::Integer=1,
                   depth_view::Union{Nothing, Vulkan.ImageView}=nothing,
                   depth_image::Union{Nothing, Vulkan.Image}=nothing,
                   clear_color::Union{Nothing, NTuple{4, Float32}}=nothing,
                   indices_buffer::Union{Nothing, Vulkan.Buffer}=nothing,
                   index_count::Integer=0,
                   descriptor_set::Union{Nothing, Vulkan.DescriptorSet}=nothing)
    # Route through record_dispatch! so the prior-dispatch → draw barrier,
    # dispatch_count bookkeeping, CB-split logic, and dispatch-log accounting
    # all come from the single shared helper.  The do-block handles the
    # graphics-specific work (image transitions, dynamic rendering scope,
    # bind + draw + end_rendering).
    record_dispatch!(bq;
        dst_stage = Vulkan.PIPELINE_STAGE_VERTEX_SHADER_BIT |
                    Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        extra_dst_access = Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        info = "draw vtx=$vertex_count",
    ) do batch
        cmd = batch.cmd_buf

        # Transition color image to COLOR_ATTACHMENT_OPTIMAL
        transition_image!(cmd, color_image,
            Vulkan.IMAGE_LAYOUT_UNDEFINED, Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            Vulkan.PIPELINE_STAGE_TOP_OF_PIPE_BIT, Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            Vulkan.AccessFlag(0), Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)

        # Transition depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        if depth_image !== nothing
            depth_barrier = Vulkan.ImageMemoryBarrier(
                Vulkan.AccessFlag(0),
                Vulkan.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | Vulkan.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                Vulkan.IMAGE_LAYOUT_UNDEFINED, Vulkan.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                Vulkan.QUEUE_FAMILY_IGNORED, Vulkan.QUEUE_FAMILY_IGNORED,
                depth_image,
                Vulkan.ImageSubresourceRange(Vulkan.IMAGE_ASPECT_DEPTH_BIT,
                    UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
            )
            Vulkan.cmd_pipeline_barrier(cmd, [], [], [depth_barrier];
                src_stage_mask=Vulkan.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                dst_stage_mask=Vulkan.PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
        end

        # Color attachment for dynamic rendering
        clear_val = if clear_color !== nothing
            Vulkan.ClearValue(Vulkan.ClearColorValue(clear_color))
        else
            Vulkan.ClearValue(Vulkan.ClearColorValue((0.0f0, 0.0f0, 0.0f0, 1.0f0)))
        end

        load_op = clear_color !== nothing ? Vulkan.ATTACHMENT_LOAD_OP_CLEAR : Vulkan.ATTACHMENT_LOAD_OP_LOAD

        color_attachment = Vulkan.RenderingAttachmentInfo(
            Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            Vulkan.IMAGE_LAYOUT_UNDEFINED,  # resolve image layout (unused)
            load_op,
            Vulkan.ATTACHMENT_STORE_OP_STORE,
            clear_val;
            image_view=color_view,
            resolve_mode=Vulkan.RESOLVE_MODE_NONE,
        )

        # Depth attachment (optional)
        depth_attachment = C_NULL
        if depth_view !== nothing
            depth_clear = Vulkan.ClearValue(Vulkan.ClearDepthStencilValue(1.0f0, UInt32(0)))
            depth_attachment = Vulkan.RenderingAttachmentInfo(
                Vulkan.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                Vulkan.IMAGE_LAYOUT_UNDEFINED,
                Vulkan.ATTACHMENT_LOAD_OP_CLEAR,
                Vulkan.ATTACHMENT_STORE_OP_DONT_CARE,
                depth_clear;
                image_view=depth_view,
                resolve_mode=Vulkan.RESOLVE_MODE_NONE,
            )
        end

        render_area = Vulkan.Rect2D(Vulkan.Offset2D(0, 0), extent)

        rendering_info = Vulkan.RenderingInfo(
            render_area,
            UInt32(1),  # layer count
            UInt32(0),  # view mask
            [color_attachment];
            depth_attachment=depth_attachment,
        )

        Vulkan.cmd_begin_rendering(cmd, rendering_info)

        # Bind pipeline + pin for batch lifetime
        Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)
        pin!(batch, pipeline)

        # Bind descriptor set (for textures)
        if descriptor_set !== nothing
            Vulkan.cmd_bind_descriptor_sets(cmd, Vulkan.PIPELINE_BIND_POINT_GRAPHICS,
                pipeline.pipeline_layout, UInt32(0), [descriptor_set], UInt32[])
        end

        # Dynamic viewport + scissor
        viewport = Vulkan.Viewport(0.0f0, 0.0f0,
            Float32(extent.width), Float32(extent.height),
            0.0f0, 1.0f0)
        Vulkan.cmd_set_viewport(cmd, [viewport])

        scissor = Vulkan.Rect2D(Vulkan.Offset2D(0, 0), extent)
        Vulkan.cmd_set_scissor(cmd, [scissor])

        # Push constants
        if !isempty(push_data)
            GC.@preserve push_data begin
                Vulkan.cmd_push_constants(cmd, pipeline.pipeline_layout,
                    pipeline.push_stage_flags, UInt32(0), UInt32(length(push_data)),
                    Ptr{Nothing}(pointer(push_data)))
            end
        end

        # Draw
        if indices_buffer !== nothing
            Vulkan.cmd_bind_index_buffer(cmd, indices_buffer, UInt64(0), Vulkan.INDEX_TYPE_UINT32)
            Vulkan.cmd_draw_indexed(cmd, UInt32(index_count), UInt32(instances),
                                     UInt32(0), Int32(0), UInt32(0))
        else
            Vulkan.cmd_draw(cmd, UInt32(vertex_count), UInt32(instances), UInt32(0), UInt32(0))
        end

        Vulkan.cmd_end_rendering(cmd)
    end
end

"""Transition an image layout using a pipeline barrier."""
function transition_image!(cmd, image::Vulkan.Image,
                              old_layout, new_layout,
                              src_stage, dst_stage,
                              src_access, dst_access)
    barrier = Vulkan.ImageMemoryBarrier(
        src_access, dst_access,
        old_layout, new_layout,
        Vulkan.QUEUE_FAMILY_IGNORED, Vulkan.QUEUE_FAMILY_IGNORED,
        image,
        Vulkan.ImageSubresourceRange(Vulkan.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
    )
    Vulkan.cmd_pipeline_barrier(cmd, [], [], [barrier];
        src_stage_mask=src_stage, dst_stage_mask=dst_stage)
end

# =============================================================================
# Multi-draw rendering API
# =============================================================================
# Separates begin/draw/end for rendering multiple plots in a single pass.

"""
    vk_begin_pass!(color_view, color_image, extent; clear_color, depth_view)

Begin a dynamic rendering pass. Call `vk_draw_in_pass!` for each draw,
then `vk_end_pass!` to finish.
"""
function vk_begin_pass!(bq::BatchQueue,
                         color_view::Vulkan.ImageView,
                         color_image::Vulkan.Image,
                         extent::Vulkan.Extent2D;
                         clear_color::Union{Nothing, NTuple{4, Float32}}=(0f0, 0f0, 0f0, 1f0),
                         depth_view::Union{Nothing, Vulkan.ImageView}=nothing)
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    if clear_color !== nothing
        # Clear mode: discard old content, start fresh
        transition_image!(cmd, color_image,
            Vulkan.IMAGE_LAYOUT_UNDEFINED, Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            Vulkan.PIPELINE_STAGE_TOP_OF_PIPE_BIT, Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            Vulkan.AccessFlag(0), Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)

        clear_val = Vulkan.ClearValue(Vulkan.ClearColorValue(clear_color))
        load_op = Vulkan.ATTACHMENT_LOAD_OP_CLEAR
    else
        # Load mode: preserve existing content (for drawing on top of previous pass)
        transition_image!(cmd, color_image,
            Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT | Vulkan.ACCESS_COLOR_ATTACHMENT_READ_BIT)

        clear_val = Vulkan.ClearValue(Vulkan.ClearColorValue((0f0, 0f0, 0f0, 0f0)))
        load_op = Vulkan.ATTACHMENT_LOAD_OP_LOAD
    end

    color_attachment = Vulkan.RenderingAttachmentInfo(
        Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        Vulkan.IMAGE_LAYOUT_UNDEFINED,
        load_op,
        Vulkan.ATTACHMENT_STORE_OP_STORE,
        clear_val;
        image_view=color_view,
        resolve_mode=Vulkan.RESOLVE_MODE_NONE,
    )

    depth_attachment = C_NULL
    if depth_view !== nothing
        depth_clear = Vulkan.ClearValue(Vulkan.ClearDepthStencilValue(1.0f0, UInt32(0)))
        depth_attachment = Vulkan.RenderingAttachmentInfo(
            Vulkan.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            Vulkan.IMAGE_LAYOUT_UNDEFINED,
            Vulkan.ATTACHMENT_LOAD_OP_CLEAR,
            Vulkan.ATTACHMENT_STORE_OP_DONT_CARE,
            depth_clear;
            image_view=depth_view,
            resolve_mode=Vulkan.RESOLVE_MODE_NONE,
        )
    end

    render_area = Vulkan.Rect2D(Vulkan.Offset2D(0, 0), extent)
    rendering_info = Vulkan.RenderingInfo(
        render_area,
        UInt32(1), UInt32(0),
        [color_attachment];
        depth_attachment=depth_attachment,
    )
    Vulkan.cmd_begin_rendering(cmd, rendering_info)
end

"""
    vk_draw_in_pass!(pipeline, vertex_count; push_data, instances,
                     viewport, scissor)

Record a draw command within an active rendering pass.
Viewport/scissor are Vulkan structs for per-scene rendering.
"""
function vk_draw_in_pass!(bq::BatchQueue,
                           pipeline::CompiledGraphicsPipeline,
                           vertex_count::Integer;
                           push_data::Vector{UInt8}=UInt8[],
                           instances::Integer=1,
                           viewport::Union{Nothing, Vulkan.Viewport}=nothing,
                           scissor::Union{Nothing, Vulkan.Rect2D}=nothing)
    batch = bq.active_batch
    batch === nothing && error("vk_draw_in_pass! called without an active rendering pass")
    cmd = batch.cmd_buf

    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)

    # Set viewport and scissor
    if viewport !== nothing
        Vulkan.cmd_set_viewport(cmd, [viewport])
    end
    if scissor !== nothing
        Vulkan.cmd_set_scissor(cmd, [scissor])
    end

    # Push constants
    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(cmd, pipeline.pipeline_layout,
                pipeline.push_stage_flags, UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data)))
        end
    end

    Vulkan.cmd_draw(cmd, UInt32(vertex_count), UInt32(instances), UInt32(0), UInt32(0))
    batch.dispatch_count += 1
    batch.last_was_rt = false
    # Pin pipeline to batch — prevents GC from destroying it while command buffer references it
    pin!(batch, pipeline)
end

"""
    vk_draw_indexed_in_pass!(pipeline, index_count; push_data, indices_buffer)

Draw indexed geometry inside an active render pass (between vk_begin_pass!/vk_end_pass!).
Uses the provided index buffer for indexed drawing.
"""
function vk_draw_indexed_in_pass!(bq::BatchQueue,
                                   pipeline::CompiledGraphicsPipeline,
                                   index_count::Integer;
                                   push_data::Vector{UInt8}=UInt8[],
                                   indices_buffer::Vulkan.Buffer,
                                   instances::Integer=1)
    batch = bq.active_batch
    batch === nothing && error("vk_draw_indexed_in_pass! called without an active rendering pass")
    cmd = batch.cmd_buf

    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)

    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(cmd, pipeline.pipeline_layout,
                pipeline.push_stage_flags, UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data)))
        end
    end

    Vulkan.cmd_bind_index_buffer(cmd, indices_buffer, UInt64(0), Vulkan.INDEX_TYPE_UINT32)
    Vulkan.cmd_draw_indexed(cmd, UInt32(index_count), UInt32(instances),
                             UInt32(0), Int32(0), UInt32(0))
    batch.dispatch_count += 1
    batch.last_was_rt = false
    pin!(batch, pipeline)
end

"""
    vk_end_pass!(bq::BatchQueue)

End the current dynamic rendering pass on the given batch queue.
"""
function vk_end_pass!(bq::BatchQueue)
    batch = bq.active_batch
    batch === nothing && error("vk_end_pass! called without an active rendering pass")
    Vulkan.cmd_end_rendering(batch.cmd_buf)
end
