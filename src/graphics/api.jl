# High-level graphics API for Lava.jl
#
# GraphicsPipeline — the central type for configuring and dispatching
# graphics shaders. Supports lazy compilation, caching, and multiple
# draw dispatch via RenderTarget types.

"""
    GraphicsPipeline

High-level graphics pipeline wrapping Julia shader functions.
Compiles lazily on first use and caches the result.

# Fields
- `vertex`, `fragment`: Required Julia shader functions
- `geometry`: Optional (func, GeometryConfig) tuple
- `tess_control`, `tess_eval`: Optional tessellation stages
- `blend`, `cull`, `topology`, `depth`: Pipeline state types
"""
struct GraphicsPipeline{V, F, G, TC, TE, B<:BlendMode, C<:CullFace, T<:Topology, D<:DepthMode, VY}
    vertex::V
    fragment::F
    geometry::G       # Nothing or (func, GeometryConfig)
    tess_control::TC  # Nothing or (func, TessConfig)
    tess_eval::TE     # Nothing or func
    blend::B
    cull::C
    topology::T
    depth::D
    varyings::VY      # Nothing or NamedTuple of types, e.g. (normal=Vec3f, uv=Vec2f)
end

const Rasterizer = GraphicsPipeline

function GraphicsPipeline(;
        vertex, fragment,
        geometry=nothing, tess_control=nothing, tess_eval=nothing,
        blend::BlendMode=Opaque(), cull::CullFace=CullBack(),
        topology::Topology=TriangleList(), depth::DepthMode=DepthLess(),
        varyings=nothing)
    GraphicsPipeline(
        vertex, fragment, geometry, tess_control, tess_eval,
        blend, cull, topology, depth,
        varyings,
    )
end

TrianglePipeline(; vertex, fragment, kw...) = GraphicsPipeline(; vertex, fragment, kw...)
LinePipeline(; vertex, fragment, kw...) = GraphicsPipeline(; vertex, fragment, topology=LineList(), kw...)

# ── Lazy Compilation ──

const GFX_SHADER_CACHE = Dict{UInt64, LavaGfxShader}()

# Clear graphics shader/pipeline caches on vk_reset_device!
push!(RESET_CALLBACKS, function()
    empty!(GFX_SHADER_CACHE)
end)

"""Return (vert_shader::LavaGfxShader, compiled::CompiledGraphicsPipeline)."""
function ensure_compiled_with_shader!(pipeline::GraphicsPipeline,
                              vert_fn, frag_fn, tt_vertex, tt_fragment;
                              color_format=Vulkan.FORMAT_B8G8R8A8_SRGB,
                              descriptor_set_layout=nothing)
    vert = get_or_compile_gfx(vert_fn, tt_vertex, :vertex)
    compiled = ensure_compiled!(pipeline, vert_fn, frag_fn, tt_vertex, tt_fragment;
        color_format, descriptor_set_layout)
    return vert, compiled
end

function ensure_compiled!(pipeline::GraphicsPipeline, vert_fn, frag_fn, tt_vertex, tt_fragment;
                              color_format=Vulkan.FORMAT_B8G8R8A8_SRGB,
                              descriptor_set_layout=nothing)
    # Cache key includes type tuples — different arg types get different compiled pipelines
    cache_key = hash((vert_fn, frag_fn, tt_vertex, tt_fragment, color_format,
                       descriptor_set_layout !== nothing))
    cached = get(GFX_PIPELINE_CACHE, cache_key, nothing)
    cached !== nothing && return cached::CompiledGraphicsPipeline

    # Compile vertex shader
    vert = get_or_compile_gfx(vert_fn, tt_vertex, :vertex)

    # Compile fragment shader
    frag = get_or_compile_gfx(frag_fn, tt_fragment, :fragment)

    # Optional stages
    geom_spirv = nothing
    geom_config = nothing
    if pipeline.geometry !== nothing
        geom_fn, geom_cfg = pipeline.geometry
        geom = get_or_compile_gfx(geom_fn, tt_vertex, :geometry; config=geom_cfg)
        geom_spirv = geom.spirv_bytes
        geom_config = geom_cfg
    end

    tc_spirv = nothing
    te_spirv = nothing
    tess_cfg = nothing
    if pipeline.tess_control !== nothing
        tc_fn, tc_cfg = pipeline.tess_control
        tc = get_or_compile_gfx(tc_fn, tt_vertex, :tess_control; config=tc_cfg)
        tc_spirv = tc.spirv_bytes
        tess_cfg = tc_cfg
    end
    if pipeline.tess_eval !== nothing
        te = get_or_compile_gfx(pipeline.tess_eval, tt_vertex, :tess_eval;
            config=tess_cfg)
        te_spirv = te.spirv_bytes
    end

    compiled = create_graphics_pipeline(vert.spirv_bytes, frag.spirv_bytes;
        blend=pipeline.blend, cull=pipeline.cull,
        topology=pipeline.topology, depth=pipeline.depth,
        color_format=color_format,
        push_constant_size=max(vert.push_info.push_size, frag.push_info.push_size),
        geometry_spirv=geom_spirv,
        tess_ctrl_spirv=tc_spirv, tess_eval_spirv=te_spirv,
        tess_config=tess_cfg,
        descriptor_set_layout=descriptor_set_layout)

    GFX_PIPELINE_CACHE[cache_key] = compiled
    return compiled
end

function get_or_compile_gfx(@nospecialize(f), @nospecialize(tt), stage::Symbol; config=nothing)
    key = hash((f, tt, stage, config))
    cached = get(GFX_SHADER_CACHE, key, nothing)
    cached !== nothing && return cached
    shader = lava_compile_gfx_shader(f, tt; stage, config)
    GFX_SHADER_CACHE[key] = shader
    return shader
end

# ── Draw API ──

"""Convert args to device-side values (LavaArray → LavaDeviceArray, scalars pass through)."""
convert_args(args::Tuple) = map(arg -> arg isa LavaArray ? LavaDeviceArray(arg) : arg, args)
convert_args(::Tuple{}) = ()

"""
Build the full vertex output NamedTuple type from a varyings spec.
E.g. `(normal=Vec3f, uv=Vec2f)` → `@NamedTuple{position::Vec4f, normal::Vec3f, uv::Vec2f}`
"""
function varyings_to_output_type(varyings::NamedTuple)
    names = (:position, keys(varyings)...)
    types = (Vec4f, values(varyings)...)
    return NamedTuple{names, Tuple{types...}}
end

"""
Resolve vertex/fragment functions and type tuples, wrapping NamedTuple-returning
shaders with VertexWrapper/FragmentWrapper as needed.
Returns (vert_fn, vert_tt, frag_fn, frag_tt).
"""
function resolve_shader_pair(pipeline, vert_tt::Type, frag_tt::Type)
    if pipeline.varyings !== nothing
        vout = varyings_to_output_type(pipeline.varyings)
        wrapped_vert = VertexWrapper{typeof(pipeline.vertex)}()
        wrapped_frag = FragmentWrapper{typeof(pipeline.fragment), vout}()
        return wrapped_vert, vert_tt, wrapped_frag, frag_tt
    else
        # Legacy API: user calls gfx_output/gfx_input directly
        return pipeline.vertex, vert_tt, pipeline.fragment, frag_tt
    end
end

"""
    draw!(pipeline::GraphicsPipeline, target::RenderTarget, vertex_count;
          args=(), frag_args=(), instances=1,
          clear_color=(0f0, 0f0, 0f0, 1f0))

Draw using the given graphics pipeline to the render target.
Device-side type tuples are inferred automatically from args.
"""
function draw!(bq::BatchQueue, pipeline::GraphicsPipeline, target::WindowTarget, vertex_count::Integer;
               args=(), frag_args=(), instances::Integer=1,
               clear_color::Union{Nothing, NTuple{4, Float32}}=(0.0f0, 0.0f0, 0.0f0, 1.0f0))
    win = target.window

    converted_vert = convert_args(args)
    converted_frag = convert_args(frag_args)
    vert_tt = typeof(converted_vert)
    frag_tt = typeof(converted_frag)

    vert_fn, vert_tt, frag_fn, frag_tt = resolve_shader_pair(pipeline, vert_tt, frag_tt)

    vert_shader, compiled = ensure_compiled_with_shader!(pipeline,
        vert_fn, frag_fn, vert_tt, frag_tt;
        color_format=win.format)

    view = win.views[win.current_image_idx + 1]
    image = win.images[win.current_image_idx + 1]

    push_data = isempty(args) ? UInt8[] : pack_gfx_args(bq, args, vert_shader.push_info)

    vk_draw!(bq, compiled, view, image, win.extent, vertex_count;
        push_data, instances, clear_color)
end

function draw!(bq::BatchQueue, pipeline::GraphicsPipeline, target::OffscreenTarget, vertex_count::Integer;
               args=(), frag_args=(), instances::Integer=1,
               clear_color::Union{Nothing, NTuple{4, Float32}}=(0.0f0, 0.0f0, 0.0f0, 1.0f0),
               descriptor_set_layout=nothing,
               descriptor_set=nothing)
    fb = target.fb

    converted_vert = convert_args(args)
    converted_frag = convert_args(frag_args)
    vert_tt = typeof(converted_vert)
    frag_tt = typeof(converted_frag)

    vert_fn, vert_tt, frag_fn, frag_tt = resolve_shader_pair(pipeline, vert_tt, frag_tt)

    vert_shader, compiled = ensure_compiled_with_shader!(pipeline,
        vert_fn, frag_fn, vert_tt, frag_tt;
        color_format=fb.color_format, descriptor_set_layout)

    push_data = isempty(args) ? UInt8[] : pack_gfx_args(bq, args, vert_shader.push_info)

    vk_draw!(bq, compiled, fb.color_view, fb.color_image,
        Vulkan.Extent2D(UInt32(fb.width), UInt32(fb.height)),
        vertex_count;
        push_data, instances,
        depth_view=fb.depth_view,
        depth_image=fb.depth_image,
        clear_color, descriptor_set)
end

function pack_gfx_args(bq::BatchQueue, args, push_info::PushConstantInfo)
    push_info.push_size == 0 && return UInt8[]
    isempty(args) && return UInt8[]

    batch = ensure_active_batch!(bq)
    # LavaAdaptor is the single point that strips LavaArray → LavaDeviceArray
    # AND pins the original into batch.pinned.  Adapt.jl's recursion handles
    # wrapper structs / Broadcasted / NamedTuple, so any nested LavaArray
    # gets both its pointer strip and its pin in the same pass.
    adaptor = LavaAdaptor(batch)
    converted = map(a -> Adapt.adapt(adaptor, a), args)

    # Use the same arg buffer packing as compute: inline byval structs into
    # the arg buffer with self-referencing BDA pointers.
    offsets = [p.first for p in push_info.arg_layout]
    byval_sizes = push_info.byval_llvm_sizes

    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(bq, total_size)

    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       push_info.arg_buffer_size, byval_sizes, converted)

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    GC.@preserve push_data begin
        unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)
    end
    return push_data
end

# Empty-args variant
function pack_gfx_args(::BatchQueue, args, ::Nothing=nothing)
    isempty(args) && return UInt8[]
    error("pack_gfx_args requires push_info for non-empty args.")
end

# ── Blit: Fullscreen Display of GPU Buffer ──

# Built-in vertex shader for fullscreen triangle (3 vertices, no buffer)
function blit_vertex()
    vid = vertex_index() - Int32(1)  # 0-based for bit tricks
    # Fullscreen triangle: covers entire NDC [-1,1]×[-1,1]
    # vid=0: (-1,-1), vid=1: (3,-1), vid=2: (-1,3)
    x = Float32(Int32(vid & Int32(1)) * 4 - 1)
    y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
    set_position!(Vec4f(x, y, 0.0f0, 1.0f0))
    # Pass UV coordinates
    u = (x + 1.0f0) * 0.5f0
    v = (y + 1.0f0) * 0.5f0
    gfx_output(0, Vec2f(u, v))
    return nothing
end

# Convert any RGBA-like color to Vec4f for fragment output
to_vec4f(v::Vec4f) = v
to_vec4f(c) = Vec4f(c.r, c.g, c.b, c.alpha)

# Built-in fragment shader for blitting a GPU buffer to screen.
# Buffer is in Julia column-major layout: element [row, col] is at linear index col * height + row + 1.
# Screen coords: fx = column (x), fy = row (y, 0 = top in Vulkan).
function blit_fragment(buffer, width::Int32, height::Int32)
    fx = frag_coord_x()
    fy = frag_coord_y()
    ix = unsafe_trunc(Int32, fx)   # column (0-based)
    iy = unsafe_trunc(Int32, fy)   # row (0-based)
    # Column-major indexing: col * height + row + 1
    idx = ix * height + iy + Int32(1)
    pixel = to_vec4f(buffer[idx])
    gfx_output(0, pixel)
    return nothing
end

const BLIT_PIPELINE = Ref{Any}(nothing)

push!(RESET_CALLBACKS, function()
    BLIT_PIPELINE[] = nothing
end)

"""
    blit!(bq, target::RenderTarget, source::LavaArray; clear=true)

Display a GPU array on screen using a fullscreen blit.
The source array should contain RGBA Float32 pixels (or any 4-component type).
The buffer is read in Julia column-major order by the fragment shader.
"""
function blit!(bq::BatchQueue, target::RenderTarget, source::LavaArray;
               clear::Bool=true)
    if target isa WindowTarget
        win = target.window
        w, h = size(win)
        color_format = win.format
        view = win.views[win.current_image_idx + 1]
        image = win.images[win.current_image_idx + 1]
        extent = win.extent
    elseif target isa OffscreenTarget
        fb = target.fb
        w, h = fb.width, fb.height
        color_format = fb.color_format
        view = fb.color_view
        image = fb.color_image
        extent = Vulkan.Extent2D(UInt32(w), UInt32(h))
    else
        error("blit! only supports WindowTarget and OffscreenTarget")
    end

    # Create or reuse blit pipeline
    if BLIT_PIPELINE[] === nothing
        BLIT_PIPELINE[] = GraphicsPipeline(;
            vertex=blit_vertex,
            fragment=blit_fragment,
            blend=Opaque(),
            cull=NoCull(),
            depth=DepthOff(),
        )
    end

    pipeline = BLIT_PIPELINE[]

    # Fragment shader takes: (buffer::LavaDeviceArray, width::Int32, height::Int32)
    # Use the fragment shader's push_info for arg packing since vertex has no args.
    frag_args = (source, Int32(w), Int32(h))
    converted_frag = convert_args(frag_args)
    frag_tt = typeof(converted_frag)

    _, compiled = ensure_compiled_with_shader!(pipeline,
        pipeline.vertex, pipeline.fragment, Tuple{}, frag_tt;
        color_format=color_format)

    # Pack fragment args via the fragment shader's push_info
    frag_shader = get_or_compile_gfx(pipeline.fragment, frag_tt, :fragment)
    push_data = pack_gfx_args(bq, frag_args, frag_shader.push_info)

    clear_color = clear ? (0.0f0, 0.0f0, 0.0f0, 1.0f0) : nothing

    vk_draw!(bq, compiled, view, image, extent, 3;
        push_data, clear_color)
end

"""
    present_frame!(bq::BatchQueue, win::RenderWindow)

Submit recorded draw commands and present to screen.
"""
function present_frame!(bq::BatchQueue, win::RenderWindow)
    batch = bq.active_batch
    batch === nothing && error("present_frame! called without an active recording batch")
    cmd = batch.cmd_buf

    # Transition swapchain image to PRESENT_SRC before presenting
    image = win.images[win.current_image_idx + 1]
    transition_image!(cmd, image,
        Vulkan.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, Vulkan.IMAGE_LAYOUT_PRESENT_SRC_KHR,
        Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, Vulkan.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        Vulkan.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, Vulkan.AccessFlag(0))

    # End command buffer
    unwrap(Vulkan.end_command_buffer(cmd))

    fi = win.current_frame

    # ensure_active_batch! pre-assigned batch.signal_value = next_timeline + 1.
    # We must signal the timeline semaphore here so any `last_write` entries
    # set by dispatches in this batch can be observed as "complete".
    # Otherwise `wait_for_write(buf)` will hang forever on a value that
    # never arrives.
    bq.next_timeline += 1
    @assert batch.signal_value == bq.next_timeline "present_frame! signal desync"

    wait_infos = [
        # Wait for collected cross-queue deps:
        [Vulkan.SemaphoreSubmitInfo(s, v, UInt32(0); stage_mask=stage)
         for (s, v, stage) in batch.wait_semaphores]...,
        # Wait for swapchain image availability before color attachment writes:
        Vulkan.SemaphoreSubmitInfo(win.image_available[fi], UInt64(0), UInt32(0);
            stage_mask=Vulkan.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT),
    ]
    signal_infos = [
        # Signal timeline for in-Lava lifetime tracking:
        Vulkan.SemaphoreSubmitInfo(bq.timeline_sem, batch.signal_value, UInt32(0);
            stage_mask=Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
        # Signal render_finished for the subsequent present operation:
        Vulkan.SemaphoreSubmitInfo(win.render_finished[fi], UInt64(0), UInt32(0);
            stage_mask=Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
    ]
    cb_info = Vulkan.CommandBufferSubmitInfo(cmd, UInt32(0))
    submit_info = Vulkan.SubmitInfo2(wait_infos, [cb_info], signal_infos)
    queue_submit_2!(bq, [submit_info]; fence=win.in_flight[fi])

    # Store batch in window's per-frame slot — it will be reclaimed in
    # acquire_next_image! after the fence wait confirms GPU completion.
    # Do NOT push to free_batches here: the GPU is still using this command buffer.
    bq.active_batch = nothing
    win.frame_batches[fi] = batch
    empty!(batch.wait_semaphores)

    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    reset_arg_buffer_pool!(bq)

    # Present
    present!(win)
end
