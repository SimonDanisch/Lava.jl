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
struct GraphicsPipeline{V, F, G, TC, TE, B<:BlendMode, C<:CullFace, T<:Topology, D<:DepthMode}
    vertex::V
    fragment::F
    geometry::G       # Nothing or (func, GeometryConfig)
    tess_control::TC  # Nothing or (func, TessConfig)
    tess_eval::TE     # Nothing or func
    blend::B
    cull::C
    topology::T
    depth::D
    _compiled::Base.RefValue{Any}
end

const Rasterizer = GraphicsPipeline

function GraphicsPipeline(;
        vertex, fragment,
        geometry=nothing, tess_control=nothing, tess_eval=nothing,
        blend::BlendMode=Opaque(), cull::CullFace=CullBack(),
        topology::Topology=TriangleList(), depth::DepthMode=DepthLess())
    GraphicsPipeline(
        vertex, fragment, geometry, tess_control, tess_eval,
        blend, cull, topology, depth,
        Ref{Any}(nothing),
    )
end

TrianglePipeline(; vertex, fragment, kw...) = GraphicsPipeline(; vertex, fragment, kw...)
LinePipeline(; vertex, fragment, kw...) = GraphicsPipeline(; vertex, fragment, topology=LineList(), kw...)

# ── Lazy Compilation ──

const GFX_SHADER_CACHE = Dict{UInt64, LavaGfxShader}()

"""Return (vert_shader::LavaGfxShader, compiled::CompiledGraphicsPipeline)."""
function _ensure_compiled_with_shader!(pipeline::GraphicsPipeline, tt_vertex, tt_fragment;
                              color_format=Vulkan.FORMAT_B8G8R8A8_SRGB)
    vert = get_or_compile_gfx(pipeline.vertex, tt_vertex, :vertex)
    compiled = ensure_compiled!(pipeline, tt_vertex, tt_fragment; color_format)
    return vert, compiled
end

function ensure_compiled!(pipeline::GraphicsPipeline, tt_vertex, tt_fragment;
                              color_format=Vulkan.FORMAT_B8G8R8A8_SRGB)
    if pipeline._compiled[] !== nothing
        return pipeline._compiled[]::CompiledGraphicsPipeline
    end

    # Compile vertex shader
    vert = get_or_compile_gfx(pipeline.vertex, tt_vertex, :vertex)

    # Compile fragment shader
    frag = get_or_compile_gfx(pipeline.fragment, tt_fragment, :fragment)

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
        tess_config=tess_cfg)

    pipeline._compiled[] = compiled
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

"""
    draw!(pipeline::GraphicsPipeline, target::RenderTarget, vertex_count;
          args=(), instances=1, tt_vertex=Tuple{}, tt_fragment=Tuple{},
          clear_color=(0f0, 0f0, 0f0, 1f0))

Draw using the given graphics pipeline to the render target.
"""
function draw!(pipeline::GraphicsPipeline, target::WindowTarget, vertex_count::Integer;
               args=(), instances::Integer=1,
               tt_vertex::Type=Tuple{}, tt_fragment::Type=Tuple{},
               clear_color::Union{Nothing, NTuple{4, Float32}}=(0.0f0, 0.0f0, 0.0f0, 1.0f0))
    win = target.window
    vert_shader, compiled = _ensure_compiled_with_shader!(pipeline, tt_vertex, tt_fragment;
        color_format=win.format)

    view = win.views[win.current_image_idx + 1]
    image = win.images[win.current_image_idx + 1]

    push_data = isempty(args) ? UInt8[] : pack_gfx_args(args, vert_shader.push_info)

    vk_draw!(compiled, view, image, win.extent, vertex_count;
        push_data, instances, clear_color)
end

function draw!(pipeline::GraphicsPipeline, target::OffscreenTarget, vertex_count::Integer;
               args=(), instances::Integer=1,
               tt_vertex::Type=Tuple{}, tt_fragment::Type=Tuple{},
               clear_color::Union{Nothing, NTuple{4, Float32}}=(0.0f0, 0.0f0, 0.0f0, 1.0f0))
    fb = target.fb
    vert_shader, compiled = _ensure_compiled_with_shader!(pipeline, tt_vertex, tt_fragment;
        color_format=fb.color_format)

    push_data = isempty(args) ? UInt8[] : pack_gfx_args(args, vert_shader.push_info)

    vk_draw!(compiled, fb.color_view, fb.color_image,
        Vulkan.Extent2D(UInt32(fb.width), UInt32(fb.height)),
        vertex_count;
        push_data, instances,
        depth_view=fb.depth_view,
        depth_image=fb.depth_image,
        clear_color)
end

function pack_gfx_args(args, push_info::PushConstantInfo)
    push_info.push_size == 0 && return UInt8[]
    isempty(args) && return UInt8[]

    # Convert LavaArray -> LavaDeviceArray (same as KA's Adapt path)
    # so the byval struct {ptr, dims} gets inlined correctly into the arg buffer.
    converted = map(args) do arg
        if arg isa LavaArray
            return LavaDeviceArray(arg)
        else
            return arg
        end
    end

    # Use the same arg buffer packing as compute: inline byval structs into
    # the arg buffer with self-referencing BDA pointers.
    offsets = [p.first for p in push_info.arg_layout]
    byval_sizes = push_info.byval_llvm_sizes

    inline_extra = _compute_inline_extra_from_byval(byval_sizes)
    total_size = push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(total_size)
    _pack_args_direct!(arg_buf.mapped_ptr, arg_buf.address, offsets,
                       push_info.arg_buffer_size, byval_sizes, converted)

    # Keep data buffer references alive until vk_flush!()
    keep_data_alive!(args)

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    GC.@preserve push_data begin
        unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)
    end
    return push_data
end

# Backwards compat for no-args case
function pack_gfx_args(args)
    isempty(args) && return UInt8[]
    error("pack_gfx_args requires push_info for non-empty args. Use pack_gfx_args(args, push_info).")
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

# Built-in fragment shader for blitting a BDA buffer to screen
function blit_fragment(buffer_ptr::Ptr{Vec4f}, width::Int32, height::Int32)
    fx = frag_coord_x()
    fy = frag_coord_y()
    ix = unsafe_trunc(Int32, fx)
    iy = unsafe_trunc(Int32, fy)
    idx = iy * width + ix + Int32(1)
    pixel = unsafe_load(buffer_ptr, idx)
    gfx_output(0, pixel)
    return nothing
end

const BLIT_PIPELINE = Ref{Any}(nothing)

"""
    blit!(target::RenderTarget, source::LavaArray; clear=true)

Display a GPU array on screen using a fullscreen blit.
The source array should contain RGBA Float32 pixels.
"""
function blit!(target::RenderTarget, source::LavaArray;
               clear::Bool=true)
    win = if target isa WindowTarget
        target.window
    else
        error("blit! currently only supports WindowTarget")
    end

    w, h = size(win)

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

    # Pack args into arg buffer (same BDA pattern as compute)
    # Fragment shader takes: (buffer_ptr::Ptr{...}, width::Int32, height::Int32)
    # Arg buffer layout: [0:8] BDA ptr, [8:12] width i32, [12:16] height i32
    arg_buf = get_arg_buffer(24)  # 8 (ptr) + 4 (i32) + 4 (i32), padded to 8-align = 24
    ptr = arg_buf.mapped_ptr
    managed = source.buf[]
    addr = managed.address + UInt64(source.offset)
    unsafe_store!(Ptr{UInt64}(ptr), addr)
    unsafe_store!(Ptr{Int32}(ptr + 8), Int32(w))
    unsafe_store!(Ptr{Int32}(ptr + 12), Int32(h))
    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    GC.@preserve push_data begin
        unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)
    end

    compiled = ensure_compiled!(pipeline, Tuple{}, Tuple{Ptr{Vec4f}, Int32, Int32};
        color_format=win.format)

    view = win.views[win.current_image_idx + 1]
    image = win.images[win.current_image_idx + 1]

    clear_color = clear ? (0.0f0, 0.0f0, 0.0f0, 1.0f0) : nothing

    vk_draw!(compiled, view, image, win.extent, 3;
        push_data, clear_color)
end

"""
    present_frame!(win::RenderWindow)

Submit recorded draw commands and present to screen.
"""
function present_frame!(win::RenderWindow)
    ctx = vk_context()
    batch = ctx.active_batch
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

    # Submit with per-frame semaphores (using window's own fence, not batch fence)
    fi = win.current_frame
    submit_info = Vulkan.SubmitInfo(
        [win.image_available[fi]],
        [Vulkan.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
        [cmd],
        [win.render_finished[fi]],
    )
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence=win.in_flight[fi]))

    # Reset batch state (we handled submission ourselves via window fence)
    # Note: batch state must be reset BEFORE flush_deferred_frees! so GC finalizers
    # triggered during deferred free processing free immediately (GPU is idle).
    batch.recording = false
    batch.dispatch_count = 0
    batch.last_was_rt = false
    empty!(batch.data_refs)
    ctx.active_batch = nothing
    push!(ctx.free_batches, batch)

    flush_deferred_frees!()
    reset_arg_buffer_pool!()

    # Present
    present!(win)
end
