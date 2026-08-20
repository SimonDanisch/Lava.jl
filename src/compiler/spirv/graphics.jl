# Graphics shader SPIR-V emission for Lava.jl
#
# Extends the compute SPIR-V emitter to support all 5 graphics shader stages:
# - Vertex, Fragment, Geometry, TessellationControl, TessellationEvaluation
#
# Follows the same pattern as raytracing.jl:
# 1. Same LLVM passes + BDA entry wrapper
# 2. Stage-specific execution model, capabilities, execution modes
# 3. Stage-specific builtin globals (VertexIndex, FragCoord, TessCoord, etc.)
# 4. User-defined I/O variables with Location decorations
# 5. Intrinsic call interception (set_position, output_vec4, emit_vertex, etc.)

# ── Graphics Builtin Mapping ──
# Maps __spirv_BuiltIn* names to BuiltIn decoration IDs for graphics shaders.

const SPIRV_GFX_BUILTIN_MAP = Dict{String, UInt32}(
    # Vertex input
    "__spirv_BuiltInVertexIndex"       => BuiltIn.VertexIndex,
    "__spirv_BuiltInInstanceIndex"     => BuiltIn.InstanceIndex,
    # Fragment input
    "__spirv_BuiltInFragCoord"         => BuiltIn.FragCoord,
    "__spirv_BuiltInFrontFacing"       => BuiltIn.FrontFacing,
    # Geometry/Tessellation
    "__spirv_BuiltInInvocationId"      => BuiltIn.InvocationId,
    "__spirv_BuiltInPrimitiveId"       => BuiltIn.PrimitiveId,
    # Tessellation
    "__spirv_BuiltInTessCoord"         => BuiltIn.TessCoord,
)

# Register graphics builtins in the global builtin map
merge!(SPIRV_BUILTIN_MAP, SPIRV_GFX_BUILTIN_MAP)

# ── Graphics Shader Stage Info ──

struct GfxShaderStageInfo
    exec_model::UInt32
    stage_name::String
end

const GFX_STAGE_INFO = Dict{Symbol, GfxShaderStageInfo}(
    :vertex       => GfxShaderStageInfo(ExecModel.Vertex, "vertex"),
    :fragment     => GfxShaderStageInfo(ExecModel.Fragment, "fragment"),
    :geometry     => GfxShaderStageInfo(ExecModel.Geometry, "geometry"),
    :tess_control => GfxShaderStageInfo(ExecModel.TessellationControl, "tess_control"),
    :tess_eval    => GfxShaderStageInfo(ExecModel.TessellationEvaluation, "tess_eval"),
)

# ── Graphics I/O Variable Tracking ──

"""
Track I/O variables created during emission for graphics shaders.
Maps location → SPIR-V variable ID.
"""
mutable struct GfxIOState
    # Output variables: location → (var_id, type: :f32, :vec2, :vec3, :vec4)
    output_vars::Dict{UInt32, Tuple{UInt32, Symbol}}
    # Input variables: location → (var_id, type)
    input_vars::Dict{UInt32, Tuple{UInt32, Symbol}}
    # Geometry shader arrayed input variables: location → (var_id, type)
    # These are array-typed (e.g. OpTypeArray(vec4, 4) for lines_adjacency)
    geom_input_vars::Dict{UInt32, Tuple{UInt32, Symbol}}
    # Geometry shader gl_in input (PerVertex struct array) variable ID
    geom_position_input_var_id::Union{Nothing, UInt32}
    # Number of input vertices for geometry shader (determined by input topology)
    geom_input_vertex_count::Int
    # Position output variable (BuiltIn Position)
    position_var_id::Union{Nothing, UInt32}
    # PointSize output variable
    point_size_var_id::Union{Nothing, UInt32}
    # TessLevelOuter/Inner output variables
    tess_outer_var_id::Union{Nothing, UInt32}
    tess_inner_var_id::Union{Nothing, UInt32}
    # Texture sampler variables: binding → var_id
    sampler_vars::Dict{UInt32, UInt32}
    # Combined image sampler type ID (cached)
    sampler_type_id::Union{Nothing, UInt32}
end

GfxIOState() = GfxIOState(
    Dict{UInt32, Tuple{UInt32, Symbol}}(),
    Dict{UInt32, Tuple{UInt32, Symbol}}(),
    Dict{UInt32, Tuple{UInt32, Symbol}}(),
    nothing,
    0,
    nothing, nothing, nothing, nothing,
    Dict{UInt32, UInt32}(),
    nothing,
)

# ── Main Emission Function ──

"""
    emit_spirv_from_llvm_gfx(llvm_mod, entry_name, stage; config=nothing)

Emit SPIR-V from LLVM IR for a graphics shader stage.
Same skeleton as `emit_spirv_from_llvm` (compute) and `emit_spirv_from_llvm_rt` but:
1. Execution model = graphics stage (Vertex, Fragment, Geometry, etc.)
2. No LocalSize execution mode (except geometry/tessellation have stage-specific modes)
3. Geometry capability for geometry stage, Tessellation for tess stages
4. Graphics builtin globals (VertexIndex, FragCoord, TessCoord, etc.)
5. User-defined I/O variables with Location decorations
"""
function emit_spirv_from_llvm_gfx(llvm_mod::LLVM.Module, entry_name::String,
                                     stage::Symbol; config=nothing)
    stage_info = get(GFX_STAGE_INFO, stage, nothing)
    stage_info === nothing && error("Unknown graphics shader stage: $stage")

    # Build pointee type map (same as compute)
    ptm = build_pointee_type_map(llvm_mod)

    # Create SPIR-V module
    spirv_mod = SPIRVModule()
    type_ctx = SPIRVTypeContext(spirv_mod, ptm)

    # Setup module header
    setup_memory_model!(spirv_mod; physical_storage_buffer=true)
    require_capability!(spirv_mod, Cap.Shader)
    require_capability!(spirv_mod, Cap.VariablePointers)
    require_extension!(spirv_mod, "SPV_KHR_variable_pointers")

    # Stage-specific capabilities
    if stage == :geometry
        require_capability!(spirv_mod, Cap.Geometry)
    elseif stage in (:tess_control, :tess_eval)
        require_capability!(spirv_mod, Cap.Tessellation)
    end

    # Build struct pointer member type map
    build_struct_ptr_member_types!(type_ctx, llvm_mod)
    collect_module_types!(type_ctx, llvm_mod)

    # Create emitter state
    state = SPIRVEmitterState(spirv_mod, type_ctx)
    state.data_layout = LLVM.datalayout(llvm_mod)

    # Graphics I/O state — stored in a module-level ref during emission
    gfx_io = GfxIOState()

    # Set geometry shader input vertex count from config
    if stage == :geometry && config !== nothing
        gfx_io.geom_input_vertex_count = geometry_input_vertex_count(config.input_topology)
    end

    # Find entry function
    entry_fn = LLVM.functions(llvm_mod)[entry_name]

    # Emit standard globals (push constants, builtins, constants)
    interface_ids = emit_globals!(state, llvm_mod)

    # Pre-scan for graphics intrinsic calls to determine needed I/O variables.
    # This must happen before function emission so variables exist when calls are emitted.
    gfx_prescan_io!(state, gfx_io, entry_fn, stage)

    # Add I/O variables to interface list
    for (_, (var_id, _)) in gfx_io.output_vars
        push!(interface_ids, var_id)
    end
    for (_, (var_id, _)) in gfx_io.input_vars
        push!(interface_ids, var_id)
    end
    for (_, (var_id, _)) in gfx_io.geom_input_vars
        push!(interface_ids, var_id)
    end
    gfx_io.geom_position_input_var_id !== nothing && push!(interface_ids, gfx_io.geom_position_input_var_id)
    gfx_io.position_var_id !== nothing && push!(interface_ids, gfx_io.position_var_id)
    gfx_io.point_size_var_id !== nothing && push!(interface_ids, gfx_io.point_size_var_id)
    gfx_io.tess_outer_var_id !== nothing && push!(interface_ids, gfx_io.tess_outer_var_id)
    gfx_io.tess_inner_var_id !== nothing && push!(interface_ids, gfx_io.tess_inner_var_id)
    for (_, var_id) in gfx_io.sampler_vars
        push!(interface_ids, var_id)
    end

    # Store gfx_io in a place the call handler can access it.
    # We use the emitter state's rt fields repurposed (or we can pass it differently).
    # For now, store as a tagged value in value_map using a sentinel.
    state.gfx_io = gfx_io

    # Emit function
    fn_ty = LLVM.function_type(entry_fn)
    n_params = length(collect(LLVM.parameters(fn_ty)))

    if n_params == 0
        func_id = emit_function!(state, entry_fn; is_entry=true)
    else
        func_id = emit_entry_wrapper!(state, entry_fn)
    end

    # Entry point
    emit_entry_point!(spirv_mod, stage_info.exec_model, func_id, "main", interface_ids)

    # Execution modes (stage-dependent)
    emit_gfx_execution_modes!(spirv_mod, func_id, stage, config)

    emit_name!(spirv_mod, func_id, entry_name)

    # Struct layout decorations
    decorate_psb_struct_layouts!(type_ctx, llvm_mod)

    return serialize(spirv_mod), spirv_mod.source_locations
end

# ── Execution Modes ──

function emit_gfx_execution_modes!(mod::SPIRVModule, func_id::UInt32,
                                      stage::Symbol, config)
    if stage == :fragment
        emit_execution_mode!(mod, func_id, ExecMode.OriginUpperLeft)
    elseif stage == :geometry && config !== nothing
        # Input topology
        input_mode = geometry_input_mode(config.input_topology)
        emit_execution_mode!(mod, func_id, input_mode)
        # Output topology
        output_mode = geometry_output_mode(config.output_topology)
        emit_execution_mode!(mod, func_id, output_mode)
        # Max vertices
        emit_execution_mode!(mod, func_id, ExecMode.OutputVertices, UInt32(config.max_vertices))
        # Invocations (required by Vulkan even when == 1)
        emit_execution_mode!(mod, func_id, ExecMode.Invocations, UInt32(config.invocations))
    elseif stage == :tess_control && config !== nothing
        emit_execution_mode!(mod, func_id, ExecMode.OutputVertices, UInt32(config.patch_vertices))
    elseif stage == :tess_eval && config !== nothing
        # Domain
        domain_mode = tess_domain_mode(config.domain)
        emit_execution_mode!(mod, func_id, domain_mode)
        # Spacing
        spacing_mode = tess_spacing_mode(config.spacing)
        emit_execution_mode!(mod, func_id, spacing_mode)
        # Winding
        winding_mode = tess_winding_mode(config.winding)
        emit_execution_mode!(mod, func_id, winding_mode)
    end
    # Vertex stage has no execution modes
end

geometry_input_vertex_count(::PointList)          = 1
geometry_input_vertex_count(::LineList)           = 2
geometry_input_vertex_count(::TriangleList)       = 3
geometry_input_vertex_count(::LineListAdjacency)  = 4
geometry_input_vertex_count(::LineStripAdjacency) = 4

geometry_input_mode(::PointList)          = ExecMode.InputPoints
geometry_input_mode(::LineList)           = ExecMode.InputLines
geometry_input_mode(::TriangleList)       = ExecMode.Triangles
# Both adjacency topologies present the geometry stage with the same four-vertex
# primitive; list vs strip is purely how the index buffer is walked, so SPIR-V
# has one execution mode for the pair.
geometry_input_mode(::LineListAdjacency)  = ExecMode.InputLinesAdjacency
geometry_input_mode(::LineStripAdjacency) = ExecMode.InputLinesAdjacency

geometry_output_mode(::PointList)     = ExecMode.OutputPoints
geometry_output_mode(::LineStrip)     = ExecMode.OutputLineStrip
geometry_output_mode(::TriangleStrip) = ExecMode.OutputTriangleStrip

function tess_domain_mode(d::TessDomain)
    d isa TessTriangles ? ExecMode.Triangles :
    d isa TessQuads     ? ExecMode.Quads :
    d isa TessIsolines  ? ExecMode.Isolines :
    error("Unsupported tessellation domain: $d")
end

function tess_spacing_mode(s::TessSpacing)
    s isa EqualSpacing          ? ExecMode.SpacingEqual :
    s isa FractionalEvenSpacing ? ExecMode.SpacingFractionalEven :
    s isa FractionalOddSpacing  ? ExecMode.SpacingFractionalOdd :
    error("Unsupported tessellation spacing: $s")
end

function tess_winding_mode(w::TessWinding)
    w isa WindingCW  ? ExecMode.VertexOrderCw :
    w isa WindingCCW ? ExecMode.VertexOrderCcw :
    error("Unsupported tessellation winding: $w")
end

# ── I/O Variable Pre-Scan ──
# Scans the LLVM IR for graphics intrinsic calls to determine what I/O variables need
# to be created before function emission.

function gfx_prescan_io!(state::SPIRVEmitterState, gfx_io::GfxIOState,
                           entry_fn::LLVM.Function, stage::Symbol)
    mod = state.mod

    # Walk all instructions in the function to find graphics intrinsic calls
    for bb in LLVM.blocks(entry_fn)
        for inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            called = LLVM.called_operand(inst)
            called isa LLVM.Function || continue
            fn_name = LLVM.name(called)

            if fn_name == "_lava_gfx_set_position"
                gfx_ensure_position_var!(state, gfx_io, stage)
            elseif fn_name == "_lava_gfx_set_point_size"
                gfx_ensure_point_size_var!(state, gfx_io, stage)
            elseif startswith(fn_name, "_lava_gfx_output_")
                loc = extract_constant_u32(LLVM.operands(inst)[1])
                is_flat = contains(fn_name, "_flat_")
                iotype = gfx_output_type_from_name(fn_name)
                gfx_ensure_output_var!(state, gfx_io, loc, iotype, stage; flat=is_flat)
            elseif startswith(fn_name, "_lava_gfx_input_")
                loc = extract_constant_u32(LLVM.operands(inst)[1])
                is_flat = contains(fn_name, "_flat_")
                iotype = gfx_input_type_from_name(fn_name)
                gfx_ensure_input_var!(state, gfx_io, loc, iotype, stage; flat=is_flat)
            elseif fn_name == "_lava_gfx_set_tess_level_outer"
                gfx_ensure_tess_outer_var!(state, gfx_io)
            elseif fn_name == "_lava_gfx_set_tess_level_inner"
                gfx_ensure_tess_inner_var!(state, gfx_io)
            elseif fn_name == "_lava_gfx_sample_2d"
                binding = extract_constant_u32(LLVM.operands(inst)[1])
                gfx_ensure_sampler_var!(state, gfx_io, binding)
            elseif fn_name == "_lava_gfx_emit_vertex" || fn_name == "_lava_gfx_end_primitive"
                # No I/O variables needed, just capability (already added)
            elseif fn_name == "_lava_geom_input_position"
                gfx_ensure_geom_position_input_var!(state, gfx_io)
            elseif startswith(fn_name, "_lava_geom_input_")
                loc = extract_constant_u32(LLVM.operands(inst)[1])
                iotype = geom_input_type_from_name(fn_name)
                gfx_ensure_geom_input_var!(state, gfx_io, loc, iotype)
            end
        end
    end
end

function extract_constant_u32(val::LLVM.Value)
    if val isa LLVM.ConstantInt
        return UInt32(convert(Int, val))
    end
    error("Expected constant integer for graphics I/O location, got: $val")
end

function gfx_output_type_from_name(name::String)
    endswith(name, "_vec4") && return :vec4
    endswith(name, "_vec3") && return :vec3
    endswith(name, "_vec2") && return :vec2
    endswith(name, "_f32")  && return :f32
    error("Unknown graphics output type: $name")
end

function gfx_input_type_from_name(name::String)
    endswith(name, "_vec4") && return :vec4
    endswith(name, "_vec3") && return :vec3
    endswith(name, "_vec2") && return :vec2
    endswith(name, "_f32")  && return :f32
    error("Unknown graphics input type: $name")
end

# ── Create I/O Variables ──

function gfx_spirv_type_for_io(mod::SPIRVModule, iotype::Symbol)
    f32_ty = emit_type_float!(mod, UInt32(32))
    if iotype == :f32
        return f32_ty
    elseif iotype == :vec2
        return emit_type_vector!(mod, f32_ty, UInt32(2))
    elseif iotype == :vec3
        return emit_type_vector!(mod, f32_ty, UInt32(3))
    elseif iotype == :vec4
        return emit_type_vector!(mod, f32_ty, UInt32(4))
    else
        error("Unknown I/O type: $iotype")
    end
end

function gfx_ensure_position_var!(state::SPIRVEmitterState, gfx_io::GfxIOState, stage::Symbol)
    gfx_io.position_var_id !== nothing && return
    mod = state.mod
    f32_ty = emit_type_float!(mod, UInt32(32))
    vec4_ty = emit_type_vector!(mod, f32_ty, UInt32(4))
    ptr_ty = map_pointer_type!(state.type_ctx, vec4_ty, SC.Output)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Output)
    emit_decorate!(mod, var_id, Dec.BuiltIn, BuiltIn.Position)
    emit_name!(mod, var_id, "gl_Position")
    gfx_io.position_var_id = var_id
end

function gfx_ensure_point_size_var!(state::SPIRVEmitterState, gfx_io::GfxIOState, stage::Symbol)
    gfx_io.point_size_var_id !== nothing && return
    mod = state.mod
    f32_ty = emit_type_float!(mod, UInt32(32))
    ptr_ty = map_pointer_type!(state.type_ctx, f32_ty, SC.Output)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Output)
    emit_decorate!(mod, var_id, Dec.BuiltIn, BuiltIn.PointSize)
    emit_name!(mod, var_id, "gl_PointSize")
    gfx_io.point_size_var_id = var_id
end

function gfx_ensure_output_var!(state::SPIRVEmitterState, gfx_io::GfxIOState,
                                   location::UInt32, iotype::Symbol, stage::Symbol;
                                   flat::Bool=false)
    haskey(gfx_io.output_vars, location) && return
    mod = state.mod
    value_ty = gfx_spirv_type_for_io(mod, iotype)
    ptr_ty = map_pointer_type!(state.type_ctx, value_ty, SC.Output)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Output)
    emit_decorate!(mod, var_id, Dec.Location, location)
    flat && emit_decorate!(mod, var_id, Dec.Flat)
    emit_name!(mod, var_id, flat ? "out_flat_loc$(location)" : "out_loc$(location)")
    gfx_io.output_vars[location] = (var_id, iotype)
end

function gfx_ensure_input_var!(state::SPIRVEmitterState, gfx_io::GfxIOState,
                                  location::UInt32, iotype::Symbol, stage::Symbol;
                                  flat::Bool=false)
    haskey(gfx_io.input_vars, location) && return
    mod = state.mod
    value_ty = gfx_spirv_type_for_io(mod, iotype)
    ptr_ty = map_pointer_type!(state.type_ctx, value_ty, SC.Input)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Input)
    emit_decorate!(mod, var_id, Dec.Location, location)
    flat && emit_decorate!(mod, var_id, Dec.Flat)
    emit_name!(mod, var_id, flat ? "in_flat_loc$(location)" : "in_loc$(location)")
    gfx_io.input_vars[location] = (var_id, iotype)
end

# ── Geometry Shader Arrayed Input Variables ──
# In geometry shaders, vertex shader outputs become arrayed inputs:
#   in vec4 g_color[N]  →  OpTypeArray(vec4, N) with Location decoration
# where N = number of input vertices (e.g. 4 for lines_adjacency).

function geom_input_type_from_name(name::String)
    endswith(name, "_vec4") && return :vec4
    endswith(name, "_vec3") && return :vec3
    endswith(name, "_vec2") && return :vec2
    endswith(name, "_f32")  && return :f32
    endswith(name, "_i32")  && return :i32
    error("Unknown geometry input type: $name")
end

function gfx_spirv_type_for_geom_io(mod::SPIRVModule, iotype::Symbol)
    if iotype == :i32
        return emit_type_int!(mod, UInt32(32), UInt32(1))  # signed i32
    else
        return gfx_spirv_type_for_io(mod, iotype)
    end
end

"""Create an array-typed Input variable for geometry shader: `in T var[N]`."""
function gfx_ensure_geom_input_var!(state::SPIRVEmitterState, gfx_io::GfxIOState,
                                       location::UInt32, iotype::Symbol)
    haskey(gfx_io.geom_input_vars, location) && return
    n = gfx_io.geom_input_vertex_count
    n > 0 || error("Geometry shader input vertex count not set (is config missing?)")
    mod = state.mod
    elem_ty = gfx_spirv_type_for_geom_io(mod, iotype)
    len_id = emit_constant_u32!(mod, UInt32(n))
    arr_ty = emit_type_array!(mod, elem_ty, len_id)
    ptr_ty = map_pointer_type!(state.type_ctx, arr_ty, SC.Input)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Input)
    emit_decorate!(mod, var_id, Dec.Location, location)
    emit_name!(mod, var_id, "geom_in_loc$(location)")
    gfx_io.geom_input_vars[location] = (var_id, iotype)
end

"""
Create gl_in — the built-in PerVertex struct array input for geometry shaders.
In SPIR-V this is:
  OpTypeStruct { vec4 Position }  (gl_PerVertex)
  OpTypeArray(gl_PerVertex, N)
  OpVariable Input
with BuiltIn Position on member 0 and Block decoration on the struct.
"""
function gfx_ensure_geom_position_input_var!(state::SPIRVEmitterState, gfx_io::GfxIOState)
    gfx_io.geom_position_input_var_id !== nothing && return
    n = gfx_io.geom_input_vertex_count
    n > 0 || error("Geometry shader input vertex count not set (is config missing?)")
    mod = state.mod
    f32_ty = emit_type_float!(mod, UInt32(32))
    vec4_ty = emit_type_vector!(mod, f32_ty, UInt32(4))
    # gl_PerVertex struct with just Position
    struct_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeStruct, struct_ty, vec4_ty)
    emit_decorate!(mod, struct_ty, Dec.Block)
    emit_member_decorate!(mod, struct_ty, UInt32(0), Dec.BuiltIn, BuiltIn.Position)
    emit_name!(mod, struct_ty, "gl_PerVertex")
    # Array of gl_PerVertex
    len_id = emit_constant_u32!(mod, UInt32(n))
    arr_ty = emit_type_array!(mod, struct_ty, len_id)
    ptr_ty = map_pointer_type!(state.type_ctx, arr_ty, SC.Input)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Input)
    emit_name!(mod, var_id, "gl_in")
    gfx_io.geom_position_input_var_id = var_id
end

function gfx_ensure_tess_outer_var!(state::SPIRVEmitterState, gfx_io::GfxIOState)
    gfx_io.tess_outer_var_id !== nothing && return
    mod = state.mod
    f32_ty = emit_type_float!(mod, UInt32(32))
    len_id = emit_constant_u32!(mod, UInt32(4))
    arr_ty = emit_type_array!(mod, f32_ty, len_id)
    emit_decorate!(mod, arr_ty, Dec.ArrayStride, UInt32(4))
    ptr_ty = map_pointer_type!(state.type_ctx, arr_ty, SC.Output)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Output)
    emit_decorate!(mod, var_id, Dec.BuiltIn, BuiltIn.TessLevelOuter)
    emit_decorate!(mod, var_id, Dec.Patch)
    emit_name!(mod, var_id, "gl_TessLevelOuter")
    gfx_io.tess_outer_var_id = var_id
end

function gfx_ensure_tess_inner_var!(state::SPIRVEmitterState, gfx_io::GfxIOState)
    gfx_io.tess_inner_var_id !== nothing && return
    mod = state.mod
    f32_ty = emit_type_float!(mod, UInt32(32))
    len_id = emit_constant_u32!(mod, UInt32(2))
    arr_ty = emit_type_array!(mod, f32_ty, len_id)
    emit_decorate!(mod, arr_ty, Dec.ArrayStride, UInt32(4))
    ptr_ty = map_pointer_type!(state.type_ctx, arr_ty, SC.Output)
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Output)
    emit_decorate!(mod, var_id, Dec.BuiltIn, BuiltIn.TessLevelInner)
    emit_decorate!(mod, var_id, Dec.Patch)
    emit_name!(mod, var_id, "gl_TessLevelInner")
    gfx_io.tess_inner_var_id = var_id
end

function gfx_ensure_sampler_var!(state::SPIRVEmitterState, gfx_io::GfxIOState, binding::UInt32)
    haskey(gfx_io.sampler_vars, binding) && return
    mod = state.mod

    # Create combined image sampler type (cached)
    if gfx_io.sampler_type_id === nothing
        f32_ty = emit_type_float!(mod, UInt32(32))
        # OpTypeImage: sampled_type=float, dim=2D, depth=0, arrayed=0, MS=0, sampled=1, format=Unknown
        image_ty = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeImage, image_ty, f32_ty,
            UInt32(1), # Dim2D
            UInt32(0), # depth=no
            UInt32(0), # arrayed=no
            UInt32(0), # MS=no
            UInt32(1), # sampled=yes (used with sampler)
            UInt32(0), # format=Unknown
        )
        # OpTypeSampledImage
        sampled_image_ty = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeSampledImage, sampled_image_ty, image_ty)
        gfx_io.sampler_type_id = sampled_image_ty
    end

    # Pointer type: UniformConstant → SampledImage
    ptr_ty = map_pointer_type!(state.type_ctx, gfx_io.sampler_type_id, SC.UniformConstant)

    # OpVariable
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.UniformConstant)
    emit_decorate!(mod, var_id, Dec.DescriptorSet, UInt32(0))
    emit_decorate!(mod, var_id, Dec.Binding, binding)
    emit_name!(mod, var_id, "sampler_$(binding)")

    gfx_io.sampler_vars[binding] = var_id
end

# ── Graphics Intrinsic Call Emission ──
# Called from emit.jl's call handler when a _lava_gfx_* function is encountered.

function emit_gfx_set_position!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    var_id = gfx_io.position_var_id
    var_id === nothing && error("_lava_gfx_set_position called but no Position variable created")

    # Get x, y, z, w arguments
    x_id = get_value_id!(state, LLVM.operands(inst)[1])
    y_id = get_value_id!(state, LLVM.operands(inst)[2])
    z_id = get_value_id!(state, LLVM.operands(inst)[3])
    w_id = get_value_id!(state, LLVM.operands(inst)[4])

    # Construct vec4
    f32_ty = emit_type_float!(mod, UInt32(32))
    vec4_ty = emit_type_vector!(mod, f32_ty, UInt32(4))
    vec_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec4_ty, vec_id,
                        x_id, y_id, z_id, w_id)

    # Store to gl_Position
    encode_instruction!(mod.functions, Op.OpStore, var_id, vec_id)
end

function emit_gfx_set_point_size!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    var_id = gfx_io.point_size_var_id
    val_id = get_value_id!(state, LLVM.operands(inst)[1])
    encode_instruction!(mod.functions, Op.OpStore, var_id, val_id)
end

function emit_gfx_output_vec4!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    loc = extract_constant_u32(LLVM.operands(inst)[1])
    var_id, _ = gfx_io.output_vars[loc]

    x_id = get_value_id!(state, LLVM.operands(inst)[2])
    y_id = get_value_id!(state, LLVM.operands(inst)[3])
    z_id = get_value_id!(state, LLVM.operands(inst)[4])
    w_id = get_value_id!(state, LLVM.operands(inst)[5])

    f32_ty = emit_type_float!(mod, UInt32(32))
    vec4_ty = emit_type_vector!(mod, f32_ty, UInt32(4))
    vec_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec4_ty, vec_id,
                        x_id, y_id, z_id, w_id)
    encode_instruction!(mod.functions, Op.OpStore, var_id, vec_id)
end

function emit_gfx_output_vec3!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    loc = extract_constant_u32(LLVM.operands(inst)[1])
    var_id, _ = gfx_io.output_vars[loc]

    x_id = get_value_id!(state, LLVM.operands(inst)[2])
    y_id = get_value_id!(state, LLVM.operands(inst)[3])
    z_id = get_value_id!(state, LLVM.operands(inst)[4])

    f32_ty = emit_type_float!(mod, UInt32(32))
    vec3_ty = emit_type_vector!(mod, f32_ty, UInt32(3))
    vec_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec3_ty, vec_id,
                        x_id, y_id, z_id)
    encode_instruction!(mod.functions, Op.OpStore, var_id, vec_id)
end

function emit_gfx_output_vec2!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    loc = extract_constant_u32(LLVM.operands(inst)[1])
    var_id, _ = gfx_io.output_vars[loc]

    x_id = get_value_id!(state, LLVM.operands(inst)[2])
    y_id = get_value_id!(state, LLVM.operands(inst)[3])

    f32_ty = emit_type_float!(mod, UInt32(32))
    vec2_ty = emit_type_vector!(mod, f32_ty, UInt32(2))
    vec_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec2_ty, vec_id,
                        x_id, y_id)
    encode_instruction!(mod.functions, Op.OpStore, var_id, vec_id)
end

function emit_gfx_output_f32!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    loc = extract_constant_u32(LLVM.operands(inst)[1])
    var_id, _ = gfx_io.output_vars[loc]
    val_id = get_value_id!(state, LLVM.operands(inst)[2])
    encode_instruction!(mod.functions, Op.OpStore, var_id, val_id)
end

function emit_gfx_input!(state::SPIRVEmitterState, inst::LLVM.CallInst, iotype::Symbol)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    loc = extract_constant_u32(LLVM.operands(inst)[1])
    var_id, _ = gfx_io.input_vars[loc]

    value_ty = gfx_spirv_type_for_io(mod, iotype)

    if iotype == :f32
        # Direct load
        result_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, value_ty, result_id, var_id)
        state.value_map[inst] = result_id
    else
        # Load vec, then extract component
        vec_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, value_ty, vec_id, var_id)
        comp = LLVM.operands(inst)[2]  # component index
        comp_id = get_value_id!(state, comp)
        f32_ty = emit_type_float!(mod, UInt32(32))
        result_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpVectorExtractDynamic, f32_ty, result_id,
                            vec_id, comp_id)
        state.value_map[inst] = result_id
    end
end

# ── Geometry Shader Arrayed Input Emission ──
# These emit OpAccessChain into the arrayed Input variable, then OpLoad + OpCompositeExtract.
# SPIR-V: %ptr = OpAccessChain %ptr_elem_ty %arr_var %idx
#         %val = OpLoad %elem_ty %ptr
#         %comp = OpVectorExtractDynamic %f32 %val %comp_idx  (for vec types)

"""
Emit read from arrayed geometry input: `in T var[vertex_idx]`.
For vec types: extracts a single float component.
For f32/i32: loads the scalar directly.
"""
function emit_geom_input!(state::SPIRVEmitterState, inst::LLVM.CallInst, iotype::Symbol)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    loc = extract_constant_u32(LLVM.operands(inst)[1])
    var_id, _ = gfx_io.geom_input_vars[loc]

    elem_ty = gfx_spirv_type_for_geom_io(mod, iotype)
    elem_ptr_ty = map_pointer_type!(state.type_ctx, elem_ty, SC.Input)

    # OpAccessChain into the array: var[vertex_idx]
    vidx_id = get_value_id!(state, LLVM.operands(inst)[2])
    ac_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpAccessChain, elem_ptr_ty, ac_id, var_id, vidx_id)

    if iotype == :f32
        # Direct load → float result
        result_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, elem_ty, result_id, ac_id)
        state.value_map[inst] = result_id
    elseif iotype == :i32
        # Direct load → int32 result
        result_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, elem_ty, result_id, ac_id)
        state.value_map[inst] = result_id
    else
        # Load vec, extract component
        vec_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, elem_ty, vec_id, ac_id)
        comp_id = get_value_id!(state, LLVM.operands(inst)[3])
        f32_ty = emit_type_float!(mod, UInt32(32))
        result_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpVectorExtractDynamic, f32_ty, result_id,
                            vec_id, comp_id)
        state.value_map[inst] = result_id
    end
end

"""
Emit read from gl_in[vertex_idx].gl_Position[component].
SPIR-V: %ptr = OpAccessChain %ptr_vec4 %gl_in %vidx %zero  (member 0 = Position)
        %vec = OpLoad %vec4 %ptr
        %val = OpVectorExtractDynamic %f32 %vec %comp
"""
function emit_geom_input_position!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    gl_in_var = gfx_io.geom_position_input_var_id
    gl_in_var !== nothing || error("gl_in not created — did you call geom_input_position without geometry shader config?")

    f32_ty = emit_type_float!(mod, UInt32(32))
    vec4_ty = emit_type_vector!(mod, f32_ty, UInt32(4))
    vec4_ptr_ty = map_pointer_type!(state.type_ctx, vec4_ty, SC.Input)

    vidx_id = get_value_id!(state, LLVM.operands(inst)[1])
    comp_id = get_value_id!(state, LLVM.operands(inst)[2])

    # Member 0 of gl_PerVertex = Position
    zero_id = emit_constant_u32!(mod, UInt32(0))

    # OpAccessChain: gl_in[vidx].Position (member 0)
    ac_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpAccessChain, vec4_ptr_ty, ac_id,
                        gl_in_var, vidx_id, zero_id)

    # Load vec4
    vec_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLoad, vec4_ty, vec_id, ac_id)

    # Extract component
    result_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpVectorExtractDynamic, f32_ty, result_id,
                        vec_id, comp_id)
    state.value_map[inst] = result_id
end

function emit_gfx_derivative!(state::SPIRVEmitterState, inst::LLVM.CallInst, opcode::UInt16)
    # OpDPdx/OpDPdy: result_type result_id operand
    mod = state.mod
    operand = LLVM.operands(inst)[1]
    operand_id = state.value_map[operand]
    f32_ty = emit_type_float!(mod, UInt32(32))
    result_id = fresh_id!(mod)
    encode_instruction!(mod.functions, opcode, f32_ty, result_id, operand_id)
    state.value_map[inst] = result_id
end

function emit_gfx_emit_vertex!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    encode_instruction!(state.mod.functions, Op.OpEmitVertex)
end

function emit_gfx_end_primitive!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    encode_instruction!(state.mod.functions, Op.OpEndPrimitive)
end

function emit_gfx_set_tess_level!(state::SPIRVEmitterState, inst::LLVM.CallInst, is_outer::Bool)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState
    var_id = is_outer ? gfx_io.tess_outer_var_id : gfx_io.tess_inner_var_id

    idx_id = get_value_id!(state, LLVM.operands(inst)[1])
    val_id = get_value_id!(state, LLVM.operands(inst)[2])

    f32_ty = emit_type_float!(mod, UInt32(32))
    elem_ptr_ty = map_pointer_type!(state.type_ctx, f32_ty, SC.Output)

    ac_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpAccessChain, elem_ptr_ty, ac_id,
                        var_id, idx_id)
    encode_instruction!(mod.functions, Op.OpStore, ac_id, val_id)
end

function emit_gfx_sample_2d!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    gfx_io = state.gfx_io::GfxIOState

    binding = extract_constant_u32(LLVM.operands(inst)[1])
    u_id = get_value_id!(state, LLVM.operands(inst)[2])
    v_id = get_value_id!(state, LLVM.operands(inst)[3])
    comp_id = get_value_id!(state, LLVM.operands(inst)[4])

    f32_ty = emit_type_float!(mod, UInt32(32))

    # Cache texture samples by (binding, u_id, v_id) to avoid redundant
    # OpImageSampleImplicitLod when tex[Vec2f(u,v)] samples all 4 components.
    cache_key = (:tex_sample, binding, u_id, v_id)
    sample_id = get(mod.constant_cache, cache_key, UInt32(0))

    if sample_id == UInt32(0)
        sampler_var = gfx_io.sampler_vars[binding]
        sampled_image_ty = gfx_io.sampler_type_id

        sampler_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpLoad, sampled_image_ty, sampler_id, sampler_var)

        vec2_ty = emit_type_vector!(mod, f32_ty, UInt32(2))
        coord_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec2_ty, coord_id, u_id, v_id)

        vec4_ty = emit_type_vector!(mod, f32_ty, UInt32(4))
        sample_id = fresh_id!(mod)
        encode_instruction!(mod.functions, Op.OpImageSampleImplicitLod, vec4_ty, sample_id,
                            sampler_id, coord_id)

        mod.constant_cache[cache_key] = sample_id
    end

    # Extract requested component
    result_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpVectorExtractDynamic, f32_ty, result_id,
                        sample_id, comp_id)

    state.value_map[inst] = result_id
end
