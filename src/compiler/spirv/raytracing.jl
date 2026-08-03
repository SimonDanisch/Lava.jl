# Ray tracing SPIR-V emission for Lava.jl
#
# Extends the compute SPIR-V emitter to support RT shader stages:
# - RayGenerationKHR, ClosestHitKHR, MissKHR
# - OpTraceRayKHR instruction
# - RayPayloadKHR / IncomingRayPayloadKHR storage classes
# - TLAS descriptor binding (AccelerationStructureKHR)

# ── RT Builtin Mapping ──
# Maps __spirv_BuiltIn* names to BuiltIn decoration IDs for RT shaders.
# These are added to SPIRV_BUILTIN_MAP at module load time.

const SPIRV_RT_BUILTIN_MAP = Dict{String, UInt32}(
    # uvec3 builtins (raygen)
    "__spirv_BuiltInLaunchIdKHR"            => BuiltIn.LaunchIdKHR,
    "__spirv_BuiltInLaunchSizeKHR"          => BuiltIn.LaunchSizeKHR,
    # vec3 float builtins (hit/miss)
    "__spirv_BuiltInWorldRayOriginKHR"      => BuiltIn.WorldRayOriginKHR,
    "__spirv_BuiltInWorldRayDirectionKHR"   => BuiltIn.WorldRayDirectionKHR,
    "__spirv_BuiltInObjectRayOriginKHR"     => BuiltIn.ObjectRayOriginKHR,
    "__spirv_BuiltInObjectRayDirectionKHR"  => BuiltIn.ObjectRayDirectionKHR,
    # scalar float builtins
    "__spirv_BuiltInRayTminKHR"             => BuiltIn.RayTminKHR,
    "__spirv_BuiltInRayTmaxKHR"             => BuiltIn.RayTmaxKHR,
    # scalar uint builtins
    "__spirv_BuiltInHitKindKHR"             => BuiltIn.HitKindKHR,
    "__spirv_BuiltInInstanceCustomIndexKHR" => BuiltIn.InstanceCustomIndexKHR,
    "__spirv_BuiltInPrimitiveId"            => BuiltIn.PrimitiveId,
    "__spirv_BuiltInInstanceId"             => BuiltIn.InstanceId,
    "__spirv_BuiltInIncomingRayFlagsKHR"    => BuiltIn.IncomingRayFlagsKHR,
    # mat4x3 builtins (as [12 x float])
    "__spirv_BuiltInObjectToWorldKHR"       => BuiltIn.ObjectToWorldKHR,
    "__spirv_BuiltInWorldToObjectKHR"       => BuiltIn.WorldToObjectKHR,
)

# Register RT builtins in the global builtin map
merge!(SPIRV_BUILTIN_MAP, SPIRV_RT_BUILTIN_MAP)

# ── RT Shader Stage Info ──

struct RTShaderStageInfo
    exec_model::UInt32
    # Which payload storage class this stage uses
    # :raygen uses RayPayloadKHR (declares payload, reads after trace)
    # :closesthit/:miss use IncomingRayPayloadKHR (write results)
    payload_sc::UInt32
end

const RT_STAGE_INFO = Dict{Symbol, RTShaderStageInfo}(
    :raygen     => RTShaderStageInfo(ExecModel.RayGenerationKHR, SC.RayPayloadKHR),
    :closesthit => RTShaderStageInfo(ExecModel.ClosestHitKHR, SC.IncomingRayPayloadKHR),
    :miss       => RTShaderStageInfo(ExecModel.MissKHR, SC.IncomingRayPayloadKHR),
    :anyhit     => RTShaderStageInfo(ExecModel.AnyHitKHR, SC.IncomingRayPayloadKHR),
    :intersection => RTShaderStageInfo(ExecModel.IntersectionKHR, SC.IncomingRayPayloadKHR),
    :callable   => RTShaderStageInfo(ExecModel.CallableKHR, SC.IncomingCallableDataKHR),
)

"""
    emit_spirv_from_llvm_rt(llvm_mod, entry_name, stage; payload_type=:f32)

Emit SPIR-V from LLVM IR for a ray tracing shader stage.
Same skeleton as `emit_spirv_from_llvm` but:
1. Execution model = RT stage (RayGenerationKHR, ClosestHitKHR, MissKHR)
2. No LocalSize execution mode
3. RayTracingKHR capability + SPV_KHR_ray_tracing extension
4. RT builtin globals (stage-dependent)
5. Payload global variable (RayPayloadKHR or IncomingRayPayloadKHR)
6. TLAS descriptor variable (for raygen — AccelerationStructureKHR)
"""
function emit_spirv_from_llvm_rt(llvm_mod::LLVM.Module, entry_name::String,
                                   stage::Symbol; payload_type::Symbol=:f32)
    stage_info = get(RT_STAGE_INFO, stage, nothing)
    stage_info === nothing && error("Unknown RT shader stage: $stage")

    # Build pointee type map (same as compute)
    ptm = build_pointee_type_map(llvm_mod)

    # Create SPIR-V module
    spirv_mod = SPIRVModule()
    type_ctx = SPIRVTypeContext(spirv_mod, ptm)

    # Setup module header
    setup_memory_model!(spirv_mod; physical_storage_buffer=true)
    require_capability!(spirv_mod, Cap.Shader)
    require_capability!(spirv_mod, Cap.VariablePointers)
    require_capability!(spirv_mod, Cap.RayTracingKHR)
    require_extension!(spirv_mod, "SPV_KHR_variable_pointers")
    require_extension!(spirv_mod, "SPV_KHR_ray_tracing")
    # SER capability — opt-in, declared only when the device supports it.
    # Reading `ser_available` keeps the SPIR-V module valid on non-NVIDIA
    # hardware (where the capability would be a validation error).  Check
    # VK_CONTEXT_REF[] directly so emitter tests without a device don't
    # trigger lazy `init_vulkan!()`.
    if stage === :raygen
        ctx = VK_CONTEXT_REF[]
        if ctx !== nothing && ctx.ser_available
            require_capability!(spirv_mod, Cap.ShaderInvocationReorderNV)
            require_extension!(spirv_mod, "SPV_NV_shader_invocation_reorder")
        end
    end

    # Build struct pointer member type map
    build_struct_ptr_member_types!(type_ctx, llvm_mod)
    collect_module_types!(type_ctx, llvm_mod)

    # Create emitter state
    state = SPIRVEmitterState(spirv_mod, type_ctx)
    state.data_layout = LLVM.datalayout(llvm_mod)

    # Find entry function
    entry_fn = LLVM.functions(llvm_mod)[entry_name]

    # Pre-allocate SPIR-V function IDs for every function with a body so that
    # OpFunctionCall can forward-reference callees regardless of emission order.
    # Enables multi-OpFunction emission when force_inline_all=false (no effect in
    # the single-function case). Mirrors the compute emitter (emit_spirv_from_llvm).
    for fn in LLVM.functions(llvm_mod)
        isempty(LLVM.blocks(fn)) && continue
        get!(state.value_map, fn) do
            fresh_id!(spirv_mod)
        end
    end

    # Emit standard globals (push constants, builtins, constants)
    interface_ids = emit_globals!(state, llvm_mod)

    # ── RT-specific globals ──

    # Create payload variable (Float32 for now)
    payload_var_id = emit_rt_payload_global!(state, stage_info.payload_sc, payload_type)
    push!(interface_ids, payload_var_id)

    # TLAS descriptor variable.  Raygen always gets it (used by `traceRay`).
    # closesthit / miss get it too so they can fire ray queries for shadow
    # tracing — the descriptor binding is already visible to those stages
    # via the pipeline's `all_stage_flags`, so the only thing missing was
    # the SPIR-V OpVariable in the chit/miss modules. Declare RayQuery cap
    # for the non-raygen stages so OpRayQueryInitializeKHR is accepted.
    tlas_var_id = nothing
    if stage in (:raygen, :closesthit, :miss, :anyhit)
        tlas_var_id = emit_rt_tlas_descriptor!(state)
        push!(interface_ids, tlas_var_id)
        if stage in (:closesthit, :miss, :anyhit)
            setup_ray_query_capabilities!(spirv_mod)
        end
    end

    # For closesthit/anyhit: create hit attribute variable (vec2 barycentrics)
    hit_attrib_var_id = nothing
    if stage in (:closesthit, :anyhit, :intersection)
        hit_attrib_var_id = emit_rt_hit_attrib_global!(state)
        push!(interface_ids, hit_attrib_var_id)
    end

    # Store RT-specific IDs in state for the call handler
    state.rt_payload_var_id = payload_var_id
    state.rt_tlas_var_id = tlas_var_id
    state.rt_payload_type = payload_type
    state.rt_payload_storage_class = stage_info.payload_sc
    state.rt_hit_attrib_var_id = hit_attrib_var_id

    # Multi-OpFunction emission: walk the call graph from the entry and emit
    # every reachable helper as its own OpFunction (so the fat per-material chit
    # is many small functions instead of one giant inlined one — keeps compile
    # time sane). Only non-empty when force_inline_all=false. Mirrors compute.
    let reachable = collect_reachable_callees(entry_fn)
        for scc in strongly_connected_components(reachable)
            length(scc) > 1 && error("Mutual recursion is not supported in SPIR-V " *
                "multi-OpFunction emission: cycle through " * join(LLVM.name.(scc), " -> "))
        end
        for fn in reachable
            fn === entry_fn && continue
            isempty(LLVM.blocks(fn)) && continue
            emit_function!(state, fn; is_entry=false)
        end
    end

    # Emit entry function
    fn_ty = LLVM.function_type(entry_fn)
    n_params = length(collect(LLVM.parameters(fn_ty)))

    if n_params == 0
        func_id = emit_function!(state, entry_fn; is_entry=true)
    else
        func_id = emit_entry_wrapper!(state, entry_fn)
    end

    # Pick up any interface-list IDs registered during function emission
    # (currently: the Private OpVariable backing OpTypeHitObjectNV for SER).
    append!(interface_ids, state.entry_interface_ids)

    # Entry point — RT execution model, no LocalSize
    emit_entry_point!(spirv_mod, stage_info.exec_model, func_id, "main", interface_ids)
    # RT shaders have no execution modes (no LocalSize, no OriginUpperLeft, etc.)

    emit_name!(spirv_mod, func_id, entry_name)

    # Struct layout decorations
    decorate_psb_struct_layouts!(type_ctx, llvm_mod)

    return serialize(spirv_mod), spirv_mod.source_locations
end

"""
Create a payload global variable in the given storage class.
For raygen: RayPayloadKHR (location 0)
For closesthit/miss: IncomingRayPayloadKHR (location 0)
"""
function emit_rt_payload_global!(state::SPIRVEmitterState, storage_class::UInt32,
                                   payload_type::Symbol)
    mod = state.mod

    # Map payload type
    f32_ty = emit_type_float!(mod, UInt32(32))
    value_spirv_id = if payload_type == :f32
        f32_ty
    elseif payload_type == :f32_6
        # Array of 6 floats for multi-field payload
        len_id = emit_constant_u32!(mod, UInt32(6))
        emit_type_array!(mod, f32_ty, len_id)
    elseif payload_type == :f32_7
        # Array of 7 floats (adds slot 6 for gl_InstanceID alongside the
        # existing 6-slot closest-hit packet).
        len_id = emit_constant_u32!(mod, UInt32(7))
        emit_type_array!(mod, f32_ty, len_id)
    else
        error("Unsupported RT payload type: $payload_type")
    end

    # Pointer type for the payload storage class
    ptr_ty = map_pointer_type!(state.type_ctx, value_spirv_id, storage_class)

    # Create OpVariable
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, storage_class)

    # Decorate with Location 0
    emit_decorate!(mod, var_id, Dec.Location, UInt32(0))

    # Debug name
    if storage_class == SC.RayPayloadKHR
        emit_name!(mod, var_id, "ray_payload")
    else
        emit_name!(mod, var_id, "incoming_payload")
    end

    return var_id
end

"""
Create the TLAS descriptor variable for raygen shaders.
OpTypeAccelerationStructureKHR in UniformConstant storage class.
Decorated with DescriptorSet=0, Binding=0.
"""
function emit_rt_tlas_descriptor!(state::SPIRVEmitterState)
    mod = state.mod

    # OpTypeAccelerationStructureKHR
    accel_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeAccelerationStructureKHR, accel_ty)

    # OpTypePointer UniformConstant %accel_ty
    ptr_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypePointer, ptr_ty, SC.UniformConstant, accel_ty)

    # OpVariable
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.UniformConstant)

    # Decorations: DescriptorSet=0, Binding=0
    emit_decorate!(mod, var_id, Dec.DescriptorSet, UInt32(0))
    emit_decorate!(mod, var_id, Dec.Binding, UInt32(0))

    emit_name!(mod, var_id, "tlas")

    # Store the accel type ID for OpTraceRayKHR emission
    state.rt_accel_type_id = accel_ty

    return var_id
end

# ── OpTraceRayKHR Emission ──

"""
Emit OpTraceRayKHR from a call to lava_rt_trace_ray.
The call has 13 parameters (flags, mask, sbt_off, sbt_stride, miss_idx,
ox, oy, oz, tmin, dx, dy, dz, tmax).
The TLAS and payload are implicit (from RT globals).
"""
function emit_rt_trace_ray!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod

    # Get operand SPIR-V IDs
    args = UInt32[]
    for i in 1:LLVM.API.LLVMGetNumArgOperands(inst)
        arg = LLVM.operands(inst)[i]
        push!(args, get_value_id!(state, arg))
    end

    length(args) == 13 || error("lava_rt_trace_ray expects 13 arguments, got $(length(args))")

    # Unpack: flags, mask, sbt_off, sbt_stride, miss_idx, ox, oy, oz, tmin, dx, dy, dz, tmax
    flags_id      = args[1]
    mask_id       = args[2]
    sbt_off_id    = args[3]
    sbt_stride_id = args[4]
    miss_idx_id   = args[5]
    ox_id, oy_id, oz_id = args[6], args[7], args[8]
    tmin_id       = args[9]
    dx_id, dy_id, dz_id = args[10], args[11], args[12]
    tmax_id       = args[13]

    # Construct origin and direction vec3
    f32_ty = emit_type_float!(mod, UInt32(32))
    vec3_ty = emit_type_vector!(mod, f32_ty, UInt32(3))

    origin_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec3_ty, origin_id,
                        ox_id, oy_id, oz_id)

    dir_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec3_ty, dir_id,
                        dx_id, dy_id, dz_id)

    # Load TLAS
    tlas_var = state.rt_tlas_var_id
    tlas_var === nothing && error("OpTraceRayKHR requires TLAS variable (only valid in raygen)")

    accel_ty = state.rt_accel_type_id
    tlas_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLoad, accel_ty, tlas_id, tlas_var)

    # OpTraceRayKHR: void, 11 operands
    # %accel %flags %mask %sbt_off %sbt_stride %miss_idx %origin %tmin %dir %tmax %payload
    # The payload is the variable ID with RayPayloadKHR storage class
    payload_var = state.rt_payload_var_id

    # OpTraceRayKHR has no result — word count = 1 + 11 = 12
    word_count = UInt32(12)
    push!(mod.functions, (word_count << 16) | UInt32(Op.OpTraceRayKHR))
    push!(mod.functions, tlas_id)
    push!(mod.functions, flags_id)
    push!(mod.functions, mask_id)
    push!(mod.functions, sbt_off_id)
    push!(mod.functions, sbt_stride_id)
    push!(mod.functions, miss_idx_id)
    push!(mod.functions, origin_id)
    push!(mod.functions, tmin_id)
    push!(mod.functions, dir_id)
    push!(mod.functions, tmax_id)
    push!(mod.functions, payload_var)
end

# ── Payload Load/Store Emission ──

"""
Emit OpStore to the payload variable from lava_rt_payload_store_f32.
"""
function emit_rt_payload_store!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    val = LLVM.operands(inst)[1]
    val_id = get_value_id!(state, val)
    payload_var = state.rt_payload_var_id
    encode_instruction!(mod.functions, Op.OpStore, payload_var, val_id)
end

"""
Emit OpLoad from the payload variable for lava_rt_payload_load_f32.
"""
function emit_rt_payload_load!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    payload_var = state.rt_payload_var_id

    # Determine result type from payload_type
    result_ty = if state.rt_payload_type == :f32
        emit_type_float!(mod, UInt32(32))
    else
        error("Unsupported payload type for scalar load: $(state.rt_payload_type)")
    end

    result_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLoad, result_ty, result_id, payload_var)

    # Map the LLVM call result to this SPIR-V ID
    state.value_map[inst] = result_id
end

# ── Indexed Payload Load/Store (for array payloads) ──

"""
Emit OpAccessChain + OpStore for lava_rt_payload_store_f32_at(val, idx).
Payload must be an array type (e.g., :f32_6).
"""
function emit_rt_payload_store_at!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    val = LLVM.operands(inst)[1]
    idx = LLVM.operands(inst)[2]
    val_id = get_value_id!(state, val)
    idx_id = get_value_id!(state, idx)
    payload_var = state.rt_payload_var_id

    # Get pointer to element: OpAccessChain
    f32_ty = emit_type_float!(mod, UInt32(32))
    payload_sc = payload_sc_for_state(state)
    elem_ptr_ty = map_pointer_type!(state.type_ctx, f32_ty, payload_sc)

    ac_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpAccessChain, elem_ptr_ty, ac_id,
                        payload_var, idx_id)
    encode_instruction!(mod.functions, Op.OpStore, ac_id, val_id)
end

"""
Emit OpAccessChain + OpLoad for lava_rt_payload_load_f32_at(idx).
"""
function emit_rt_payload_load_at!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    idx = LLVM.operands(inst)[1]
    idx_id = get_value_id!(state, idx)
    payload_var = state.rt_payload_var_id

    f32_ty = emit_type_float!(mod, UInt32(32))
    payload_sc = payload_sc_for_state(state)
    elem_ptr_ty = map_pointer_type!(state.type_ctx, f32_ty, payload_sc)

    ac_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpAccessChain, elem_ptr_ty, ac_id,
                        payload_var, idx_id)

    result_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLoad, f32_ty, result_id, ac_id)

    state.value_map[inst] = result_id
end

"""Get the payload storage class from the emitter state."""
function payload_sc_for_state(state::SPIRVEmitterState)
    # `rt_payload_storage_class` was set when the payload OpVariable was
    # created in `emit_spirv_from_llvm_rt`, using `RT_STAGE_INFO[stage]`'s
    # `payload_sc` — `RayPayloadKHR` for raygen, `IncomingRayPayloadKHR`
    # for closesthit / miss / anyhit / intersection. Previously this was
    # inferred from `rt_tlas_var_id !== nothing` (only raygen had a TLAS),
    # but chit/miss/anyhit now also emit the TLAS descriptor so they can
    # fire inline ray queries — so the TLAS-as-proxy heuristic produced
    # the wrong storage class on chit/miss.
    state.rt_payload_storage_class
end

# ── Hit Attribute (Barycentric) Support ──

"""
Create a hit attribute global variable (vec2 float) in HitAttributeKHR storage class.
Only for closesthit/anyhit/intersection shaders.
Contains barycentric coordinates (u, v) for triangle intersections.
"""
function emit_rt_hit_attrib_global!(state::SPIRVEmitterState)
    mod = state.mod

    f32_ty = emit_type_float!(mod, UInt32(32))
    vec2_ty = emit_type_vector!(mod, f32_ty, UInt32(2))
    ptr_ty = map_pointer_type!(state.type_ctx, vec2_ty, SC.HitAttributeKHR)

    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.HitAttributeKHR)

    emit_name!(mod, var_id, "hit_attrib")

    return var_id
end

"""
Emit OpAccessChain + OpLoad for lava_rt_hit_attrib_load_f32_at(idx).
Reads a component from the HitAttributeKHR vec2 variable.
"""
function emit_rt_hit_attrib_load_at!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    idx = LLVM.operands(inst)[1]
    idx_id = get_value_id!(state, idx)

    hit_var = state.rt_hit_attrib_var_id
    hit_var === nothing && error("lava_rt_hit_attrib_load_f32_at requires HitAttributeKHR variable (only valid in closesthit/anyhit)")

    f32_ty = emit_type_float!(mod, UInt32(32))
    elem_ptr_ty = map_pointer_type!(state.type_ctx, f32_ty, SC.HitAttributeKHR)

    ac_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpAccessChain, elem_ptr_ty, ac_id,
                        hit_var, idx_id)

    result_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLoad, f32_ty, result_id, ac_id)

    state.value_map[inst] = result_id
end

# ── OpIgnoreIntersectionKHR / OpTerminateRayKHR Emission ──

"""
Emit OpIgnoreIntersectionKHR — block terminator in any-hit shaders.
Rejects the current intersection and continues traversal.
"""
function emit_rt_ignore_intersection!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    encode_instruction!(state.mod.functions, Op.OpIgnoreIntersectionKHR)
    state.rt_block_terminated = true
end

"""
Emit OpTerminateRayKHR — block terminator in any-hit shaders.
Accepts the current hit and stops traversal immediately.
"""
function emit_rt_terminate_ray!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    encode_instruction!(state.mod.functions, Op.OpTerminateRayKHR)
    state.rt_block_terminated = true
end

# ─────────────────────────────────────────────────────────────────────────────
# SER (SPV_NV_shader_invocation_reorder) emission
# ─────────────────────────────────────────────────────────────────────────────

# Read `vk_context().ser_available` defensively: the emitter runs in contexts
# where vk_context may not be initialised (unit tests without a device).  When
# the flag is unavailable we treat SER as unsupported and emit the implicit
# OpTraceRayKHR fallback so the SPIR-V stays valid without the NV capability.
function _ser_available_for_emit()
    # "No device initialised" is the documented case above, and it is the only
    # one that may answer `false`: a context that EXISTS but errors on a field
    # read is a bug, and treating it as "SER unsupported" would silently emit the
    # slower fallback forever.
    VK_CONTEXT_REF[] === nothing && return false
    return (VK_CONTEXT_REF[]::VkContext).ser_available
end

"""
Emit OpTypeHitObjectNV once per module.  Cached in `state.rt_hit_object_type_id`.
"""
function emit_rt_hit_object_type!(state::SPIRVEmitterState)
    state.rt_hit_object_type_id !== nothing && return state.rt_hit_object_type_id
    mod = state.mod
    id = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeHitObjectNV, id)
    state.rt_hit_object_type_id = id
    return id
end

"""
Get or lazily create the implicit Private-storage HitObject variable that all
SER intrinsics in this raygen share.  Mirrors the `hitObjectNV hit;` slot in
GLSL — exactly one per shader.
"""
function get_or_create_hit_object_var!(state::SPIRVEmitterState)
    state.rt_hit_object_var_id !== nothing && return state.rt_hit_object_var_id
    mod = state.mod
    ho_ty = emit_rt_hit_object_type!(state)
    # OpTypePointer Private OpTypeHitObjectNV
    ptr_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypePointer, ptr_ty,
                        SC.Private, ho_ty)
    # OpVariable Private
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Private)
    emit_name!(mod, var_id, "hit_object")
    state.rt_hit_object_var_id = var_id
    # SPIR-V 1.4+ requires every global OpVariable the entry point uses to
    # be listed in OpEntryPoint's interface list — including Private-storage
    # ones like our HitObject slot.
    push!(state.entry_interface_ids, var_id)
    return var_id
end

"""
Emit OpHitObjectTraceRayNV from a call to lava_rt_hit_object_trace_ray.
Same 13 operands as OpTraceRayKHR; result is written into the implicit
HitObject variable (no closest-hit shader is invoked yet).

On devices without VK_NV_ray_tracing_invocation_reorder the SER opcode and
its OpTypeHitObjectNV would fail spirv-val (the capability is not declared),
so we degrade to a regular OpTraceRayKHR.  The companion reorder/execute
ops then become no-ops; together they reproduce the implicit-trace path
the SER pattern emulates on hardware that does support reordering.
"""
function emit_rt_hit_object_trace_ray!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    if !_ser_available_for_emit()
        return emit_rt_trace_ray!(state, inst)
    end
    mod = state.mod

    args = UInt32[]
    for i in 1:LLVM.API.LLVMGetNumArgOperands(inst)
        push!(args, get_value_id!(state, LLVM.operands(inst)[i]))
    end
    length(args) == 13 || error("lava_rt_hit_object_trace_ray expects 13 arguments, got $(length(args))")

    flags_id      = args[1]
    mask_id       = args[2]
    sbt_off_id    = args[3]
    sbt_stride_id = args[4]
    miss_idx_id   = args[5]
    ox_id, oy_id, oz_id = args[6], args[7], args[8]
    tmin_id       = args[9]
    dx_id, dy_id, dz_id = args[10], args[11], args[12]
    tmax_id       = args[13]

    f32_ty = emit_type_float!(mod, UInt32(32))
    vec3_ty = emit_type_vector!(mod, f32_ty, UInt32(3))

    origin_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec3_ty, origin_id,
                        ox_id, oy_id, oz_id)

    dir_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec3_ty, dir_id,
                        dx_id, dy_id, dz_id)

    tlas_var = state.rt_tlas_var_id
    tlas_var === nothing && error("OpHitObjectTraceRayNV requires TLAS variable (only valid in raygen)")
    accel_ty = state.rt_accel_type_id
    tlas_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLoad, accel_ty, tlas_id, tlas_var)

    payload_var = state.rt_payload_var_id
    ho_var = get_or_create_hit_object_var!(state)

    # OpHitObjectTraceRayNV %hit_object_var %accel %flags %mask %sbt_off %sbt_stride
    #                       %miss_idx %origin %tmin %dir %tmax %payload
    # 12 operands + opcode = word count 13
    word_count = UInt32(13)
    push!(mod.functions, (word_count << 16) | UInt32(Op.OpHitObjectTraceRayNV))
    push!(mod.functions, ho_var)
    push!(mod.functions, tlas_id)
    push!(mod.functions, flags_id)
    push!(mod.functions, mask_id)
    push!(mod.functions, sbt_off_id)
    push!(mod.functions, sbt_stride_id)
    push!(mod.functions, miss_idx_id)
    push!(mod.functions, origin_id)
    push!(mod.functions, tmin_id)
    push!(mod.functions, dir_id)
    push!(mod.functions, tmax_id)
    push!(mod.functions, payload_var)
end

"""
Emit OpReorderThreadWithHitObjectNV using the implicit HitObject.

On devices without SER, the preceding `lava_rt_hit_object_trace_ray` was
already lowered to a full OpTraceRayKHR (which invoked the chit inline);
nothing remains to reorder, so this is a no-op.
"""
function emit_rt_reorder_thread!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    _ser_available_for_emit() || return
    mod = state.mod
    ho_var = get_or_create_hit_object_var!(state)
    # OpReorderThreadWithHitObjectNV %hit_object_var
    # 1 operand + opcode = word count 2
    word_count = UInt32(2)
    push!(mod.functions, (word_count << 16) | UInt32(Op.OpReorderThreadWithHitObjectNV))
    push!(mod.functions, ho_var)
end

"""
Emit OpHitObjectExecuteShaderNV — invokes the closest-hit / miss shader for
the recorded HitObject, using the current ray payload.

On devices without SER, the closest-hit / miss shader was already invoked
inline by the fallback OpTraceRayKHR in `emit_rt_hit_object_trace_ray!`, so
this is a no-op.
"""
function emit_rt_hit_object_execute_shader!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    _ser_available_for_emit() || return
    mod = state.mod
    ho_var = get_or_create_hit_object_var!(state)
    payload_var = state.rt_payload_var_id
    payload_var === nothing && error("OpHitObjectExecuteShaderNV requires a payload variable")
    # OpHitObjectExecuteShaderNV %hit_object_var %payload
    # 2 operands + opcode = word count 3
    word_count = UInt32(3)
    push!(mod.functions, (word_count << 16) | UInt32(Op.OpHitObjectExecuteShaderNV))
    push!(mod.functions, ho_var)
    push!(mod.functions, payload_var)
end
