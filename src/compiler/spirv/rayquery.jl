# VK_KHR_ray_query SPIR-V emission.
#
# This file holds:
#   * Module-level setup helpers (capabilities, extensions, type
#     registration, compute-pipeline TLAS descriptor variable).
#   * Per-opcode emission for OpRayQuery*KHR, appended in tasks A4..A5.

"""
Require the SPIR-V capabilities and extensions for inline ray queries.
Idempotent.
"""
function setup_ray_query_capabilities!(spirv_mod::SPIRVModule)
    require_capability!(spirv_mod, Cap.RayQueryKHR)
    require_capability!(spirv_mod, Cap.RayTracingKHR)
    require_extension!(spirv_mod, "SPV_KHR_ray_query")
    require_extension!(spirv_mod, "SPV_KHR_ray_tracing")
    return nothing
end

"""
Emit OpTypeAccelerationStructureKHR once per module. Caches the result id in
`state.rt_accel_type_id` (reusing the existing RT-shader field).
"""
function emit_acceleration_structure_type!(state::SPIRVEmitterState)
    state.rt_accel_type_id !== nothing && return state.rt_accel_type_id
    mod = state.mod
    id = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeAccelerationStructureKHR, id)
    state.rt_accel_type_id = id
    return id
end

"""
Emit OpTypeRayQueryKHR once per module. Caches the result id in
`state.ray_query_type_id`.
"""
function emit_ray_query_type!(state::SPIRVEmitterState)
    state.ray_query_type_id !== nothing && return state.ray_query_type_id
    mod = state.mod
    id = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypeRayQueryKHR, id)
    state.ray_query_type_id = id
    return id
end

"""
Emit the compute-pipeline TLAS descriptor variable at set=0, binding=0.
Stores the variable id in `state.rt_tlas_var_id` and pushes it onto
`state.entry_interface_ids` so emit_entry_point! sees it.
"""
function emit_compute_tlas_descriptor!(state::SPIRVEmitterState)
    mod = state.mod
    accel_ty = emit_acceleration_structure_type!(state)

    # Eagerly declare OpTypeRayQueryKHR so it appears in every ray-query
    # compute module even before the first get_or_create_ray_query_var! call.
    emit_ray_query_type!(state)

    # OpTypePointer UniformConstant %accel_ty
    ptr_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypePointer, ptr_ty,
                        SC.UniformConstant, accel_ty)

    # OpVariable
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.UniformConstant)

    # Decorations
    emit_decorate!(mod, var_id, Dec.DescriptorSet, UInt32(0))
    emit_decorate!(mod, var_id, Dec.Binding, UInt32(0))

    emit_name!(mod, var_id, "tlas")

    state.rt_tlas_var_id = var_id
    push!(state.entry_interface_ids, var_id)

    # Note: the per-function rayQuery OpVariable is allocated by
    # prescan_function_for_rayquery! at the top of every emit_function!,
    # so it lands in entry_function_locals of whichever function actually uses it
    # (entry kernel or a @noinline helper).

    return var_id
end

"""
Pre-scan a function's body for ray-query intrinsic calls. If any are present,
eagerly allocates the per-function rayQuery OpVariable so it lands in the
function's preamble (`state.entry_function_locals`) BEFORE any block emits. Without
this, the first lazy allocation happens deep inside a non-first block and
the OpVariable would be referenced before its definition.

Safe to call on every function; no-op if no ray queries are used. Also asserts
that no parameter has type OpTypeRayQueryKHR — ray queries must be allocated
inside the function that uses them per SPIR-V spec.
"""
function prescan_function_for_rayquery!(state::SPIRVEmitterState, fn::LLVM.Function)
    isempty(LLVM.blocks(fn)) && return nothing
    for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
        inst isa LLVM.CallInst || continue
        callee = LLVM.called_operand(inst)
        callee isa LLVM.Function || continue
        name = LLVM.name(callee)
        if startswith(name, "lava_ray_query_")
            get_or_create_ray_query_var!(state)
            return nothing
        end
    end
    return nothing
end

"""
Allocate a Function-storage-class OpVariable of type OpTypeRayQueryKHR on first
use within the entry function. Subsequent calls return the cached id.
The OpVariable words are buffered in `state.entry_function_locals` and injected
into the entry block's preamble section by `emit_function!`.
"""
function get_or_create_ray_query_var!(state::SPIRVEmitterState)
    state.current_ray_query_var !== nothing && return state.current_ray_query_var
    mod = state.mod
    rq_ty = emit_ray_query_type!(state)

    # OpTypePointer Function %rq_ty
    ptr_ty = fresh_id!(mod)
    encode_instruction!(mod.types_constants, Op.OpTypePointer, ptr_ty,
                        SC.Function, rq_ty)

    # OpVariable -- goes into the function-local preamble buffer, not global_vars
    var_id = fresh_id!(mod)
    encode_instruction!(state.entry_function_locals, Op.OpVariable, ptr_ty, var_id,
                        SC.Function)

    state.current_ray_query_var = var_id
    return var_id
end

"""
Lower a `lava_ray_query_init` (scalar form) call to OpRayQueryInitializeKHR.
Arguments: flags, mask, ox, oy, oz, tmin, dx, dy, dz, tmax (10 scalars).
"""
function emit_ray_query_init!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    args = LLVM.operands(inst)

    flags_id = get_value_id!(state, args[1])
    mask_id  = get_value_id!(state, args[2])
    ox_id    = get_value_id!(state, args[3])
    oy_id    = get_value_id!(state, args[4])
    oz_id    = get_value_id!(state, args[5])
    tmin_id  = get_value_id!(state, args[6])
    dx_id    = get_value_id!(state, args[7])
    dy_id    = get_value_id!(state, args[8])
    dz_id    = get_value_id!(state, args[9])
    tmax_id  = get_value_id!(state, args[10])

    # Construct origin and direction vec3 (same pattern as emit_rt_trace_ray!)
    f32_ty  = emit_type_float!(mod, UInt32(32))
    vec3_ty = emit_type_vector!(mod, f32_ty, UInt32(3))

    origin_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec3_ty, origin_id,
                        ox_id, oy_id, oz_id)

    dir_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeConstruct, vec3_ty, dir_id,
                        dx_id, dy_id, dz_id)

    # Load the TLAS acceleration structure
    accel_ty = emit_acceleration_structure_type!(state)
    tlas_var = state.rt_tlas_var_id
    tlas_var === nothing && error("OpRayQueryInitializeKHR requires a TLAS variable (enable_ray_query must be true)")

    tlas_loaded_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLoad, accel_ty, tlas_loaded_id, tlas_var)

    # Get (or create) the per-function OpVariable for the ray query
    query_var = get_or_create_ray_query_var!(state)

    # OpRayQueryInitializeKHR has no result:
    # OpRayQueryInitializeKHR %query %accel %flags %mask %origin %tmin %dir %tmax
    # word count = 1 (opcode) + 8 operands = 9
    word_count = UInt32(9)
    push!(mod.functions, (word_count << 16) | UInt32(Op.OpRayQueryInitializeKHR))
    push!(mod.functions, query_var)
    push!(mod.functions, tlas_loaded_id)
    push!(mod.functions, flags_id)
    push!(mod.functions, mask_id)
    push!(mod.functions, origin_id)
    push!(mod.functions, tmin_id)
    push!(mod.functions, dir_id)
    push!(mod.functions, tmax_id)
    return nothing
end

# Helper: emit a no-result ray-query op that takes only the implicit query variable.
# word count = 1 (opcode word) + 1 (query var) = 2.
function emit_rq_no_arg!(state::SPIRVEmitterState, opcode::UInt16)
    mod = state.mod
    qvar = get_or_create_ray_query_var!(state)
    push!(mod.functions, (UInt32(2) << 16) | UInt32(opcode))
    push!(mod.functions, qvar)
    return nothing
end

function emit_ray_query_proceed!(state::SPIRVEmitterState, inst::LLVM.CallInst)
    mod = state.mod
    qvar = get_or_create_ray_query_var!(state)
    bool_ty = emit_type_bool!(mod)
    res_id = fresh_id!(mod)
    # OpRayQueryProceedKHR %result_type %result_id %query -- word count = 4
    push!(mod.functions, (UInt32(4) << 16) | UInt32(Op.OpRayQueryProceedKHR))
    push!(mod.functions, bool_ty)
    push!(mod.functions, res_id)
    push!(mod.functions, qvar)
    state.value_map[inst] = res_id
    return nothing
end

emit_ray_query_confirm!(state::SPIRVEmitterState, ::LLVM.CallInst) =
    emit_rq_no_arg!(state, Op.OpRayQueryConfirmIntersectionKHR)

emit_ray_query_terminate!(state::SPIRVEmitterState, ::LLVM.CallInst) =
    emit_rq_no_arg!(state, Op.OpRayQueryTerminateKHR)

# Helper: emit a GetIntersection op that returns a scalar result.
# Instruction layout: opcode %result_type %result_id %query %committed -- word count = 5.
function emit_rq_get_scalar!(state::SPIRVEmitterState, inst::LLVM.CallInst,
                              opcode::UInt16, result_ty::UInt32)
    mod = state.mod
    qvar = get_or_create_ray_query_var!(state)
    committed_id = get_value_id!(state, LLVM.operands(inst)[1])
    res_id = fresh_id!(mod)
    push!(mod.functions, (UInt32(5) << 16) | UInt32(opcode))
    push!(mod.functions, result_ty)
    push!(mod.functions, res_id)
    push!(mod.functions, qvar)
    push!(mod.functions, committed_id)
    state.value_map[inst] = res_id
    return nothing
end

emit_ray_query_get_type!(state, inst) =
    emit_rq_get_scalar!(state, inst, Op.OpRayQueryGetIntersectionTypeKHR,
                        emit_type_int!(state.mod, UInt32(32), UInt32(0)))

emit_ray_query_get_t!(state, inst) =
    emit_rq_get_scalar!(state, inst, Op.OpRayQueryGetIntersectionTKHR,
                        emit_type_float!(state.mod, UInt32(32)))

emit_ray_query_get_instance_id!(state, inst) =
    emit_rq_get_scalar!(state, inst, Op.OpRayQueryGetIntersectionInstanceIdKHR,
                        emit_type_int!(state.mod, UInt32(32), UInt32(0)))

emit_ray_query_get_instance_custom_index!(state, inst) =
    emit_rq_get_scalar!(state, inst, Op.OpRayQueryGetIntersectionInstanceCustomIndexKHR,
                        emit_type_int!(state.mod, UInt32(32), UInt32(0)))

emit_ray_query_get_primitive_index!(state, inst) =
    emit_rq_get_scalar!(state, inst, Op.OpRayQueryGetIntersectionPrimitiveIndexKHR,
                        emit_type_int!(state.mod, UInt32(32), UInt32(0)))

# Helper: emit OpRayQueryGetIntersectionBarycentricsKHR and extract one component.
# The op returns a vec2 float; we extract component 0 (x) or 1 (y) via OpCompositeExtract.
function emit_rq_barycentric_component!(state::SPIRVEmitterState, inst::LLVM.CallInst,
                                        component::UInt32)
    mod = state.mod
    qvar = get_or_create_ray_query_var!(state)
    committed_id = get_value_id!(state, LLVM.operands(inst)[1])
    f32_ty  = emit_type_float!(mod, UInt32(32))
    vec2_ty = emit_type_vector!(mod, f32_ty, UInt32(2))
    vec_id  = fresh_id!(mod)
    # OpRayQueryGetIntersectionBarycentricsKHR %vec2 %vec_id %query %committed -- word count = 5
    push!(mod.functions, (UInt32(5) << 16) | UInt32(Op.OpRayQueryGetIntersectionBarycentricsKHR))
    push!(mod.functions, vec2_ty)
    push!(mod.functions, vec_id)
    push!(mod.functions, qvar)
    push!(mod.functions, committed_id)
    # OpCompositeExtract %f32 %res_id %vec_id <literal index>
    res_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpCompositeExtract, f32_ty, res_id, vec_id, component)
    state.value_map[inst] = res_id
    return nothing
end

emit_ray_query_get_barycentrics_x!(state, inst) =
    emit_rq_barycentric_component!(state, inst, UInt32(0))

emit_ray_query_get_barycentrics_y!(state, inst) =
    emit_rq_barycentric_component!(state, inst, UInt32(1))
