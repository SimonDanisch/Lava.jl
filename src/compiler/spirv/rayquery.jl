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
    return var_id
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
