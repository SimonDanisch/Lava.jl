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
