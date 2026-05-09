# Tier 1: Ray generation shader emission tests
# Tests OpTraceRayKHR, payload, launch builtins, BDA args

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

@testset "Raygen Shader" begin

    @testset "basic raygen" begin
        function basic_raygen(output)
            ix = Lava.lava_rt_launch_id_x()
            iy = Lava.lava_rt_launch_id_y()
            sx = Lava.lava_rt_launch_size_x()

            # Trace a ray
            Lava.lava_rt_payload_store_f32(0.0f0)
            Lava.lava_rt_trace_ray(
                UInt32(0), UInt32(0xFF),  # flags, cull_mask
                UInt32(0), UInt32(0), UInt32(0),  # sbt offset/stride/miss
                0.0f0, 0.0f0, -1.0f0, 0.001f0,   # origin, tmin
                0.0f0, 0.0f0, 1.0f0, 100.0f0     # direction, tmax
            )
            result = Lava.lava_rt_payload_load_f32()
            @inbounds output[ix + iy * sx + UInt32(1)] = result
            return nothing
        end
        d, _ = compile_and_disasm(basic_raygen,
                                   Tuple{Lava.LavaDeviceArray{Float32,1}};
                                   stage=:raygen)

        @testset "entry point" begin
            check(d, "OpEntryPoint RayGenerationKHR")
        end
        @testset "capability" begin
            check(d, "OpCapability RayTracingKHR")
        end
        @testset "extension" begin
            check(d, "SPV_KHR_ray_tracing")
        end
        @testset "trace ray" begin
            check(d, "OpTraceRayKHR")
        end
        @testset "payload" begin
            check(d, "RayPayloadKHR")
        end
        @testset "launch builtins" begin
            check_regex(d, "LaunchIdKHR")
            check_regex(d, "LaunchSizeKHR")
        end
    end

    @testset "raygen with BDA" begin
        function raygen_bda(output)
            ix = Lava.lava_rt_launch_id_x()
            @inbounds output[ix + UInt32(1)] = 1.0f0
            return nothing
        end
        d, _ = compile_and_disasm(raygen_bda,
                                   Tuple{Lava.LavaDeviceArray{Float32,1}};
                                   stage=:raygen)
        check(d, "PhysicalStorageBuffer")
        check(d, "PushConstant")
    end
end


