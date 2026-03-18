# Tier 1: Closest-hit shader emission tests
# Tests hit builtins, payload write, barycentric access

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

@testset "Closest Hit Shader" begin

    @testset "basic closest hit" begin
        function basic_chit()
            t = Lava.lava_rt_ray_tmax()
            Lava._lava_rt_payload_store_f32(t)
            return nothing
        end
        d, _ = compile_and_disasm(basic_chit, Tuple{};
                                   stage=:closesthit)

        @testset "entry point" begin
            check(d, "ClosestHitKHR")
        end
        @testset "payload" begin
            check(d, "IncomingRayPayloadKHR")
        end
    end

    @testset "hit builtins" begin
        function chit_builtins()
            prim = Lava.lava_rt_primitive_id()
            inst = Lava.lava_rt_instance_custom_index()
            kind = Lava.lava_rt_hit_kind()
            Lava._lava_rt_payload_store_f32(Float32(prim + inst + kind))
            return nothing
        end
        d, _ = compile_and_disasm(chit_builtins, Tuple{};
                                   stage=:closesthit)
        check_regex(d, "PrimitiveId")
        check_regex(d, "InstanceCustomIndexKHR")
        check_regex(d, "HitKindKHR")
    end

    @testset "barycentric access" begin
        function chit_bary()
            u = Lava.lava_rt_hit_bary_u()
            v = Lava.lava_rt_hit_bary_v()
            Lava._lava_rt_payload_store_f32(u + v)
            return nothing
        end
        d, _ = compile_and_disasm(chit_bary, Tuple{};
                                   stage=:closesthit)
        check(d, "HitAttributeKHR")
    end

    @testset "multi-field payload" begin
        function chit_multi()
            t = Lava.lava_rt_ray_tmax()
            prim = Lava.lava_rt_primitive_id()
            Lava._lava_rt_payload_store_f32_at(t, UInt32(0))
            Lava._lava_rt_payload_store_f32_at(Float32(prim), UInt32(1))
            return nothing
        end
        d, _ = compile_and_disasm(chit_multi, Tuple{};
                                   stage=:closesthit, payload_type=:f32_6)
        check(d, "IncomingRayPayloadKHR")
    end
end


