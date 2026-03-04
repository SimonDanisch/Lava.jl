# Tier 1: Miss shader emission tests
# Tests miss entry point and payload write

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end

@testset "Miss Shader" begin

    @testset "basic miss" begin
        function basic_miss()
            Lava._lava_rt_payload_store_f32(-1.0f0)
            return nothing
        end
        d, _ = compile_and_disasm(basic_miss, Tuple{};
                                   stage=:miss)

        @testset "entry point" begin
            check(d, "MissKHR")
        end
        @testset "payload" begin
            check(d, "IncomingRayPayloadKHR")
        end
    end

    @testset "miss with multi-field payload" begin
        function miss_multi()
            # Write sentinel values to all payload fields
            Lava._lava_rt_payload_store_f32_at(-1.0f0, UInt32(0))
            Lava._lava_rt_payload_store_f32_at(0.0f0, UInt32(1))
            return nothing
        end
        d, _ = compile_and_disasm(miss_multi, Tuple{};
                                   stage=:miss, payload_type=:f32_6)
        check(d, "MissKHR")
        check(d, "IncomingRayPayloadKHR")
    end
end
