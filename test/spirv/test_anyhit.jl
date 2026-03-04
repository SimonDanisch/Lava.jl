# Tier 1: Any-hit shader emission tests
# Tests OpIgnoreIntersectionKHR, OpTerminateRayKHR

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end

@testset "Any-Hit Shader" begin

    @testset "ignore intersection" begin
        function anyhit_ignore()
            Lava._lava_rt_ignore_intersection()
            return nothing
        end
        d, _ = compile_and_disasm(anyhit_ignore, Tuple{};
                                   stage=:anyhit)

        @testset "entry point" begin
            check(d, "AnyHitKHR")
        end
        @testset "ignore" begin
            check(d, "OpIgnoreIntersectionKHR")
        end
    end

    @testset "terminate ray" begin
        function anyhit_terminate()
            Lava._lava_rt_terminate_ray()
            return nothing
        end
        d, _ = compile_and_disasm(anyhit_terminate, Tuple{};
                                   stage=:anyhit)
        check(d, "AnyHitKHR")
        check(d, "OpTerminateRayKHR")
    end
end
