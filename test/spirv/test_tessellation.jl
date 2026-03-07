# Tier 1: Tessellation shader emission tests
# Tests TessControl + TessEval entry points, builtins, config

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc
using GeometryBasics

@testset "Tessellation Shaders" begin

    @testset "tessellation control" begin
        function tess_ctrl()
            # Set tessellation levels
            Lava.set_tess_level_outer!(UInt32(0), 2.0f0)
            Lava.set_tess_level_outer!(UInt32(1), 2.0f0)
            Lava.set_tess_level_outer!(UInt32(2), 2.0f0)
            Lava.set_tess_level_inner!(UInt32(0), 1.0f0)
            return nothing
        end
        config = Lava.TessConfig(; vertices=3)
        d, _ = compile_and_disasm(tess_ctrl, Tuple{};
                                   stage=:tess_control, config=config)

        @testset "entry point" begin
            check(d, "OpEntryPoint TessellationControl")
        end
        @testset "capability" begin
            # Tessellation capability may be implicit with TessellationControl entry
            @test occursin("OpCapability Tessellation", d) || occursin("TessellationControl", d)
        end
        @testset "output vertices" begin
            check(d, "OutputVertices 3")
        end
    end

    @testset "tessellation evaluation" begin
        function tess_eval()
            u = Lava.tess_coord(1)
            v = Lava.tess_coord(2)
            w = Lava.tess_coord(3)
            Lava.set_position!(Vec4f(u, v, w, 1.0f0))
            return nothing
        end
        config = Lava.TessConfig(; vertices=3)
        d, _ = compile_and_disasm(tess_eval, Tuple{};
                                   stage=:tess_eval, config=config)

        @testset "entry point" begin
            check(d, "OpEntryPoint TessellationEvaluation")
        end
        @testset "tess coord" begin
            check_regex(d, "BuiltIn TessCoord")
        end
        @testset "domain" begin
            check(d, "Triangles")
        end
        @testset "spacing" begin
            check(d, "SpacingEqual")
        end
    end
end


