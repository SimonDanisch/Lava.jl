# Tier 1: Fragment shader emission tests
# Tests FragCoord, FrontFacing, multi-output, BDA args

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end
using GeometryBasics

@testset "Fragment Shader" begin

    @testset "solid color" begin
        function solid_frag()
            Lava.gfx_output(0, Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(solid_frag, Tuple{}; stage=:fragment)

        @testset "entry point" begin
            check(d, "OpEntryPoint Fragment")
        end
        @testset "execution mode" begin
            check(d, "OriginUpperLeft")
        end
        @testset "output" begin
            check_regex(d, "Location 0")
        end
    end

    @testset "frag coord builtin" begin
        function frag_coord_test()
            x = Lava.frag_coord_x()
            y = Lava.frag_coord_y()
            Lava.gfx_output(0, Vec4f(x / 800.0f0, y / 600.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(frag_coord_test, Tuple{}; stage=:fragment)
        check_regex(d, "BuiltIn FragCoord")
    end

    @testset "front facing builtin" begin
        function front_facing_test()
            ff = Lava.front_facing()
            r = ff ? 1.0f0 : 0.0f0
            Lava.gfx_output(0, Vec4f(r, 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(front_facing_test, Tuple{}; stage=:fragment)
        check_regex(d, "BuiltIn FrontFacing")
    end

    @testset "input variables from vertex" begin
        function frag_input()
            color = Lava.gfx_input(Vec4f, 0)
            Lava.gfx_output(0, color)
            return nothing
        end
        d, _ = compile_and_disasm(frag_input, Tuple{}; stage=:fragment)
        check_regex(d, "Input")
        check_regex(d, "Location 0")
    end

    @testset "multi-output" begin
        function frag_multi_out()
            Lava.gfx_output(0, Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))  # color
            Lava.gfx_output(1, Vec4f(0.0f0, 0.0f0, 1.0f0, 0.0f0))  # normal
            return nothing
        end
        d, _ = compile_and_disasm(frag_multi_out, Tuple{}; stage=:fragment)
        check_regex(d, "Location 0")
        check_regex(d, "Location 1")
    end

    @testset "BDA args in fragment" begin
        function frag_bda(colors)
            @inbounds c = colors[1]
            Lava.gfx_output(0, Vec4f(c[1], c[2], c[3], 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(frag_bda,
                                   Tuple{Lava.LavaDeviceArray{Vec3f,1}};
                                   stage=:fragment)
        check(d, "PhysicalStorageBuffer")
        check(d, "PushConstant")
    end
end
