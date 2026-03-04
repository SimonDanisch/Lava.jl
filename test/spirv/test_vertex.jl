# Tier 1: Vertex shader emission tests
# Tests entry point, builtins, I/O variables, BDA args

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end
using GeometryBasics

@testset "Vertex Shader" begin

    @testset "passthrough vertex" begin
        function passthrough_vert()
            idx = Lava.vertex_index()
            # Simple triangle positions
            x = idx == Int32(1) ? -1.0f0 : (idx == Int32(2) ? 1.0f0 : 0.0f0)
            y = idx == Int32(1) ? -1.0f0 : (idx == Int32(2) ? -1.0f0 : 1.0f0)
            Lava.set_position!(Vec4f(x, y, 0.0f0, 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(passthrough_vert, Tuple{}; stage=:vertex)

        @testset "entry point" begin
            check(d, "OpEntryPoint Vertex")
        end
        @testset "no compute markers" begin
            check_not(d, "GLCompute")
            check_not(d, "LocalSize")
        end
        @testset "position builtin" begin
            check_regex(d, "BuiltIn Position")
        end
        @testset "vertex index builtin" begin
            check_regex(d, "BuiltIn VertexIndex")
        end
    end

    @testset "instance index builtin" begin
        function instance_vert()
            vidx = Lava.vertex_index()
            iidx = Lava.instance_index()
            Lava.set_position!(Vec4f(Float32(vidx), Float32(iidx), 0.0f0, 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(instance_vert, Tuple{}; stage=:vertex)
        check_regex(d, "BuiltIn InstanceIndex")
    end

    @testset "vertex output variables" begin
        function vert_output()
            idx = Lava.vertex_index()
            Lava.set_position!(Vec4f(0.0f0, 0.0f0, 0.0f0, 1.0f0))
            Lava.gfx_output(0, Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(vert_output, Tuple{}; stage=:vertex)
        check_regex(d, "Location 0")
        check_regex(d, "Output")
    end

    @testset "vertex with BDA args" begin
        function vert_bda(positions)
            idx = Lava.vertex_index()
            @inbounds pos = positions[idx]
            Lava.set_position!(Vec4f(pos[1], pos[2], pos[3], 1.0f0))
            return nothing
        end
        d, _ = compile_and_disasm(vert_bda,
                                   Tuple{Lava.LavaDeviceArray{Vec3f,1}};
                                   stage=:vertex)
        check(d, "PhysicalStorageBuffer")
        check(d, "PushConstant")
    end

    @testset "multiple output locations" begin
        function vert_multi_out()
            idx = Lava.vertex_index()
            Lava.set_position!(Vec4f(0.0f0, 0.0f0, 0.0f0, 1.0f0))
            Lava.gfx_output(0, Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))  # color
            Lava.gfx_output(1, Vec2f(0.5f0, 0.5f0))                 # uv
            return nothing
        end
        d, _ = compile_and_disasm(vert_multi_out, Tuple{}; stage=:vertex)
        check_regex(d, "Location 0")
        check_regex(d, "Location 1")
    end
end
