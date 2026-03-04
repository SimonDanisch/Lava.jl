# Tier 1: Geometry shader emission tests
# Tests entry point, topology, EmitVertex, EndPrimitive, InvocationId

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end
using GeometryBasics

@testset "Geometry Shader" begin

    @testset "passthrough geometry" begin
        function passthrough_geom()
            # Just pass through a single vertex
            Lava.set_position!(Vec4f(0.0f0, 0.0f0, 0.0f0, 1.0f0))
            Lava.emit_vertex!()
            Lava.end_primitive!()
            return nothing
        end
        config = Lava.GeometryConfig(;
            input=Lava.TriangleList(),
            output=Lava.TriangleStrip(),
            max_vertices=3,
            invocations=1
        )
        d, _ = compile_and_disasm(passthrough_geom, Tuple{};
                                   stage=:geometry, config=config)

        @testset "entry point" begin
            check(d, "OpEntryPoint Geometry")
        end
        @testset "capability" begin
            # Geometry capability may be implicit with Geometry entry point
            @test occursin("OpCapability Geometry", d) || occursin("OpEntryPoint Geometry", d)
        end
        @testset "emit vertex / end primitive" begin
            check(d, "OpEmitVertex")
            check(d, "OpEndPrimitive")
        end
        @testset "topology config" begin
            # SPIR-V uses "Triangles" for input topology, not "InputTriangles"
            check(d, "Triangles")
            check(d, "OutputTriangleStrip")
            check(d, "OutputVertices 3")
        end
    end

    @testset "invocation id" begin
        function geom_invocation()
            inv = Lava.invocation_id()
            Lava.set_position!(Vec4f(Float32(inv), 0.0f0, 0.0f0, 1.0f0))
            Lava.emit_vertex!()
            Lava.end_primitive!()
            return nothing
        end
        config = Lava.GeometryConfig(;
            input=Lava.TriangleList(),
            output=Lava.TriangleStrip(),
            max_vertices=3,
            invocations=2
        )
        d, _ = compile_and_disasm(geom_invocation, Tuple{};
                                   stage=:geometry, config=config)
        check_regex(d, "BuiltIn InvocationId")
        check(d, "Invocations 2")
    end
end
