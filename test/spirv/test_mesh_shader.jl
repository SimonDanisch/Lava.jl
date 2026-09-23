# Mesh shader (SPV_EXT_mesh_shader) emission.
#
# A mesh stage is unlike every other graphics stage in this compiler: it runs as
# a WORKGROUP, writes into output arrays at slots it chooses, and then declares
# how much of those arrays it filled. So it carries `LocalSize` like a compute
# entry point, and its outputs are arrays rather than one value per invocation.
#
# The constants these tests pin (5364/5365, 5283, 5294/5295, 5270/5298, 5296)
# came out of /usr/include/glslang/SPIRV/spirv.hpp11 rather than from memory —
# a wrong opcode here assembles and then draws garbage.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_regex, compile_and_disasm
using GeometryBasics
import KernelInterface

const MESHOUT = Lava.LavaMeshOut()

@testset "Mesh Shaders" begin

    @testset "a triangle-emitting mesh stage" begin
        function one_triangle()
            KernelInterface.set_mesh_outputs!(MESHOUT, 3, 1)
            KernelInterface.set_mesh_vertex!(MESHOUT, 1, (position = Vec4f(-1, -1, 0, 1),))
            KernelInterface.set_mesh_vertex!(MESHOUT, 2, (position = Vec4f( 1, -1, 0, 1),))
            KernelInterface.set_mesh_vertex!(MESHOUT, 3, (position = Vec4f( 0,  1, 0, 1),))
            KernelInterface.set_mesh_triangle!(MESHOUT, 1, 1, 2, 3)
            return nothing
        end
        config = KernelInterface.MeshConfig(; max_vertices = 3, max_primitives = 1,
                                             topology = KernelInterface.TriangleList(),
                                             threads = 1)
        d, bytes = compile_and_disasm(one_triangle, Tuple{}; stage = :mesh, config)

        @testset "entry point and capability" begin
            check(d, "OpEntryPoint MeshEXT")
            check(d, "OpCapability MeshShadingEXT")
            # The capability is an extension one, so the extension has to be
            # declared too — spirv-val rejects the module otherwise.
            check(d, "OpExtension \"SPV_EXT_mesh_shader\"")
            # Not the NV models, which take different builtins and a different op.
            check_not(d, "MeshNV")
            check_not(d, "TaskNV")
        end

        # `check_regex` on the entry-point id rather than a literal `%main`: the
        # id carries the MANGLED FUNCTION NAME and only the entry point's string
        # operand is "main".
        @testset "execution modes" begin
            # A workgroup, like compute.
            check_regex(d, "OpExecutionMode %\\S+ LocalSize 1 1 1")
            # The declared MAXIMA the pipeline allocates for.
            check_regex(d, "OpExecutionMode %\\S+ OutputVertices 3")
            check_regex(d, "OpExecutionMode %\\S+ OutputPrimitivesEXT 1")
            check_regex(d, "OpExecutionMode %\\S+ OutputTrianglesEXT")
        end

        @testset "output arrays" begin
            # Per-vertex block array, Position on member 0.
            check(d, "OpMemberDecorate %gl_MeshPerVertexEXT 0 BuiltIn Position")
            check(d, "OpDecorate %gl_MeshPerVertexEXT Block")
            # The primitive plane: one uvec3 per primitive slot.
            check(d, "OpDecorate %gl_PrimitiveIndicesEXT BuiltIn PrimitiveTriangleIndicesEXT")
        end

        @testset "the counts are declared" begin
            check(d, "OpSetMeshOutputsEXT")
        end

        @testset "spirv-val accepts it" begin
            # `compile_and_disasm` passes `validate = true`, and validation
            # throws rather than returning — so reaching here at all is the
            # assertion. Stated explicitly because "it disassembled" and "it is
            # a legal module" are different claims, and only the second one
            # means a driver will take it.
            @test length(bytes) > 0
            @test bytes[1:4] == UInt8[0x03, 0x02, 0x23, 0x07]   # SPIR-V magic
        end
    end

    @testset "topology decides the index builtin" begin
        function two_points()
            KernelInterface.set_mesh_outputs!(MESHOUT, 2, 2)
            KernelInterface.set_mesh_vertex!(MESHOUT, 1, (position = Vec4f(0, 0, 0, 1),))
            KernelInterface.set_mesh_vertex!(MESHOUT, 2, (position = Vec4f(1, 1, 0, 1),))
            KernelInterface.set_mesh_point!(MESHOUT, 1, 1)
            KernelInterface.set_mesh_point!(MESHOUT, 2, 2)
            return nothing
        end
        config = KernelInterface.MeshConfig(; max_vertices = 2, max_primitives = 2,
                                             topology = KernelInterface.PointList(),
                                             threads = 1)
        d, _ = compile_and_disasm(two_points, Tuple{}; stage = :mesh, config)
        check_regex(d, "OpExecutionMode %\\S+ OutputPoints")
        check(d, "OpDecorate %gl_PrimitiveIndicesEXT BuiltIn PrimitivePointIndicesEXT")
        # Points index with a scalar, not a one-component vector.
        check_not(d, "PrimitiveTriangleIndicesEXT")
    end

    @testset "a wider workgroup keeps its LocalSize" begin
        function wide()
            KernelInterface.set_mesh_outputs!(MESHOUT, 3, 1)
            KernelInterface.set_mesh_vertex!(MESHOUT, 1, (position = Vec4f(0, 0, 0, 1),))
            KernelInterface.set_mesh_triangle!(MESHOUT, 1, 1, 1, 1)
            return nothing
        end
        config = KernelInterface.MeshConfig(; max_vertices = 64, max_primitives = 32,
                                             topology = KernelInterface.TriangleList(),
                                             threads = 32)
        d, _ = compile_and_disasm(wide, Tuple{}; stage = :mesh, config)
        check_regex(d, "OpExecutionMode %\\S+ LocalSize 32 1 1")
        check_regex(d, "OpExecutionMode %\\S+ OutputVertices 64")
        check_regex(d, "OpExecutionMode %\\S+ OutputPrimitivesEXT 32")
    end

    @testset "a strip topology is refused" begin
        # The primitive index array gives every primitive its own vertices, so a
        # strip has nothing to express and mapping it onto a list would silently
        # change what the shader draws.
        @test_throws ErrorException Lava.mesh_output_mode(KernelInterface.TriangleStrip())
    end
end
