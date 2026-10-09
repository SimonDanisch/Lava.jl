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
import .SPIRVTestUtils: check, check_not, check_dag, check_regex, check_count, compile_and_disasm
using GeometryBasics
import KernelInterface

# The output object carries the stage's declaration — its output type, its
# `Flat` names, its topology — the way `MeshWrapper` builds it from Mantle's
# `MeshShader`. A stage that writes only `position` declares nothing else.
const POSITION_ONLY = NamedTuple{(:position,), Tuple{Vec4f}}
const MESHOUT = Lava.LavaMeshOut{POSITION_ONLY, (), KernelInterface.TriangleList}()
const POINTOUT = Lava.LavaMeshOut{POSITION_ONLY, (), KernelInterface.PointList}()

# `uv` smooth and `colour` flat, in that order: Location 0 and 1, `colour` in
# the primitive plane. Mantle's portable MeshPipeline test declares exactly this.
const UV_COLOUR = NamedTuple{(:position, :uv, :colour),
                             Tuple{Vec4f, NTuple{2,Float32}, NTuple{4,Float32}}}
const PRIMOUT = Lava.LavaMeshOut{UV_COLOUR, (:colour,), KernelInterface.TriangleList}()
const GREEN = (0f0, 1f0, 0f0, 1f0)

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
            KernelInterface.set_mesh_outputs!(POINTOUT, 2, 2)
            KernelInterface.set_mesh_vertex!(POINTOUT, 1, (position = Vec4f(0, 0, 0, 1),))
            KernelInterface.set_mesh_vertex!(POINTOUT, 2, (position = Vec4f(1, 1, 0, 1),))
            KernelInterface.set_mesh_point!(POINTOUT, 1, 1)
            KernelInterface.set_mesh_point!(POINTOUT, 2, 2)
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

    # Two maxima that differ, so an array sized by the wrong one shows.
    primconfig = KernelInterface.MeshConfig(; max_vertices = 4, max_primitives = 2,
                                             topology = KernelInterface.TriangleList(),
                                             threads = 1)

    @testset "per-primitive data is a PerPrimitiveEXT array" begin
        function by_primitive()
            KernelInterface.set_mesh_outputs!(PRIMOUT, 3, 1)
            KernelInterface.set_mesh_vertex!(PRIMOUT, 1, (position = Vec4f(-1, -1, 0, 1), uv = (0f0, 0f0)))
            KernelInterface.set_mesh_vertex!(PRIMOUT, 2, (position = Vec4f( 3, -1, 0, 1), uv = (2f0, 0f0)))
            KernelInterface.set_mesh_vertex!(PRIMOUT, 3, (position = Vec4f(-1,  3, 0, 1), uv = (0f0, 2f0)))
            KernelInterface.set_mesh_triangle!(PRIMOUT, 1, 1, 2, 3)
            KernelInterface.set_mesh_primitive_data!(PRIMOUT, 1, (colour = GREEN,))
            return nothing
        end
        d, _ = compile_and_disasm(by_primitive, Tuple{}; stage = :mesh, config = primconfig)
        # `uv` per vertex at Location 0, an array of max_VERTICES…
        check(d, "OpDecorate %mesh_out_loc0 Location 0")
        check(d, "%mesh_out_loc0 = OpVariable %_ptr_Output__arr_v2float_uint_4 Output")
        # …`colour` per primitive at Location 1, its declared place, although the
        # tuple that writes it has no `uv` in front of it: an array of
        # max_PRIMITIVES, decorated as glslang decorates `perprimitiveEXT out`.
        check(d, "OpDecorate %mesh_prim_out_loc1 Location 1")
        check(d, "OpDecorate %mesh_prim_out_loc1 PerPrimitiveEXT")
        check(d, "%mesh_prim_out_loc1 = OpVariable %_ptr_Output__arr_v4float_uint_2 Output")
        check_not(d, "%mesh_out_loc1")
    end

    @testset "a Flat field written per vertex goes to the primitive plane" begin
        # The other legal spelling: `colour` in every vertex tuple, and FIRST in
        # it, ahead of `uv`. It lands where the declaration says — `uv` at 0,
        # `colour` per primitive at 1 — and only the provoking vertex (slot 1 of
        # the triangle) writes it, so there is one store into the primitive
        # array, not three.
        function by_vertex()
            KernelInterface.set_mesh_outputs!(PRIMOUT, 3, 1)
            KernelInterface.set_mesh_vertex!(PRIMOUT, 1, (position = Vec4f(-1, -1, 0, 1), colour = GREEN, uv = (0f0, 0f0)))
            KernelInterface.set_mesh_vertex!(PRIMOUT, 2, (position = Vec4f( 3, -1, 0, 1), colour = GREEN, uv = (2f0, 0f0)))
            KernelInterface.set_mesh_vertex!(PRIMOUT, 3, (position = Vec4f(-1,  3, 0, 1), colour = GREEN, uv = (0f0, 2f0)))
            KernelInterface.set_mesh_triangle!(PRIMOUT, 1, 1, 2, 3)
            return nothing
        end
        d, _ = compile_and_disasm(by_vertex, Tuple{}; stage = :mesh, config = primconfig)
        check(d, "OpDecorate %mesh_out_loc0 Location 0")
        check(d, "OpDecorate %mesh_prim_out_loc1 PerPrimitiveEXT")
        check_count(d, "OpAccessChain %_ptr_Output_v4float %mesh_prim_out_loc1", 1)
        check_not(d, "%mesh_out_loc1")
    end

    @testset "a fragment stage reads the primitive plane per primitive" begin
        # Vulkan requires `PerPrimitiveEXT` on both sides of the location, and the
        # decoration is SPV_EXT_mesh_shader's — so a fragment stage that reads one
        # declares the capability, although it is not a mesh stage itself.
        frag_colour(inputs) = inputs.colour
        wrapped = Lava.FragmentWrapper{typeof(frag_colour), UV_COLOUR, (:colour,), (:colour,)}()
        d, _ = compile_and_disasm(wrapped, Tuple{}; stage = :fragment)
        check(d, "OpCapability MeshShadingEXT")
        check(d, "OpExtension \"SPV_EXT_mesh_shader\"")
        check(d, "OpDecorate %in_primitive_loc1 Location 1")
        check(d, "OpDecorate %in_primitive_loc1 PerPrimitiveEXT")
        check(d, "OpDecorate %in_primitive_loc1 Flat")
        # The same stage fed by a VERTEX stage's flat varying reads it flat and
        # nothing else: no mesh capability in a pipeline that has no mesh stage.
        flat = Lava.FragmentWrapper{typeof(frag_colour), UV_COLOUR, (:colour,)}()
        d, _ = compile_and_disasm(flat, Tuple{}; stage = :fragment)
        check(d, "OpDecorate %in_flat_loc1 Flat")
        check_not(d, "PerPrimitive")
        check_not(d, "MeshShadingEXT")
    end

    @testset "a strip topology is refused" begin
        # The primitive index array gives every primitive its own vertices, so a
        # strip has nothing to express and mapping it onto a list would silently
        # change what the shader draws.
        @test_throws ErrorException Lava.mesh_output_mode(KernelInterface.TriangleStrip())
    end
end
