# Tier 2: Golden file comparison tests
# Compile representative kernels and compare disassembly against blessed .spvasm files.
# Run with LAVA_BLESS=1 to create/update golden files.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc
using GeometryBasics

const GOLDEN_DIR = joinpath(@__DIR__, "golden")

@testset "Golden Files" begin

    @testset "compute_vadd" begin
        function vadd(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] + B[i]
            return nothing
        end
        _, bytes = compile_and_disasm(vadd, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                   Lava.LavaDeviceArray{Float32,1},
                                                   Lava.LavaDeviceArray{Float32,1}})
        compare_golden(bytes, joinpath(GOLDEN_DIR, "compute_vadd.spvasm"))
    end

    @testset "compute_struct" begin
        struct GoldenStruct
            x::Float32
            y::Float32
            z::Float32
        end
        function read_struct(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = A[i].x + A[i].y + A[i].z
            return nothing
        end
        _, bytes = compile_and_disasm(read_struct,
                                       Tuple{Lava.LavaDeviceArray{GoldenStruct,1},
                                             Lava.LavaDeviceArray{Float32,1}})
        compare_golden(bytes, joinpath(GOLDEN_DIR, "compute_struct.spvasm"))
    end

    @testset "vertex_passthrough" begin
        function vert_pt()
            idx = Lava.vertex_index()
            x = idx == Int32(1) ? -1.0f0 : (idx == Int32(2) ? 1.0f0 : 0.0f0)
            y = idx == Int32(1) ? -1.0f0 : (idx == Int32(2) ? -1.0f0 : 1.0f0)
            Lava.set_position!(Vec4f(x, y, 0.0f0, 1.0f0))
            return nothing
        end
        _, bytes = compile_and_disasm(vert_pt, Tuple{}; stage=:vertex)
        compare_golden(bytes, joinpath(GOLDEN_DIR, "vertex_passthrough.spvasm"))
    end

    @testset "fragment_solid" begin
        function frag_solid()
            Lava.gfx_output(0, Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        _, bytes = compile_and_disasm(frag_solid, Tuple{}; stage=:fragment)
        compare_golden(bytes, joinpath(GOLDEN_DIR, "fragment_solid.spvasm"))
    end

    @testset "raygen_simple" begin
        function raygen_simple(output)
            ix = Lava.lava_rt_launch_id_x()
            Lava._lava_rt_payload_store_f32(0.0f0)
            Lava._lava_rt_trace_ray(
                UInt32(0), UInt32(0xFF),
                UInt32(0), UInt32(0), UInt32(0),
                0.0f0, 0.0f0, -1.0f0, 0.001f0,
                0.0f0, 0.0f0, 1.0f0, 100.0f0
            )
            result = Lava._lava_rt_payload_load_f32()
            @inbounds output[ix + UInt32(1)] = result
            return nothing
        end
        _, bytes = compile_and_disasm(raygen_simple,
                                       Tuple{Lava.LavaDeviceArray{Float32,1}};
                                       stage=:raygen)
        compare_golden(bytes, joinpath(GOLDEN_DIR, "raygen_simple.spvasm"))
    end

    @testset "closesthit_simple" begin
        function chit_simple()
            t = Lava.lava_rt_ray_tmax()
            Lava._lava_rt_payload_store_f32(t)
            return nothing
        end
        _, bytes = compile_and_disasm(chit_simple, Tuple{};
                                       stage=:closesthit)
        compare_golden(bytes, joinpath(GOLDEN_DIR, "closesthit_simple.spvasm"))
    end

    @testset "miss_simple" begin
        function miss_simple()
            Lava._lava_rt_payload_store_f32(-1.0f0)
            return nothing
        end
        _, bytes = compile_and_disasm(miss_simple, Tuple{};
                                       stage=:miss)
        compare_golden(bytes, joinpath(GOLDEN_DIR, "miss_simple.spvasm"))
    end
end


