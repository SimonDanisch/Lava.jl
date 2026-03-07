# Tier 1: Compute shader type emission tests
# Tests structs, NTuples, nested types via BDA

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

struct SimpleStruct
    x::Float32
    y::Float32
    z::Float32
end

struct NestedStruct
    pos::SimpleStruct
    value::Float32
end

struct TupleStruct
    data::NTuple{4, Float32}
    id::Int32
end

struct DeepNested
    inner::NestedStruct
    scale::Float32
    flags::UInt32
end

@testset "Compute Types" begin

    @testset "simple struct via BDA" begin
        function read_struct(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = A[i]
                B[i] = s.x + s.y + s.z
            end
            return nothing
        end
        d, _ = compile_and_disasm(read_struct,
                                   Tuple{Lava.LavaDeviceArray{SimpleStruct,1},
                                         Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpTypeStruct")
        check(d, "OpFAdd")
    end

    @testset "nested struct" begin
        function read_nested(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = A[i]
                B[i] = s.pos.x + s.value
            end
            return nothing
        end
        d, _ = compile_and_disasm(read_nested,
                                   Tuple{Lava.LavaDeviceArray{NestedStruct,1},
                                         Lava.LavaDeviceArray{Float32,1}})
        # Should have at least 2 struct types (SimpleStruct and NestedStruct)
        n_structs = count("OpTypeStruct", d)
        @test n_structs >= 2
    end

    @testset "NTuple field" begin
        function read_tuple(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = A[i]
                B[i] = s.data[1] + s.data[2] + s.data[3] + s.data[4]
            end
            return nothing
        end
        d, _ = compile_and_disasm(read_tuple,
                                   Tuple{Lava.LavaDeviceArray{TupleStruct,1},
                                         Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpTypeArray")
    end

    @testset "deeply nested struct" begin
        function deep_read(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = A[i]
                B[i] = s.inner.pos.x * s.scale
            end
            return nothing
        end
        d, _ = compile_and_disasm(deep_read,
                                   Tuple{Lava.LavaDeviceArray{DeepNested,1},
                                         Lava.LavaDeviceArray{Float32,1}})
        # Multiple struct types for the nesting
        n_structs = count("OpTypeStruct", d)
        @test n_structs >= 3
    end

    @testset "multi-field struct write" begin
        function write_struct(A)
            i = Lava.lava_global_invocation_id_x()
            @inbounds A[i] = SimpleStruct(1.0f0, 2.0f0, 3.0f0)
            return nothing
        end
        d, _ = compile_and_disasm(write_struct,
                                   Tuple{Lava.LavaDeviceArray{SimpleStruct,1}})
        check(d, "OpTypeStruct")
        # Should have stores
        check(d, "OpStore")
    end

    @testset "struct member offset decorations" begin
        function offset_test(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = A[i].x
            return nothing
        end
        d, _ = compile_and_disasm(offset_test,
                                   Tuple{Lava.LavaDeviceArray{SimpleStruct,1},
                                         Lava.LavaDeviceArray{Float32,1}})
        # PSB struct member offsets should be decorated
        check_regex(d, "MemberDecorate.*Offset")
    end
end


