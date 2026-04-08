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

# ── Bool padding alignment tests (Tier 1 SPIR-V pattern checks) ──
# Verifies that structs with Bool (i8) fields produce correct MemberOffset decorations
# and that loads/stores after Bool fields don't use wrong Aligned operands.

struct BoolPadSPIRV
    x::Float32      # offset 0
    flag::Bool       # offset 4 (i8), 3 bytes padding
    y::Float32       # offset 8
end

struct TwoBoolSPIRV
    a::Float32       # offset 0
    b::Float32       # offset 4
    c::Float32       # offset 8
    f1::Bool         # offset 12
    f2::Bool         # offset 13, 2 bytes padding
    d::Float32       # offset 16
end

struct BoolBetweenPtrsSPIRV
    arr1::Ptr{Float32}  # 0: 8 bytes
    active::Bool         # 8: 1 byte, 7 bytes padding
    arr2::Ptr{Float32}  # 16: 8 bytes
    count::Int32         # 24: 4 bytes
end

@testset "Bool Padding SPIR-V" begin

    @testset "BoolPadSPIRV member offsets" begin
        function read_bool_pad(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = A[i]
                B[i] = s.flag ? s.x + s.y : s.x - s.y
            end
            return nothing
        end
        d, bytes = compile_and_disasm(read_bool_pad,
            Tuple{Lava.LavaDeviceArray{BoolPadSPIRV,1}, Lava.LavaDeviceArray{Float32,1}};
            validate=true)
        # MemberOffset for y (field 2) must be 8, not 5 (which would be without padding)
        check_regex(d, "MemberDecorate.*Offset 8")
        # spirv-val must pass (validates Aligned operands)
    end

    @testset "TwoBoolSPIRV member offsets" begin
        function read_two_bool(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = A[i]
                B[i] = s.f1 ? s.a + s.d : s.b - s.d
            end
            return nothing
        end
        d, bytes = compile_and_disasm(read_two_bool,
            Tuple{Lava.LavaDeviceArray{TwoBoolSPIRV,1}, Lava.LavaDeviceArray{Float32,1}};
            validate=true)
        # d (field 5) must be at offset 16, not 14
        check_regex(d, "MemberDecorate.*Offset 16")
    end

    @testset "BoolBetweenPtrsSPIRV write after Bool" begin
        function write_after_bool(A, B, flag::Bool, val::Float32)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                if flag
                    B[i] = val
                else
                    B[i] = -val
                end
            end
            return nothing
        end
        d, bytes = compile_and_disasm(write_after_bool,
            Tuple{Lava.LavaDeviceArray{Float32,1}, Lava.LavaDeviceArray{Float32,1},
                  Bool, Float32};
            validate=true)
        # Bool scalar arg followed by Float32 arg: no alignment issues in push constant layout
        check(d, "OpStore")
    end

    @testset "no Aligned 2 or Aligned 3 in any Bool struct kernel" begin
        # Aligned 2 or Aligned 3 on a Float32 store would indicate broken padding
        function all_fields_bool_struct(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                s = A[i]
                B[i] = s.x + s.y + (s.flag ? 1f0 : 0f0)
            end
            return nothing
        end
        d, _ = compile_and_disasm(all_fields_bool_struct,
            Tuple{Lava.LavaDeviceArray{BoolPadSPIRV,1}, Lava.LavaDeviceArray{Float32,1}};
            validate=true)
        # Must NOT have Aligned 2 or Aligned 3 (would mean wrong field offset)
        check_not(d, "Aligned 2")
        check_not(d, "Aligned 3")
    end

    # Exotic element types: CartesianIndex, Tuple{Int64, CartesianIndex}
    # These produce non-standard LLVM integer widths (i3 for type tags)
    # and struct layouts with pointer + dims array ({ptr, [N x i64]}).

    @testset "CartesianIndex element type (no OpTypeInt 3)" begin
        function read_cartesian(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                ci = A[i]
                B[i] = Int64(ci[1])
            end
            return nothing
        end
        d, _ = compile_and_disasm(read_cartesian,
            Tuple{Lava.LavaDeviceArray{CartesianIndex{3},1}, Lava.LavaDeviceArray{Int64,1}};
            validate=true)
        # Must NOT have invalid 3-bit OpTypeInt
        check_not(d, "OpTypeInt 3 ")
    end

    @testset "Tuple{Int64, CartesianIndex{4}} element type" begin
        function read_tuple_cartesian(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                tup = A[i]
                level = tup[1]
                B[i] = level
            end
            return nothing
        end
        d, _ = compile_and_disasm(read_tuple_cartesian,
            Tuple{Lava.LavaDeviceArray{Tuple{Int64, CartesianIndex{4}},1}, Lava.LavaDeviceArray{Int64,1}};
            validate=true)
        check_not(d, "OpTypeInt 3 ")
    end

    @testset "closure capturing LavaDeviceArray with dynamic dims access" begin
        # Reproduces the BiotSavartBCs pattern: a closure captures a LavaDeviceArray,
        # and the kernel reads from it using a dynamic index. The emitter must correctly
        # decompose byte-offset GEPs into the struct's dims array ({ptr, [N x i64]}).
        function make_closure_kernel(captured::Lava.LavaDeviceArray{Float32,1})
            function inner(output)
                i = Lava.lava_global_invocation_id_x()
                @inbounds output[i] = captured[i]
                return nothing
            end
            return inner
        end
        captured_da = Lava.LavaDeviceArray{Float32,1}(Ptr{Float32}(), (64,))
        kern = make_closure_kernel(captured_da)
        d, _ = compile_and_disasm(kern,
            Tuple{Lava.LavaDeviceArray{Float32,1}};
            validate=true)
        check(d, "OpAccessChain")
    end
end
