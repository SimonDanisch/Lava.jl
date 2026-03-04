# Tier 1: Compute shader basic emission tests
# Tests arithmetic, BDA, conversions, push constants

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end

@testset "Compute Basic" begin

    @testset "vadd (f32 add)" begin
        function vadd(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] + B[i]
            return nothing
        end
        d, _ = compile_and_disasm(vadd, Tuple{Lava.LavaDeviceArray{Float32,1},
                                               Lava.LavaDeviceArray{Float32,1},
                                               Lava.LavaDeviceArray{Float32,1}})

        @testset "entry point" begin
            check(d, "OpEntryPoint GLCompute")
            check(d, "OpExecutionMode")
            check(d, "LocalSize")
        end
        @testset "BDA + push constant" begin
            check(d, "PhysicalStorageBuffer")
            check(d, "PushConstant")
        end
        @testset "float add" begin
            check(d, "OpFAdd")
        end
        @testset "no float64 leakage" begin
            check_not(d, "OpTypeFloat 64")
        end
    end

    @testset "int arithmetic" begin
        function int_arith(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] * B[i] + A[i]
            return nothing
        end
        d, _ = compile_and_disasm(int_arith, Tuple{Lava.LavaDeviceArray{Int32,1},
                                                    Lava.LavaDeviceArray{Int32,1},
                                                    Lava.LavaDeviceArray{Int32,1}})
        check(d, "OpIMul")
        check(d, "OpIAdd")
    end

    @testset "bitwise ops" begin
        function bitwise(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = (A[i] & B[i]) | (A[i] >> 2)
            return nothing
        end
        d, _ = compile_and_disasm(bitwise, Tuple{Lava.LavaDeviceArray{UInt32,1},
                                                  Lava.LavaDeviceArray{UInt32,1},
                                                  Lava.LavaDeviceArray{UInt32,1}})
        check(d, "OpBitwiseAnd")
        check(d, "OpBitwiseOr")
        check(d, "OpShiftRightLogical")
    end

    @testset "conversions" begin
        function conv(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = Float32(A[i])
            return nothing
        end
        d, _ = compile_and_disasm(conv, Tuple{Lava.LavaDeviceArray{Int32,1},
                                               Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpConvertSToF")
    end

    @testset "unsigned to float conversion" begin
        function uconv(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = Float32(A[i])
            return nothing
        end
        d, _ = compile_and_disasm(uconv, Tuple{Lava.LavaDeviceArray{UInt32,1},
                                                Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpConvertUToF")
    end

    @testset "float to int conversion" begin
        function f2i(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = unsafe_trunc(Int32, A[i])
            return nothing
        end
        d, _ = compile_and_disasm(f2i, Tuple{Lava.LavaDeviceArray{Float32,1},
                                              Lava.LavaDeviceArray{Int32,1}})
        check(d, "OpConvertFToS")
    end

    @testset "capabilities and memory model" begin
        function noop(A)
            return nothing
        end
        d, _ = compile_and_disasm(noop, Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "OpCapability Shader")
        check(d, "OpCapability PhysicalStorageBuffer")
        check(d, "OpMemoryModel PhysicalStorageBuffer64 Vulkan")
    end

    @testset "workgroup size annotation" begin
        function sized_kernel(A)
            return nothing
        end
        d, _ = compile_and_disasm(sized_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}};
                                   workgroup_size=(128, 2, 1))
        check(d, "LocalSize 128 2 1")
    end
end
