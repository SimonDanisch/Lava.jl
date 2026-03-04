# Tier 1: llc reference oracle comparison
#
# Compares our emitter output against llc (SPIRV_LLVM_Backend_jll) for compute kernels.
# llc produces OpenCL-flavor SPIR-V (Kernel capability, Physical64 OpenCL), while we
# produce Vulkan-flavor (Shader capability, PhysicalStorageBuffer64 Vulkan). Structural
# comparison: same types, same arithmetic ops, same entry point patterns.
#
# llc uses `spirv64-unknown-unknown` triple — the Vulkan triple crashes on BDA kernels.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
    using .SPIRVTestUtils
end

@testset "llc Reference Comparison" begin

    # Helper: compile with our emitter, then compare structurally against llc
    function compare_against_llc(f, tt; workgroup_size=(64, 1, 1))
        # Our emitter
        r = Lava.lava_compile(f, tt; workgroup_size)
        our_disasm = r.spirv_disasm

        # llc reference — use pre-pass IR (raw GPUCompiler output, before BDA wrapping)
        llc_ok, llc_disasm = compile_with_llc(r.pre_pass_ir)

        return (our_disasm=our_disasm, llc_ok=llc_ok, llc_disasm=llc_disasm)
    end

    @testset "vadd — both compile, same float ops" begin
        function vadd(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] + B[i]
            return nothing
        end

        cmp = compare_against_llc(vadd, Tuple{Lava.LavaDeviceArray{Float32,1},
                                               Lava.LavaDeviceArray{Float32,1},
                                               Lava.LavaDeviceArray{Float32,1}})

        @test !isempty(cmp.our_disasm)
        @test cmp.llc_ok
        @test !isempty(cmp.llc_disasm)

        # Both emit OpFAdd for float addition
        check(cmp.our_disasm, "OpFAdd")
        check(cmp.llc_disasm, "OpFAdd")

        # Both have 32-bit float type
        check(cmp.our_disasm, "OpTypeFloat 32")
        check(cmp.llc_disasm, "OpTypeFloat 32")

        # Both have 64-bit int (for pointer arithmetic)
        check(cmp.our_disasm, "OpTypeInt 64")
        check(cmp.llc_disasm, "OpTypeInt 64")

        # Our output is Vulkan flavor
        check(cmp.our_disasm, "OpCapability Shader")
        check(cmp.our_disasm, "Vulkan")

        # llc output is OpenCL flavor
        check(cmp.llc_disasm, "OpCapability Kernel")
        check(cmp.llc_disasm, "OpenCL")
    end

    @testset "integer arithmetic — same ops" begin
        function int_arith(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = A[i] * B[i] + A[i]
            return nothing
        end

        cmp = compare_against_llc(int_arith, Tuple{Lava.LavaDeviceArray{Int32,1},
                                                    Lava.LavaDeviceArray{Int32,1},
                                                    Lava.LavaDeviceArray{Int32,1}})

        @test cmp.llc_ok

        # Both emit integer multiply and add
        check(cmp.our_disasm, "OpIMul")
        check(cmp.llc_disasm, "OpIMul")
        check(cmp.our_disasm, "OpIAdd")
        check(cmp.llc_disasm, "OpIAdd")

        # Both have 32-bit int type
        check(cmp.our_disasm, "OpTypeInt 32")
        check(cmp.llc_disasm, "OpTypeInt 32")
    end

    @testset "branching — both handle conditionals" begin
        function branching(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                if A[i] > 0.0f0
                    B[i] = 1.0f0
                else
                    B[i] = 0.0f0
                end
            end
            return nothing
        end

        cmp = compare_against_llc(branching, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                    Lava.LavaDeviceArray{Float32,1}})

        @test cmp.llc_ok

        # LLVM may optimize simple if/else into OpSelect (no branches).
        # Both should use either structured CF or OpSelect.
        @test occursin("OpBranchConditional", cmp.our_disasm) ||
              occursin("OpSelect", cmp.our_disasm)
        @test occursin("OpBranchConditional", cmp.llc_disasm) ||
              occursin("OpSelect", cmp.llc_disasm)

        # Both emit float comparison
        @test occursin("OpFOrd", cmp.our_disasm) || occursin("OpFUnord", cmp.our_disasm) ||
              occursin("OpSelect", cmp.our_disasm)
    end

    @testset "conversions — same conversion ops" begin
        function conv(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = Float32(A[i])
            return nothing
        end

        cmp = compare_against_llc(conv, Tuple{Lava.LavaDeviceArray{Int32,1},
                                               Lava.LavaDeviceArray{Float32,1}})

        @test cmp.llc_ok

        # Both emit signed int to float conversion
        check(cmp.our_disasm, "OpConvertSToF")
        check(cmp.llc_disasm, "OpConvertSToF")
    end

    @testset "bitwise ops — same bitwise instructions" begin
        function bitwise(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = (A[i] & B[i]) | (A[i] >> 2)
            return nothing
        end

        cmp = compare_against_llc(bitwise, Tuple{Lava.LavaDeviceArray{UInt32,1},
                                                   Lava.LavaDeviceArray{UInt32,1},
                                                   Lava.LavaDeviceArray{UInt32,1}})

        @test cmp.llc_ok

        check(cmp.our_disasm, "OpBitwiseAnd")
        check(cmp.llc_disasm, "OpBitwiseAnd")
        check(cmp.our_disasm, "OpBitwiseOr")
        check(cmp.llc_disasm, "OpBitwiseOr")
        check(cmp.our_disasm, "OpShiftRightLogical")
        check(cmp.llc_disasm, "OpShiftRightLogical")
    end

    @testset "memory model agreement" begin
        function noop(A)
            return nothing
        end

        cmp = compare_against_llc(noop, Tuple{Lava.LavaDeviceArray{Float32,1}})

        @test cmp.llc_ok

        # Both have OpMemoryModel
        check(cmp.our_disasm, "OpMemoryModel")
        check(cmp.llc_disasm, "OpMemoryModel")

        # Both have entry point
        check(cmp.our_disasm, "OpEntryPoint")
        check(cmp.llc_disasm, "OpEntryPoint")
    end

end
