# The SPIR-V module builder, used directly: a compute shader and three
# ray-tracing shaders assembled instruction by instruction
# (`handbuilt_shaders.jl`), then validated with spirv-val and disassembled.
#
# Moved here from Mantle's `test/vulkan/test_handwritten_spirv.jl` and
# `test_handwritten_rt.jl`. The compute shader's device half ran the module on a
# Vulkan device of its own through Vulkan.jl and checked output == 2 * input,
# which every kernel test in Mantle checks of its own module through the
# backend's pipeline path. The ray-tracing shaders' device half is Mantle's
# `test/test_rt_pipeline.jl`, with the shaders written in Julia.

using Test, Lava
include(joinpath(@__DIR__, "handbuilt_shaders.jl"))

@testset "hand-built shader modules" begin
    # output[gid] = input[gid] * 2, two storage-buffer descriptors.
    @testset "compute: times 2" begin
        spirv_binary = build_times2_shader()
        @test length(spirv_binary) > 0
        @test length(spirv_binary) % 4 == 0  # must be UInt32 aligned

        # Check magic number
        magic = reinterpret(UInt32, spirv_binary[1:4])[1]
        @test magic == 0x07230203

        # Validate with spirv-val
        @test_nowarn Lava.validate_spirv(spirv_binary)

        # Disassemble and check key instructions
        disasm = Lava.disassemble_spirv(spirv_binary)
        @test occursin("OpEntryPoint GLCompute", disasm)
        @test occursin("OpExecutionMode %main LocalSize 64 1 1", disasm)
        @test occursin("OpFMul", disasm)
        @test occursin("BuiltIn GlobalInvocationId", disasm)
        @test occursin("StorageBuffer", disasm)
    end

    @testset "ray tracing: raygen, closest-hit, miss" begin
        raygen_spirv = build_raygen_shader()
        chit_spirv = build_closesthit_shader()
        miss_spirv = build_miss_shader()

        @test length(raygen_spirv) > 0
        @test length(chit_spirv) > 0
        @test length(miss_spirv) > 0

        # Validate all three
        @test_nowarn Lava.validate_spirv(raygen_spirv)
        @test_nowarn Lava.validate_spirv(chit_spirv)
        @test_nowarn Lava.validate_spirv(miss_spirv)

        # Check key instructions in raygen
        raygen_dis = Lava.disassemble_spirv(raygen_spirv)
        @test occursin("OpEntryPoint RayGenerationKHR", raygen_dis)
        @test occursin("OpTraceRayKHR", raygen_dis)
        @test occursin("RayTracingKHR", raygen_dis)

        # Check closest-hit
        chit_dis = Lava.disassemble_spirv(chit_spirv)
        @test occursin("OpEntryPoint ClosestHitKHR", chit_dis)

        # Check miss
        miss_dis = Lava.disassemble_spirv(miss_spirv)
        @test occursin("OpEntryPoint MissKHR", miss_dis)
    end
end
