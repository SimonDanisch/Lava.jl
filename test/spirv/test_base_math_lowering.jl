# Tier 1: Base math keeps Julia's semantics; `@fastmath` is the native instruction.
#
# The values are checked on devices by Mantle's `test_math_intrinsics.jl`. This checks the
# instructions, so a lowering that would only differ for a NaN, a tie or a negative
# constant fails here, on any machine.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

const BLF32 = Tuple{Lava.LavaDeviceArray{Float32,1}, Lava.LavaDeviceArray{Float32,1}}

function bl_round!(out, x)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = round(x[i])
    return nothing
end

function bl_min!(out, x)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = min(x[i], out[i])
    return nothing
end

function bl_min_fast!(out, x)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = @fastmath min(x[i], out[i])
    return nothing
end

@noinline bl_tofloat(n::Int32) = Float32(n)

function bl_negative_constant!(out, x)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = x[i] * bl_tofloat(Int32(-3))
    return nothing
end

ext(disasm, name) = occursin(Regex("OpExtInst %\\w+ %\\w+ $name "), disasm)

@testset "round is RoundEven" begin
    # Julia's `round` breaks ties to even; GLSL.std.450 Round breaks them in a direction
    # the implementation chooses
    d, _ = compile_and_disasm(bl_round!, BLF32)
    @test ext(d, "RoundEven")
    @test !ext(d, "Round")
end

@testset "min is Julia's, @fastmath min is NMin" begin
    d, _ = compile_and_disasm(bl_min!, BLF32)
    @test ext(d, "NMin")
    @test occursin("OpIsNan", d)   # a NaN operand gives NaN
    f, _ = compile_and_disasm(bl_min_fast!, BLF32)
    @test ext(f, "NMin")
    @test !occursin("OpIsNan", f)
end

@testset "a negative constant converts as signed after spirv-opt" begin
    # spirv-opt inlines `bl_tofloat` and folds the conversion. It reads the operand's type
    # for the signedness, and an unsigned one made -3 into 4.2949673e9.
    d, _ = compile_and_disasm(bl_negative_constant!, BLF32)
    @test !occursin("4.2949673e+09", d)
    @test occursin(r"OpConstant %float -3\b", d)
end
