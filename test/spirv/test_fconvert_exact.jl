# Tier 1: a float-to-float conversion is exact.
#
# `Float32(Float16(x))` rounds `x` to half precision, and Julia means that
# rounding. SPIR-V lets a driver treat any instruction WITHOUT `NoContraction` as
# inexact, and Mesa's NIR then folds the narrowing and the widening back into
# `x`: the rounding disappears. It surfaced as a fused Qwen-Image norm + rotary
# kernel that disagreed with the two kernels it replaced in a third of its
# outputs, one fp16 ulp each, while the same rounding through memory survived.
#
# So every `OpFConvert` Lava emits carries `NoContraction`. Arithmetic does not:
# fusing `a * b + c` into an fma is the contraction SPIR-V permits and every GPU
# Julia backend does, and it is not what this is about.
#
# NVIDIA folds the pair whatever the module says, so a widening of a value that
# was just narrowed is not an `OpFConvert` at all: `widen_narrowed_floats!`
# spells it in integer operations on the narrowed bits. An ordinary widening (a
# half loaded from memory) stays one instruction.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

function fc_roundtrip!(dst, src, r::Float32)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds dst[i] = Float32(Float16(src[i] * r)) * r
    return nothing
end

function fc_widen!(dst, src)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds dst[i] = Float32(src[i]) * 2f0
    return nothing
end

@testset "float conversions carry NoContraction" begin
    tt = Tuple{Lava.LavaDeviceArray{Float32,1},Lava.LavaDeviceArray{Float32,1},Float32}
    d, _ = compile_and_disasm(fc_roundtrip!, tt; validate = true)
    converts = [m[1] for m in eachmatch(r"(%\w+) = OpFConvert", d)]
    # The narrowing only: the widening of what it produced is integer arithmetic.
    @test length(converts) == 1
    for id in converts
        @test occursin("OpDecorate $id NoContraction", d)
    end
    @test occursin("OpConvertUToF", d)        # the subnormal case
    @test occursin("OpSelect", d)
end

@testset "a widening of a loaded half stays one instruction" begin
    tt = Tuple{Lava.LavaDeviceArray{Float32,1},Lava.LavaDeviceArray{Float16,1}}
    d, _ = compile_and_disasm(fc_widen!, tt; validate = true)
    @test length(collect(eachmatch(r"= OpFConvert", d))) == 1
    @test !occursin("OpConvertUToF", d)
end
