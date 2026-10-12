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
# NVIDIA drops the rounding of a narrowed value it keeps in registers, whatever
# the module says, so `narrow_in_integers!` spells a narrowing the kernel computes
# with in integer operations: no `OpFConvert` to half is left for it. A narrowing
# that only goes to memory, and a widening of a loaded half, stay one instruction.

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

function fc_store!(dst, src, r::Float32)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds dst[i] = Float16(src[i] * r)
    return nothing
end

function fc_halfmul!(dst, src, g, r::Float32)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds dst[i] = Float32(Float16(src[i] * r) * g[i])
    return nothing
end

converts(d, ty) = [m[1] for m in eachmatch(Regex("(%\\w+) = OpFConvert %$ty "), d)]

const F32V = Lava.LavaDeviceArray{Float32,1}
const F16V = Lava.LavaDeviceArray{Float16,1}

@testset "float conversions carry NoContraction" begin
    # Stored only: the narrowing stays an instruction, decorated.
    d, _ = compile_and_disasm(fc_store!, Tuple{F16V,F32V,Float32}; validate = true)
    @test length(converts(d, "half")) == 1
    @test all(id -> occursin("OpDecorate $id NoContraction", d), converts(d, "half"))
    # Widened again: the narrowing is integer operations, the widening stays.
    d, _ = compile_and_disasm(fc_roundtrip!, Tuple{F32V,F32V,Float32}; validate = true)
    @test isempty(converts(d, "half"))
    @test length(converts(d, "float")) == 1
    @test all(id -> occursin("OpDecorate $id NoContraction", d), converts(d, "float"))
    @test occursin("OpShiftRightLogical", d)
end

@testset "a narrowing multiplied in half is integer operations too" begin
    d, _ = compile_and_disasm(fc_halfmul!, Tuple{F32V,F32V,F16V,Float32}; validate = true)
    @test isempty(converts(d, "half"))
end

@testset "a widening of a loaded half stays one instruction" begin
    d, _ = compile_and_disasm(fc_widen!, Tuple{F32V,F16V}; validate = true)
    @test length(converts(d, "float")) == 1
    @test isempty(converts(d, "half"))
end
