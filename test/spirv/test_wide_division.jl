# Tier 1: no 64-bit integer division by a constant, and a runtime one only behind
# a test that both magnitudes fit in 32 bits.
#
# NVIDIA miscompiles the code around its own 64-bit division (Mantle's
# `test_shared_index_division.jl`, `test_int32_cartesian_miscompile.jl`), so
# `lower_wide_divisions!` turns a constant divisor into shifts (a power of two)
# or a multiply-high sequence, and a runtime divisor into a 32-bit division with
# the 64-bit one kept for magnitudes past 2^32. The values are checked on the
# host by `test_exact_division.jl` and on a device by Mantle's
# `test_integer_division.jl`; this checks the shape of the module.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

function wd_const!(dst, src)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds dst[i] = src[i] ÷ 96 + src[i] % 7
    return nothing
end

function wd_runtime!(dst, src, d::Int64)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds dst[i] = src[i] ÷ d
    return nothing
end

divisions(d, ty) = count(l -> occursin(Regex("= Op(SDiv|UDiv|SRem|UMod|SMod) %$ty "), l), split(d, '\n'))

@testset "64-bit division" begin
    A = Lava.LavaDeviceArray{Int64,1}
    @testset "by a constant: none" begin
        d, _ = compile_and_disasm(wd_const!, Tuple{A,A}; validate = true)
        @test divisions(d, "ulong") == 0
        @test divisions(d, "long") == 0
    end
    @testset "by a runtime value: one of each width" begin
        d, _ = compile_and_disasm(wd_runtime!, Tuple{A,A,Int64}; validate = true)
        @test divisions(d, "ulong") + divisions(d, "long") == 1
        @test divisions(d, "uint") == 1
    end
end
