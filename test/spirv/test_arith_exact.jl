# Tier 1: float arithmetic is exact unless Julia allows contracting it.
#
# Julia rounds every float operation on its own and fuses only where the code says so
# (`muladd`, `@fastmath`), which LLVM records as the `contract` flag. SPIR-V lets a driver
# treat an instruction without `NoContraction` as inexact, and Mesa then reassociates and
# folds across it: the error term of TwoSum, `(a - (s - bb)) + (b - bb)`, comes back 0 on
# lavapipe, which breaks double-float arithmetic. So every float operation without the flag
# carries `NoContraction`, and the ones with it do not.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

function ax_twosum!(s, e, a, b)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds begin
        x, y = a[i], b[i]
        t = x + y
        bb = t - x
        s[i] = t
        e[i] = (x - (t - bb)) + (y - bb)
    end
    return nothing
end

function ax_muladd!(out, a, b, c)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = muladd(a[i], b[i], c[i])
    return nothing
end

function ax_fastmath!(out, a, b, c)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = @fastmath a[i] * b[i] + c[i]
    return nothing
end

function ax_fma!(out, a, b, c)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = fma(a[i], b[i], c[i])
    return nothing
end

function ax_negdiv!(out, a, b)
    i = Int(KernelInterface.get_global_id().x)
    @inbounds out[i] = -(a[i] / b[i])
    return nothing
end

"Result ids of every `op` in disassembly `d`."
results(d, op) = [m[1] for m in eachmatch(Regex("(%\\w+) = $op "), d)]
exact(d, id) = occursin("OpDecorate $id NoContraction", d)

const AXV = Lava.LavaDeviceArray{Float32,1}

@testset "float arithmetic carries NoContraction" begin
    d, _ = compile_and_disasm(ax_twosum!, Tuple{AXV,AXV,AXV,AXV}; validate = true)
    ops = vcat(results(d, "OpFAdd"), results(d, "OpFSub"))
    @test length(ops) == 6
    @test all(id -> exact(d, id), ops)

    d, _ = compile_and_disasm(ax_negdiv!, Tuple{AXV,AXV,AXV}; validate = true)
    ops = vcat(results(d, "OpFDiv"), results(d, "OpFNegate"))
    @test length(ops) == 2
    @test all(id -> exact(d, id), ops)

    # `fma` is one rounding: exact as well.
    d, _ = compile_and_disasm(ax_fma!, Tuple{AXV,AXV,AXV,AXV}; validate = true)
    ops = results(d, "OpExtInst")
    @test length(ops) == 1
    @test exact(d, only(ops))
end

@testset "contractible arithmetic is left to fuse" begin
    # `muladd` fuses: LLVM turns its `contract` multiply and add into one `fma` (which then
    # carries the mark of a single rounding, as an `fma` written out does).
    d, _ = compile_and_disasm(ax_muladd!, Tuple{AXV,AXV,AXV,AXV}; validate = true)
    @test length(results(d, "OpExtInst")) == 1
    @test isempty(results(d, "OpFMul")) && isempty(results(d, "OpFAdd"))
    # `@fastmath` allows contraction and more: nothing is marked.
    d, _ = compile_and_disasm(ax_fastmath!, Tuple{AXV,AXV,AXV,AXV}; validate = true)
    @test !occursin("NoContraction", d)
end
