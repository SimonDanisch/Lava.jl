# Tier 1: LLVM's `shufflevector` is SPIR-V's `OpVectorShuffle`.
#
# Taking two components of a vector — `(x[1], x[2])` of an
# `NTuple{4,VecElement{Float16}}` — is what SROA turns into
# `shufflevector <4 x half> %x, poison, <0, 1>`, and the emitter had no case for
# it: "Unsupported LLVM instruction: ShuffleVectorInst". Found staging a
# transposed V block in DNNKernels' flash attention, which splits each 16-byte
# load into 4-byte pairs.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

const _VS_H4 = NTuple{4,VecElement{Float16}}
const _VS_H2 = NTuple{2,VecElement{Float16}}

function vs_split_halves!(dst, src)
    i = Int(KernelInterface.get_global_id().x)
    x = Core.Intrinsics.pointerref(reinterpret(Ptr{_VS_H4}, pointer(src)), i, 8)
    q = reinterpret(Ptr{_VS_H2}, pointer(dst))
    # The halves swapped, so the shuffle's component indices matter.
    Core.Intrinsics.pointerset(q, (x[3], x[4]), 2i - 1, 4)
    Core.Intrinsics.pointerset(q, (x[1], x[2]), 2i, 4)
    return nothing
end

@testset "shufflevector lowers to OpVectorShuffle" begin
    tt = Tuple{Lava.LavaDeviceArray{Float16,1},Lava.LavaDeviceArray{Float16,1}}
    # `validate = true` is half the assertion: the module passes spirv-val.
    d, _ = compile_and_disasm(vs_split_halves!, tt; validate = true)
    @test occursin("OpVectorShuffle", d)
end
