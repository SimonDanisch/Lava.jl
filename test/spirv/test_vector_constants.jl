# Tier 1: two vector shapes an int8 GEMM kernel met in one afternoon.
#
#   * A constant VECTOR operand — `fsub <2 x half> %x, <half 1152, half 1152>`,
#     the packed subtract that turns 0x6400|u into an int8 — reached the emitter
#     as `LLVM.ConstantDataVector` and stopped at "not in value map and not a
#     constant". It is an `OpConstantComposite`, like a constant array.
#   * A store of `<2 x i32>` into a workgroup array of `<4 x half>`: InstCombine
#     folds the `bitcast` that made the value into the store. The emitter wrote
#     the mismatched `OpStore` and spirv-val refused it. The value is bitcast
#     back; a logical pointer cannot be retyped.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

const _VC_H2 = NTuple{2,VecElement{Float16}}
const _VC_H4 = NTuple{4,VecElement{Float16}}
const _VC_U4 = NTuple{4,VecElement{UInt32}}

@inline _vc_sub(x::_VC_H2) = Base.llvmcall("""
    %r = fsub <2 x half> %0, <half 0xH6480, half 0xH6480>
    ret <2 x half> %r""", _VC_H2, Tuple{_VC_H2}, x)
@inline _vc_lo(x::_VC_U4) = Base.llvmcall("""
    %s = shufflevector <4 x i32> %0, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
    %r = bitcast <2 x i32> %s to <4 x half>
    ret <4 x half> %r""", _VC_H4, Tuple{_VC_U4}, x)

function vc_constant_vector!(dst, src)
    i = Int(KernelInterface.get_global_id().x)
    x = Core.Intrinsics.pointerref(reinterpret(Ptr{_VC_H2}, pointer(src)), i, 4)
    Core.Intrinsics.pointerset(reinterpret(Ptr{_VC_H2}, pointer(dst)), _vc_sub(x), i, 4)
    return nothing
end

# Read back as halves, with arithmetic on them: a plain copy lets InstCombine
# retype the load as well, and then nothing disagrees about the element type.
@inline _vc_double(x::_VC_H4) = Base.llvmcall("""
    %r = fadd <4 x half> %0, %0
    ret <4 x half> %r""", _VC_H4, Tuple{_VC_H4}, x)

function vc_store_bitcast!(dst, src)
    sh = KernelInterface.localmemory(_VC_H4, Val((64,)), Val(1))
    t = Int(KernelInterface.get_local_id().x)
    x = Core.Intrinsics.pointerref(reinterpret(Ptr{_VC_U4}, pointer(src)), t, 16)
    @inbounds sh[t] = _vc_lo(x)
    KernelInterface.barrier()
    @inbounds v = _vc_double(sh[65 - t])
    Core.Intrinsics.pointerset(reinterpret(Ptr{_VC_H4}, pointer(dst)), v, t, 8)
    return nothing
end

@testset "vector constants and same-width vector stores" begin
    tt = Tuple{Lava.LavaDeviceArray{Float16,1},Lava.LavaDeviceArray{Float16,1}}
    # `validate = true` is half of each assertion: the module passes spirv-val.
    d, _ = compile_and_disasm(vc_constant_vector!, tt; validate = true)
    @test occursin(r"OpConstantComposite %v2half", d)
    d, _ = compile_and_disasm(vc_store_bitcast!, tt; validate = true)
    @test occursin(r"OpBitcast %v4half", d)
end
