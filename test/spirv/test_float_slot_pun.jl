# Tier 1: a narrow load out of a FLOATING-POINT slot at offset zero.
#
# A struct of three `Float64` passed by value lands in a `[3 x double]` alloca,
# and storing it into a device array whose element is only 4-byte aligned is
# lowered as six `i32` chunk copies out of that alloca. Five of the six chunks
# sit at a nonzero offset, reach `decompose_typepun_gep_loads!` as byte GEPs, and
# become `load double` + bitcast + shift + trunc. The chunk at offset zero has
# no GEP at all — `gep i8, ptr %a, 0` folds to `%a` — and the final
# `lift_byte_geps_on_allocas!` gives it `gep i32, ptr %alloca, 0`. Read by its
# own source type, that GEP looked like an `i32` field, the load like no pun, and
# the emitter wrote
#
#     OpLoad %uint %arr_double_3
#
# which spirv-val rejects. `lower_constant_subelement_access!` handles the same
# offset-zero shape but only for INTEGER slots, so a double array fell between
# the two. Found through `fill!` on a `LavaArray` of such a struct.
#
# The WIDER half of the same misreading: a `ComplexF32` that SROA left in a
# `[2 x float]` alloca, because its value comes from a phi of two such allocas
# (the loop below may run zero times), is copied out whole. The pointer phi is
# lowered by `lower_phi_select_function_ptrs!` into a load per leaf, which reads
# the complex as `load i64, ptr (gep i64, ptr %alloca, 0)`: two floats, which the
# pass sized by the GEP's `i64` and left alone. GPUArrays' `generic_trimatmul!`
# for a complex matrix is the kernel this comes from, and it failed spirv-val.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

struct PunThreeF64
    a::Float64; b::Float64; c::Float64
end

struct PunTwoF64
    a::Float64; b::Float64
end

function pun_fill!(a, v)
    i = KernelInterface.get_global_id().x
    i <= length(a) || return nothing
    @inbounds a[i] = v
    return nothing
end

function pun_trimatvec!(C, A, B, upper::Bool, unit::Bool, oA::ComplexF32)
    i = KernelInterface.get_global_id().x
    i <= length(C) || return nothing
    m = length(B)
    @inbounds begin
        Cij = zero(ComplexF32)
        Cij = muladd(unit ? oA : A[i, i], B[i], Cij)
        for k in (upper ? (i + 1) : 1):(upper ? m : (i - 1))
            Cij = muladd(A[i, k], B[k], Cij)
        end
        C[i] = Cij
    end
    return nothing
end

@testset "a narrow load at offset zero of a floating-point slot" begin
    for T in (PunThreeF64, PunTwoF64)
        # `validate = true` is the assertion: before the fix spirv-val threw on
        # `OpLoad Result Type %uint does not match Pointer`.
        d, _ = compile_and_disasm(pun_fill!, Tuple{Lava.LavaDeviceArray{T,1}, T};
                                  validate = true)
        @test occursin("OpBitcast", d)
    end
end

@testset "a load across two floating-point slots" begin
    tt = Tuple{Lava.LavaDeviceArray{ComplexF32,1}, Lava.LavaDeviceArray{ComplexF32,2},
               Lava.LavaDeviceArray{ComplexF32,1}, Bool, Bool, ComplexF32}
    # `validate = true` is the assertion: before the fix spirv-val threw on
    # `OpLoad Result Type %ulong does not match Pointer`.
    d, _ = compile_and_disasm(pun_trimatvec!, tt; validate = true)
    @test occursin("OpBitcast", d)
end
