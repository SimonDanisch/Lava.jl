# Tier 1: a private array whose element is a VECTOR narrower than the slot Julia
# allocates it in — the sibling of `test_packed_private_scalar.jl`, and a
# different pair of defects.
#
# `MVector{10, NTuple{2,VecElement{Float16}}}` is 40 bytes and Julia allocas it
# as `[5 x i64]`, so an index the compiler cannot fold turns the access into
# read-modify-write on a 64-bit word. That is the same packed path the scalar
# file covers; what is new is that the thing being packed is `<2 x half>`.
#
# **`llvm_type_size` had no `VectorType` branch.** It fell through to the
# "assume 8 bytes (pointer-sized)" default, so `<2 x half>` measured the same
# width as the `i64` it is packed into. `convert_typepunned_geps_to_byte_geps!`
# is gated on the access being STRICTLY narrower than the alloca element, so it
# skipped the GEP; `lower_byte_gep_chain_on_allocas!` only takes `i8` GEPs and
# never saw it either. A typed, dynamically indexed GEP then reached the
# emitter, which has no lowering for one, and it emitted the store with the
# access chain simply missing:
#
#     OpStore Pointer's type does not match Object's type
#
# with the pointer being the `[N x i64]` variable itself. `scalar_size`, a few
# lines below it in the same file, has always had the branch.
#
# **`emit_trunc!` then assumed the unpacked value is a scalar float.** Reading an
# element back truncates the word to `i32` and bitcasts it to `<2 x half>`; the
# trunc+bitcast fold built its destination as `emit_type_float!` of
# `scalar_bit_width(dst)`, and `scalar_bit_width` answers `nothing` for a vector,
# so it reached `UInt32(nothing)`. Mapping the bitcast's own destination type
# covers half, float, double and vectors of them alike.
#
# Found by Mantle's cooperative-matrix GEMM: its `nextA` register queue is an
# MVector of `f16vec2`, and it only ever compiled because the staging loop
# unrolled and folded every index to a constant. Putting a gathering load in
# that loop stopped the unroll, the indices became dynamic, and both defects
# appeared at once. Whether a loop unrolls is LLVM's decision and not something
# a lowering may depend on, which is why this file indexes dynamically on
# purpose instead of waiting for an optimiser to decline.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, compile_and_disasm

using StaticArrays

# Ten `f16vec2`s: 40 bytes, which Julia packs two-per-word into `[5 x i64]`, and
# past the point where the scalariser keeps each element in its own SSA value.
# `gid % 10` is opaque to the compiler, which is the whole point — a constant
# index folds and never reaches the code under test.
#
# The STORES here are constant-indexed on purpose. A dynamic store index looks
# like the stronger test and is a weaker one: written as a loop over
# `(gid + r) % 10` the stores are a permutation, LLVM inverts it, and the array
# is gone into SSA values before any of this runs — measured, the compiled
# module then had no Function-storage variable at all. Both fixes are about the
# WIDTH of `<2 x half>`, and the dynamic read reaches both: the typed GEP that
# `convert_typepunned_geps_to_byte_geps!` used to skip, and the trunc+bitcast
# that unpacks the word.
function packed_private_vec2(A)
    stage = MVector{10, NTuple{2,VecElement{Float16}}}(undef)
    gid = Int(Lava.lava_global_invocation_id_x())
    for r in 1:10
        @inbounds stage[r] = (VecElement(A[gid + 2r - 1]), VecElement(A[gid + 2r]))
    end
    ptr = Lava.lava_alloc_shared(Val(:packed_vec2_stage), Float16, Val(64))
    shared = Lava.LavaSharedArray{Float16}(ptr, 64)
    @inbounds v = stage[1 + (gid % 10)]
    @inbounds shared[1 + (gid % 64)] = v[1].value + v[2].value
    Lava.lava_workgroup_barrier()
    @inbounds A[gid + 1] = shared[1 + (gid % 64)]
    return nothing
end

# `<4 x half>` in the same slot: one element per word rather than two, so the
# packing has no shift and only the width matters. It is here because the fix is
# in `llvm_type_size`, which is a width computation and not a two-per-word one.
function packed_private_vec4(A)
    stage = MVector{10, NTuple{4,VecElement{Float16}}}(undef)
    gid = Int(Lava.lava_global_invocation_id_x())
    for r in 1:10
        @inbounds stage[r] =
            (VecElement(A[gid + 4r - 3]), VecElement(A[gid + 4r - 2]),
             VecElement(A[gid + 4r - 1]), VecElement(A[gid + 4r]))
    end
    ptr = Lava.lava_alloc_shared(Val(:packed_vec4_stage), Float16, Val(64))
    shared = Lava.LavaSharedArray{Float16}(ptr, 64)
    @inbounds v = stage[1 + (gid % 10)]
    @inbounds shared[1 + (gid % 64)] = v[1].value + v[4].value
    Lava.lava_workgroup_barrier()
    @inbounds A[gid + 1] = shared[1 + (gid % 64)]
    return nothing
end

@testset "Packed private vector" begin
    # The root cause on its own, because it is one line and every symptom above
    # is downstream of it. A vector's size is its length times its element's,
    # exactly as for an array.
    @testset "llvm_type_size measures a vector" begin
        LLVM.@dispose ctx = LLVM.Context() begin
            @test Lava.llvm_type_size(LLVM.VectorType(LLVM.HalfType(), 2)) == 4
            @test Lava.llvm_type_size(LLVM.VectorType(LLVM.HalfType(), 4)) == 8
            @test Lava.llvm_type_size(LLVM.VectorType(LLVM.FloatType(), 4)) == 16
            @test Lava.llvm_type_size(LLVM.VectorType(LLVM.IntType(32), 2)) == 8
            # The default this used to fall through to, so a regression that
            # merely restores it cannot pass the cases above by accident.
            @test Lava.llvm_type_size(LLVM.VectorType(LLVM.HalfType(), 2)) !=
                  Lava.llvm_type_size(LLVM.Int64Type())
        end
    end

    # `compile_and_disasm` validates, so the dropped access chain fails here on
    # its own and the checks say the lowering is the RIGHT one rather than
    # merely well-typed.
    @testset "f16vec2 element in an i64 slot" begin
        d, _ = compile_and_disasm(packed_private_vec2,
                                   Tuple{Lava.LavaDeviceArray{Float16,1}})
        # The packed slot is what makes this test the test it is: if Julia ever
        # allocas the MVector as `[10 x <2 x half>]`, the path under test is gone
        # and everything below would pass vacuously.
        check(d, "_ptr_Function_ulong")
        # The access chain the emitter used to drop, and the unpack that used to
        # ask `emit_type_float!` for a vector's width.
        check(d, "OpAccessChain %_ptr_Function_ulong")
        check(d, "OpBitcast %v2half")
        # The defect, spelled out: the value went to the array variable itself
        # rather than through an access chain to one of its words.
        check_not(d, "OpStore %_ptr_Function__arr_ulong")
    end

    @testset "f16vec4 element in an i64 slot" begin
        d, _ = compile_and_disasm(packed_private_vec4,
                                   Tuple{Lava.LavaDeviceArray{Float16,1}})
        check(d, "_ptr_Function_ulong")
        check(d, "OpBitcast %v4half")
    end
end
