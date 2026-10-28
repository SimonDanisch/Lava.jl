# Tier 1: a private array whose element type is NARROWER than the slot Julia
# allocates it in.
#
# `MVector{10,Float16}` is 20 bytes and Julia allocas it as `[3 x i64]`, so an
# index the compiler cannot fold turns every access into read-modify-write on a
# 64-bit word: `lower_variable_idx_access!` → `emit_element_rmw_access!`. That
# packing is arithmetic on integers, and the value being packed is not one.
#
# `zext` and `trunc` are integer-only in LLVM — but nothing in the C builder API
# enforces it. `LLVM.zext!(builder, <half>, i64)` returns `zext half %x to i64`
# and `LLVM.trunc!(builder, <i64>, half)` returns `trunc i64 %x to half`; both
# verify nowhere, survive every later pass, and arrive at the SPIR-V emitter as
# conversions with a float on one end. It lowered them to a bare `OpUConvert
# %uint`, so the store that followed put a 32-bit integer through a
# `_ptr_Workgroup_half` and spirv-val reported
#
#     OpStore Pointer's type does not match Object's type
#
# which is what a `@private Float16` staging array in DNNKernels' flash
# attention hit at `BR = 64, BC = 64`, `E = 72` — ten elements a thread, one
# more than the nine the scalariser folds.
#
# The fix is in two places and this pins both: the pass bitcasts to an integer
# of the same width before packing and back after unpacking, and `emit_trunc!`
# sizes its float from `scalar_bit_width` rather than from a two-way
# `LLVMFloat ? 32 : 64` test that has no answer for `half`.
#
# Since 2026-10-23 these kernels no longer take that path: `retype_allocas.jl`
# decomposes a wider access through integers for float elements too, so the
# MVector becomes a typed `[12 x half]` private array with no packing at all.
# The kernels below check that; the packed path, which allocas the retype pass
# leaves alone still take, is pinned as IR in the last testset.

using Test
using LLVM
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, compile_and_disasm

using StaticArrays

# Ten `Float16`s: past the point where the scalariser keeps each element in its
# own SSA value, so the dynamic read below lands on the packed path. `gid % 10`
# and `gid % 64` are opaque to the compiler, which is the whole point — a
# constant index folds and never reaches the code under test.
function packed_private_half(A)
    stage = MVector{10, Float16}(undef)
    gid = Int(Lava.lava_global_invocation_id_x())
    for r in 1:10
        @inbounds stage[r] = A[gid + r]
    end
    ptr = Lava.lava_alloc_shared(Val(:packed_half_stage), Float16, Val(64))
    shared = Lava.LavaSharedArray{Float16}(ptr, 64)
    @inbounds shared[1 + (gid % 64)] = stage[1 + (gid % 10)]
    Lava.lava_workgroup_barrier()
    @inbounds A[gid + 1] = shared[1 + (gid % 64)]
    return nothing
end

# Same shape in fp32 against an `[N x i64]` slot: two per word rather than four.
# This one VALIDATED before the fix, and only by coincidence — the emitter's
# hardcoded 32-bit narrowing happens to be the right width for a `float`, so the
# bitcast it produced was well-typed. It is here because it shares every line of
# the fix with the `half` case above and would be the thing to break if the fix
# were later narrowed to 16 bits.
function packed_private_float(A)
    stage = MVector{10, Float32}(undef)
    gid = Int(Lava.lava_global_invocation_id_x())
    for r in 1:10
        @inbounds stage[r] = A[gid + r]
    end
    ptr = Lava.lava_alloc_shared(Val(:packed_float_stage), Float32, Val(64))
    shared = Lava.LavaSharedArray{Float32}(ptr, 64)
    @inbounds shared[1 + (gid % 64)] = stage[1 + (gid % 10)]
    Lava.lava_workgroup_barrier()
    @inbounds A[gid + 1] = shared[1 + (gid % 64)]
    return nothing
end

# What the packed path is handed: an i64-slotted alloca read and written as
# `half` at an index the compiler cannot fold.
const _IR_PACKED_HALF = """
define void @main(ptr addrspace(1) %out, i64 %k, i64 %j, half %v) {
entry:
  %a = alloca [3 x i64], align 16
  %base = getelementptr half, ptr %a, i64 0
  %p = getelementptr half, ptr %base, i64 %k
  store half %v, ptr %p, align 2
  %q = getelementptr half, ptr %base, i64 %j
  %x = load half, ptr %q, align 2
  store half %x, ptr addrspace(1) %out, align 2
  ret void
}
"""

@testset "Packed private scalar" begin
    # `compile_and_disasm` validates, so the OpStore mismatch fails here on its
    # own. The private array is typed by its element: nothing is packed.
    @testset "Float16 element in an i64 slot" begin
        d, _ = compile_and_disasm(packed_private_half,
                                   Tuple{Lava.LavaDeviceArray{Float16,1}})
        check(d, "_ptr_Function__arr_half_uint_12")
        check_not(d, "_ptr_Function_ulong")
        # The defect, spelled out: a 32-bit integer where a 16-bit float goes.
        check_not(d, "OpStore %_ptr_Workgroup_half")
    end

    @testset "Float32 element in an i64 slot" begin
        d, _ = compile_and_disasm(packed_private_float,
                                   Tuple{Lava.LavaDeviceArray{Float32,1}})
        check(d, "_ptr_Function__arr_float_uint_10")
        check_not(d, "_ptr_Function_ulong")
    end

    # The narrowing is the RIGHT one rather than merely a well-typed one: the
    # half goes into and out of its 64-bit word as 16 bits, bitcast at either
    # end, and no integer conversion has a float on one side.
    @testset "the packed path, as IR" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_PACKED_HALF)
            Lava.lower_chained_mismatched_geps!(mod)
            @test LLVM.verify(mod) === nothing
            ir = string(mod)
            @test occursin("bitcast half %v to i16", ir)
            @test occursin(r"bitcast i16 %\w+ to half", ir)
            @test occursin(r"load i64, ptr %\w+", ir)
            @test !occursin(r"zext half|trunc i64 %\w+ to half", ir)
        end
    end
end
