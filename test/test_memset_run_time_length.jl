# A memset of a length known only at run time becomes a loop.
#
# LLVM's loop-idiom pass turns a loop that zeroes an array of a run-time length into
# `llvm.memset` with that length, and `lower_memset!` stored only constant lengths: a
# kernel zeroing the counters of Hikari's material queues (one per material slot)
# failed to compile with "Cannot lower memset with non-constant length".
#
# Not a GPU test: the pass on LLVM IR. On the device, Hikari's `test_workqueue.jl`
# ("the counter reset zeroes every counter of an array").

using Test
using Lava
using LLVM

const _IR_MEMSET_N = """
declare void @llvm.memset.p1.i64(ptr addrspace(1) nocapture writeonly, i8, i64, i1 immarg)

define void @main(ptr addrspace(1) %counts, i64 %n) {
entry:
  %bytes = shl i64 %n, 2
  call void @llvm.memset.p1.i64(ptr addrspace(1) align ALIGN %counts, i8 0, i64 %bytes, i1 false)
  ret void
}
"""

memset_ir(align) = replace(_IR_MEMSET_N, "ALIGN" => string(align))

"""The stores of `main`, and whether a block branches back to itself."""
function memset_loop_shape(mod)
    fn = mod.functions["main"]
    stores = [i for b in LLVM.blocks(fn) for i in LLVM.instructions(b) if i isa LLVM.StoreInst]
    loops = any(b -> b in LLVM.successors(LLVM.terminator(b)), LLVM.blocks(fn))
    return stores, loops
end

@testset "a memset of a run-time length is a loop of stores" begin
    # Four-aligned and four times a count: words.
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, memset_ir(4))
        Lava.lower_unsupported_intrinsics!(mod)
        stores, loops = memset_loop_shape(mod)
        @test loops
        @test length(stores) == 1
        @test only(stores).operands[1].value_type == LLVM.Int32Type()
        @test LLVM.alignment(only(stores)) == 4
        @test !occursin("llvm.memset", string(mod))
        @test LLVM.verify(mod) === nothing
    end
    # Byte-aligned: bytes, whatever the length.
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, memset_ir(1))
        Lava.lower_unsupported_intrinsics!(mod)
        stores, loops = memset_loop_shape(mod)
        @test loops
        @test only(stores).operands[1].value_type == LLVM.Int8Type()
        @test LLVM.verify(mod) === nothing
    end
end
