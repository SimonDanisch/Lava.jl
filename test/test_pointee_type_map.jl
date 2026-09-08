# Unit test for the pointee-type agreement between `infer_inner_ptr_pointee`
# and the `PointeeTypeMap` (PTM).
#
# What broke: an `inttoptr` whose result points to a pointer gets its declared
# pointee from `infer_inner_ptr_pointee`, while the `OpLoad` consuming it gets
# its result type from the PTM (via `emit_load!` → `map_pointer_type_for_value!`).
# The two traversed the uses in different orders — inference follows the use
# list (reverse creation order, so a byte-offset GEP feeding a float field load
# can win) while the PTM is built in instruction scan order (so a direct
# `load i32` at offset 0 sets the entry). When they disagreed, the `OpLoad`
# result type mismatched the pointer's declared pointee type and emission
# failed — first seen in Hikari's camera kernel, whose argument area holds
# exactly this shape (a loaded pointer read both as a struct field and as a
# plain i32).
#
# The fix: `infer_inner_ptr_pointee` takes the PTM and prefers the consuming
# load's PTM entry, so the two can no longer disagree. This file pins both
# halves: that the IR below really does produce the disagreement without the
# PTM, and that with it the answer is the PTM's.

using Test
using Lava
using LLVM

# The order of the last three instructions is the whole setup: the i32 load is
# parsed FIRST (so the PTM, built in scan order, records i32 for `%q`), and the
# byte-offset GEP is parsed LAST (so it heads the use list and use-based
# inference follows it to the float field). Swapping them makes the two agree
# and the test proves nothing.
const _IR_PTM_DISAGREE = """
define void @k(i64 %raw) {
entry:
  %p = inttoptr i64 %raw to ptr
  %q = load ptr, ptr %p
  %d = load i32, ptr %q
  %g = getelementptr i8, ptr %q, i64 4
  %f = load float, ptr %g
  ret void
}
"""

@testset "infer_inner_ptr_pointee agrees with the PointeeTypeMap" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_PTM_DISAGREE)
        @test (LLVM.verify(mod); true)
        insts = collect(LLVM.instructions(first(LLVM.blocks(LLVM.functions(mod)["k"]))))
        itp, q = insts[1], insts[2]
        itp isa LLVM.IntToPtrInst || error("test IR drifted: first instruction is $itp")
        q isa LLVM.LoadInst || error("test IR drifted: second instruction is $q")

        ptm = Lava.build_pointee_type_map(mod)

        # The PTM's entry for the consuming load is i32 — the type `emit_load!`
        # will give its result.
        @test Lava.get_pointee_type(ptm, q) == LLVM.IntType(32)

        # Without the PTM, use-based inference follows the byte-offset GEP and
        # answers float — the disagreement that broke emission.
        @test Lava.infer_inner_ptr_pointee(itp) == LLVM.FloatType()

        # With it, the answer is the PTM's, so the pointer's declared pointee
        # and the load's result type are the same type.
        @test Lava.infer_inner_ptr_pointee(itp, ptm) == LLVM.IntType(32)
    end
end
