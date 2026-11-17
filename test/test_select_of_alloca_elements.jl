# A load through a select between elements of two allocas is a select between two loads.
#
# Hikari's node interpreter reads an input as `o == NO_SLOT ? ins.k[i] : stack[o]`,
# and LLVM makes that one load through a select between two element pointers: one
# into the instruction's private copy (`[16 x i32]`), one into the stack, an `MArray`
# alloca of i64 slots. SPIR-V cannot select between those: the two pointers have
# different types and a Function pointer cannot be bitcast, so the module failed
# validation ("Expected both objects to be of Result Type: Select").
# `lower_phi_select_function_ptrs!` loaded at whole allocas only; it now loads at
# element pointers too, with their indices clamped into the alloca, since the element
# not picked may be out of bounds: at `o == NO_SLOT` the stack element is at byte -4.
#
# Not a GPU test: the pass, emission and spirv-val.

using Test
using Lava
using LLVM

const _IR_SELECT_OF_ALLOCA_ELEMENTS = """
@lava_shared_out = internal addrspace(3) global [1 x float] undef, align 4
@__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>

define void @main() {
entry:
  %stack = alloca [2 x i64], align 8
  %ins = alloca [16 x i32], align 4
  store [2 x i64] zeroinitializer, ptr %stack, align 8
  store [16 x i32] zeroinitializer, ptr %ins, align 4
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
  %x = extractelement <3 x i32> %v, i64 0
  %o = trunc i32 %x to i8
  %const = icmp eq i8 %o, 0
  %o64 = zext i8 %o to i64
  %shifted = shl nuw nsw i64 %o64, 2
  %offset = add i64 %shifted, -4
  %slot = getelementptr i8, ptr %stack, i64 %offset
  %k = getelementptr i8, ptr %ins, i64 28
  %p = select i1 %const, ptr %k, ptr %slot
  %value = load float, ptr %p, align 4
  store float %value, ptr addrspace(3) @lava_shared_out, align 4
  ret void
}
"""

@testset "a load through a select between alloca elements is a select between loads" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_SELECT_OF_ALLOCA_ELEMENTS)
        Lava.lower_phi_select_function_ptrs!(mod)
        ir = string(mod)
        @test !occursin(r"select i1 [^,]+, ptr", ir)     # no pointer select is left
        @test occursin("vmirror_sel", ir)                # the loads are selected instead
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end
