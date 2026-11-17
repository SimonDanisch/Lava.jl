# A constant expression that is a phi's operand is emitted inside a block.
#
# KernelAbstractions' clamped read of shared memory,
#     r = lid == 128 ? sh[lid] : sh[lid + 1]
# becomes, once `!range` on the local id (532e128) lets LLVM substitute
# `lid = 128` in that arm, a phi over workgroup pointers whose operand on one edge
# is the constant `getelementptr(@sh, 0, 127)`. Phis are resolved after every block
# is emitted, and the emitter materialized the constant where it was asked for:
# after the function's last `OpReturn`, which spirv-val rejects ("AccessChain must
# appear in a block"). It now goes into the edge's predecessor, before its
# terminator. Mantle's `test_select_width_mismatch.jl` and
# `test_logical_pointer_bitcast.jl` run the kernel on a device; the fold depends on
# KernelAbstractions' index arithmetic, so the shape is pinned here as IR, in the
# structured form Lava hands the emitter: the constant arrives on the edge from a
# block that ends in a selection header, so it goes in before the merge.
#
# Not a GPU test: emission and spirv-val only.

using Test
using Lava
using LLVM

const _IR_CONSTEXPR_PHI = """
@lava_shared_sh = internal addrspace(3) global [128 x float] undef, align 4
@__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>

define void @main() {
entry:
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
  %x = extractelement <3 x i32> %v, i64 0
  %c = icmp ne i32 %x, 127
  br i1 %c, label %other, label %join
other:
  %i = zext i32 %x to i64
  %p = getelementptr inbounds float, ptr addrspace(3) @lava_shared_sh, i64 %i
  br label %join
join:
  %q = phi ptr addrspace(3) [ %p, %other ], [ getelementptr inbounds ([128 x float], ptr addrspace(3) @lava_shared_sh, i64 0, i64 127), %entry ]
  store float 1.0, ptr addrspace(3) %q, align 4
  ret void
}
"""

@testset "a constant expression as a phi operand is emitted inside a block" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_CONSTEXPR_PHI)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (128, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end
