# A constant struct as an instruction's operand is emitted as a constant.
#
# Hikari's shader materials address a tuple of closures through `ifelse` chains, and
# LLVM keeps such a select over whole structs: the closure is built by
# `insertvalue` from a constant struct, its known fields constants and the rest
# poison, and selected against another. `get_value_id!` made constants of scalars,
# arrays and vectors but not of structs, so every shader material failed with "LLVM
# value not in value map and not a constant: LLVM.ConstantStruct". It is now the
# `OpConstantComposite` of its members. The shape depends on how much LLVM sees at
# once (a small kernel gets per-field selects), so it is pinned here as IR.
#
# Not a GPU test: emission and spirv-val only.

using Test
using Lava
using LLVM

const _IR_CONSTANT_STRUCT_OPERAND = """
@lava_shared_out = internal addrspace(3) global [1 x float] undef, align 4
@__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>

define void @main() {
entry:
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
  %x = extractelement <3 x i32> %v, i64 0
  %f = uitofp i32 %x to float
  %a = insertvalue { i8, i8, [1 x [3 x float]], float } { i8 3, i8 0, [1 x [3 x float]] poison, float poison }, float %f, 2, 0, 0
  %b = insertvalue { i8, i8, [1 x [3 x float]], float } %a, float 1.0, 3
  %c = icmp ne i32 %x, 7
  %s = select i1 %c, { i8, i8, [1 x [3 x float]], float } %b, { i8, i8, [1 x [3 x float]], float } { i8 5, i8 3, [1 x [3 x float]] zeroinitializer, float 2.0 }
  %w = extractvalue { i8, i8, [1 x [3 x float]], float } %s, 2, 0, 0
  %t = extractvalue { i8, i8, [1 x [3 x float]], float } %s, 3
  %l = extractvalue { i8, i8, [1 x [3 x float]], float } %s, 0
  %lf = uitofp i8 %l to float
  %sum = fadd float %w, %t
  %sum2 = fadd float %sum, %lf
  store float %sum2, ptr addrspace(3) @lava_shared_out, align 4
  ret void
}
"""

@testset "a constant struct operand is emitted as a constant composite" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_CONSTANT_STRUCT_OPERAND)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
        @test occursin("OpConstantComposite", Lava.disassemble_spirv(bytes))
    end
end
