# LLVM retains these guards after Julia's constant NTuple indices become a
# private lookup table. This is a compiler-only regression: no device setup.
#
# The comparison is an `icmp` constant expression up to LLVM 18 and an `icmp`
# instruction with constant operands from LLVM 19 on, which removed comparison
# constant expressions; both are decided by `Lava.constant_pointer_comparison`.
using Test,Lava,LLVM

const CMP_TABLE = "private unnamed_addr addrspace(1) constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]"
const CMP_NULL = "ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1))"
const HAS_ICMP_CONSTEXPR = LLVM.version() < v"19"

"`@main` branching on `a predicate b`, as an instruction or as a constant expression."
function comparison_module(declaration, predicate, a, b; constexpr::Bool)
    # An instruction names the operand type once, a constant expression per operand.
    branch = constexpr ?
        "br i1 icmp $predicate ($a, $b), label %yes, label %join" :
        "%c = icmp $predicate $a, $(replace(b, "ptr addrspace(1) " => ""; count = 1))\n  br i1 %c, label %yes, label %join"
    parse(LLVM.Module, """
    @table = $declaration
    define void @main() {
    entry:
      $branch
    yes:
      br label %join
    join:
      ret void
    }
    """)
end

"What Lava decides for the comparison `@main` branches on."
function decided(mod)
    branch = last(collect(LLVM.instructions(first(LLVM.blocks(mod.functions["main"])))))
    cmp = first(branch.operands)
    cmp isa LLVM.ConstantExpr && return Lava.constexpr_pointer_comparison(cmp)
    return Lava.constant_pointer_comparison(LLVM.API.LLVMGetICmpPredicate(cmp), cmp.operands...)
end

const CMP_FORMS = HAS_ICMP_CONSTEXPR ? (false, true) : (false,)

@testset "cast-null/private-table constant comparisons fold, unknown pointers refuse" begin
    LLVM.Context() do ctx
        for constexpr in CMP_FORMS
            for (predicate, expected) in (("ne", true), ("eq", false))
                mod = comparison_module(CMP_TABLE, predicate, CMP_NULL,
                                        "ptr addrspace(1) @table"; constexpr)
                @test decided(mod) === expected
                bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (1, 1, 1))
                @test Lava.validate_spirv(bytes) === nothing
            end
            for (predicate, expected) in (("ne", true), ("eq", false)), reverse in (false, true)
                table = "ptr addrspace(1) getelementptr inbounds ([4 x i32], ptr addrspace(1) @table, i64 0, i64 2)"
                a, b = reverse ? (CMP_NULL, table) : (table, CMP_NULL)
                mod = comparison_module(CMP_TABLE, predicate, a, b; constexpr)
                @test decided(mod) === expected
                bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (1, 1, 1))
                @test Lava.validate_spirv(bytes) === nothing
            end
            # An extern-weak address can resolve to null: never fold it as an
            # allocated local table, and never translate unknown comparisons to a
            # guessed boolean or physical host address.
            mod = comparison_module("extern_weak addrspace(1) constant [4 x i32]", "ne",
                                    CMP_NULL, "ptr addrspace(1) @table"; constexpr)
            @test decided(mod) === nothing
            for (declaration, gep) in (
                    ("extern_weak addrspace(1) constant [4 x i32]", "getelementptr inbounds ([4 x i32], ptr addrspace(1) @table, i64 0, i64 2)"),
                    ("private addrspace(1) constant [4 x i32] zeroinitializer", "getelementptr ([4 x i32], ptr addrspace(1) @table, i64 0, i64 -2)"))
                mod = comparison_module(declaration, "ne", "ptr addrspace(1) $gep", CMP_NULL; constexpr)
                @test decided(mod) === nothing
            end
        end
    end
end
