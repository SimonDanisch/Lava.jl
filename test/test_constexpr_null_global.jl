# LLVM retains these guards after Julia's constant NTuple indices become a
# private lookup table. This is a compiler-only regression: no device setup.
using Test,Lava,LLVM

@testset "cast-null/private-table constant comparisons fold, unknown pointers refuse" begin
    LLVM.Context() do ctx
        for (predicate,expected) in (("ne",true),("eq",false))
            text="""
            @table = private unnamed_addr addrspace(1) constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
            define void @main() {
            entry:
              br i1 icmp $predicate (ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1)), ptr addrspace(1) @table), label %yes, label %join
            yes:
              br label %join
            join:
              ret void
            }
            """
            mod=parse(LLVM.Module,text)
            branch=first(LLVM.instructions(first(LLVM.blocks(mod.functions["main"]))))
            comparison=first(branch.operands)
            @test comparison isa LLVM.ConstantExpr
            @test Lava.constexpr_pointer_comparison(comparison)===expected
            bytes,_=Lava.emit_spirv_from_llvm(mod,"main",(1,1,1))
            @test Lava.validate_spirv(bytes)===nothing
        end
        # An extern-weak address can resolve to null: never fold it as an
        # allocated local table, and never translate unknown comparisons to a
        # guessed boolean or physical host address.
        mod=parse(LLVM.Module,"""
        @table = extern_weak addrspace(1) constant [4 x i32]
        define void @main() {
        entry:
          br i1 icmp ne (ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1)), ptr addrspace(1) @table), label %yes, label %join
        yes:
          br label %join
        join:
          ret void
        }
        """)
        comparison=first(first(LLVM.instructions(first(LLVM.blocks(mod.functions["main"])))).operands)
        @test Lava.constexpr_pointer_comparison(comparison)===nothing
    end
end
