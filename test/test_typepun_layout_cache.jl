using Lava, LLVM, Test

@testset "Typepun field layouts follow each module's DataLayout" begin
    # A word at byte four is padding with 64-bit alignment, and the low word of
    # the i64 field with 32-bit alignment. The literal LLVM struct type is shared
    # across the modules; a layout cache must not outlive one pass invocation.
    LLVM.Context() do ctx
        types = LLVM.LLVMType[]
        for alignment in (64, 32, 64)
            mod = parse(
                LLVM.Module, """
                target datalayout = "e-p:64:64-i64:$alignment"
                define i32 @read_word() {
                entry:
                  %a = alloca { i8, i64 }
                  %p = getelementptr i8, ptr %a, i64 4
                  %v = load i32, ptr %p
                  ret i32 %v
                }
                """
            )
            try
                push!(types, LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(first(first(mod.functions["read_word"].blocks).instructions))))
                Lava.decompose_typepun_gep_loads!(mod, mod.datalayout)
                @test LLVM.verify(mod) === nothing
                @test occursin("typepun_load", string(mod)) == (alignment == 32)
            finally
                LLVM.dispose(mod)
            end
        end
        @test all(==(first(types)), types)
    end
end
