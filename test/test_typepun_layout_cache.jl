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

@testset "Typepun rewrites retain interleaved accesses" begin
    LLVM.Context() do ctx
        # Each wide access causes an iterator restart. Earlier correctly typed
        # loads/stores stay valid while all later wide accesses must be lowered.
        body = IOBuffer()
        println(body, "target datalayout = \"e-p:64:64-i64:64\"")
        println(body, "define void @interleaved() {\nentry:\n %a = alloca [64 x float]")
        for i in 0:15
            println(body, " %p$i = getelementptr [64 x float], ptr %a, i32 0, i32 $(4i)")
            println(body, " store float 1.0, ptr %p$i\n %f$i = load float, ptr %p$i")
            println(body, " store i64 0, ptr %p$i\n %v$i = load i64, ptr %p$i")
        end
        println(body, " ret void\n}")
        mod = parse(LLVM.Module, String(take!(body)))
        try
            Lava.decompose_typepun_gep_loads!(mod, mod.datalayout)
            @test LLVM.verify(mod) === nothing
            lowered = string(mod)
            @test !occursin("load i64", lowered)
            @test !occursin("store i64", lowered)
            @test count("load float", lowered) == 48
            @test count("store float", lowered) == 48
            Lava.decompose_typepun_gep_loads!(mod, mod.datalayout)
            @test string(mod) == lowered
        finally
            LLVM.dispose(mod)
        end
    end
end
