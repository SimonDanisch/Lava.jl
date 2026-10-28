# A dynamic NTuple index selected between two triangles left a Function
# pointer phi behind. Drivers read zero instead of the selected vertex ID.
using Test,Lava,LLVM

@testset "private GEP phi loads become edge-local scalar values" begin
    LLVM.Context() do ctx
        for conditional in (false,true)
            ir="""
            define i32 @choose(i1 %which, i64 %index, i1 %take) {
            entry:
              %a = alloca [3 x i32]
              %b = alloca [3 x i32]
              store [3 x i32] [i32 11, i32 12, i32 13], ptr %a
              store [3 x i32] [i32 21, i32 22, i32 23], ptr %b
              br i1 %which, label %left, label %right
            left:
              %pa = getelementptr [3 x i32], ptr %a, i64 0, i64 %index
              $(conditional ? "br i1 %take, label %join, label %exit" : "br label %join")
            right:
              %pb = getelementptr [3 x i32], ptr %b, i64 0, i64 %index
              br label %join
            join:
              %p = phi ptr [ %pa, %left ], [ %pb, %right ]
              %v = load i32, ptr %p
              ret i32 %v
            exit:
              ret i32 0
            }
            """
            mod=parse(LLVM.Module,ir)
            Lava.lower_phi_select_function_ptrs!(mod)
            LLVM.verify(mod)
            text=string(mod)
            if conditional
                @test occursin("phi ptr",text) # refuse speculative OOB load
                @test !occursin("vmirror_leaf_load",text)
            else
                @test !occursin("phi ptr",text)
                @test occursin("phi i32",text)
                @test count("vmirror_leaf_load = load i32",text)==1
                @test occursin("vmirror_leaf_load1 = load i32",text)
                # Do not hoist across a write in the join block.
                stored=parse(LLVM.Module,replace(ir,"%v = load"=>"store i32 99, ptr %p\n  %v = load"))
                Lava.lower_phi_select_function_ptrs!(stored)
                LLVM.verify(stored)
                @test occursin("phi ptr",string(stored))
                @test !occursin("vmirror_leaf_load",string(stored))
                # The extension is local-storage only, never BDA pointers.
                external=parse(LLVM.Module,replace(ir,"%a = alloca [3 x i32]"=>"%a = inttoptr i64 4096 to ptr"))
                Lava.lower_phi_select_function_ptrs!(external)
                LLVM.verify(external)
                @test occursin("phi ptr",string(external))
                @test !occursin("vmirror_leaf_load",string(external))
            end
        end
    end
end
