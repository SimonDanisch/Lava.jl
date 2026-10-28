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
                # An untaken edge may have an out-of-range index. Trunk's
                # lowering clamps every speculative private load to its array.
                @test !occursin("phi ptr",text)
                @test occursin("phi i32",text)
                @test count("icmp slt i64 %index, 0",text) == 2
                @test count("icmp sgt i64",text) == 2
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
                called=parse(LLVM.Module,"declare void @mutate(ptr)\n" *
                    replace(ir,"%v = load"=>"call void @mutate(ptr %p)\n  %v = load"))
                Lava.lower_phi_select_function_ptrs!(called)
                LLVM.verify(called)
                @test occursin("phi ptr",string(called))
                @test !occursin("vmirror_leaf_load",string(called))
                # StructurizeCFG may separate a pointer phi from its load by
                # an unconditional edge. Reads can cross that edge, never a
                # write in either block.
                across_ir=replace(ir,"%v = load"=>"br label %read\nread:\n  %v = load")
                across=parse(LLVM.Module,across_ir)
                Lava.lower_phi_select_function_ptrs!(across)
                LLVM.verify(across)
                @test !occursin("phi ptr",string(across))
                @test occursin("phi i32",string(across))
                guarded_ir=replace(across_ir,"br label %read"=>
                    "br i1 %take, label %read, label %exit")
                guarded=parse(LLVM.Module,guarded_ir)
                Lava.lower_phi_select_function_ptrs!(guarded)
                LLVM.verify(guarded)
                @test !occursin("phi ptr",string(guarded))
                @test occursin("phi i32",string(guarded))
                guarded_write=parse(LLVM.Module,replace(guarded_ir,"%v = load"=>
                    "store i32 99, ptr %p\n  %v = load"))
                Lava.lower_phi_select_function_ptrs!(guarded_write)
                LLVM.verify(guarded_write)
                @test occursin("phi ptr",string(guarded_write))
                # A load in a separate loop can observe a write from its
                # previous iteration. It must not become one predecessor
                # read reused for every iteration (the final result is 99).
                loop_ir=replace(across_ir,
                    "read:\n"=>"read:\n  %iteration = phi i32 [0, %join], [%next, %read]\n",
                    "ret i32 %v"=>"%next = add i32 %iteration, 1\n  store i32 99, ptr %p\n  %again = icmp ult i32 %next, 2\n  br i1 %again, label %read, label %exit",
                    "ret i32 0"=>"ret i32 %v")
                looped=parse(LLVM.Module,loop_ir)
                Lava.lower_phi_select_function_ptrs!(looped)
                LLVM.verify(looped)
                @test occursin("phi ptr",string(looped))
                @test !occursin("vmirror_leaf_load",string(looped))
                for location in ("br label %read", "%v = load")
                    changed=replace(across_ir,location=>"store i32 99, ptr %p\n  " * location)
                    written=parse(LLVM.Module,changed)
                    Lava.lower_phi_select_function_ptrs!(written)
                    LLVM.verify(written)
                    @test occursin("phi ptr",string(written))
                    @test !occursin("vmirror_leaf_load",string(written))
                end
                volatile=parse(LLVM.Module,replace(ir,"%v = load"=>"%v = load volatile"))
                Lava.lower_phi_select_function_ptrs!(volatile)
                LLVM.verify(volatile)
                @test occursin("load volatile i32",string(volatile))
                @test !occursin("vmirror_leaf_load",string(volatile))
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
