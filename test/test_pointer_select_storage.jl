# Opaque pointer selects must retain the storage class of local arrays.
using Test,Lava
const LLVM=Lava.LLVM

@testset "local and device pointer-select storage" begin
    LLVM.Context() do ctx
        mod=parse(LLVM.Module,"""
        define i32 @choose(i1 %cond, i64 %index, ptr %external) {
        entry:
          %local = alloca [4 x i32]
          %a = getelementptr [4 x i32], ptr %local, i64 0, i64 %index
          %b = getelementptr [4 x i32], ptr %local, i64 0, i64 0
          %selected = select i1 %cond, ptr %a, ptr %b
          %nested = select i1 %cond, ptr %selected, ptr %a
          %device = select i1 %cond, ptr %external, ptr %external
          %value = load i32, ptr %nested
          ret i32 %value
        }
        """)
        insts=collect(LLVM.instructions(first(LLVM.blocks(LLVM.functions(mod)["choose"]))))
        selects=filter(i->i isa LLVM.SelectInst,insts)
        @test length(selects)==3
        for ptr in selects[1:2]
            @test Lava.trace_pointer_to_alloca(ptr)
            @test !Lava.trace_to_non_alloca(ptr)
            @test !Lava.is_psb_pointer(ptr)
            @test Lava.get_pointer_storage_class(ptr)==Lava.SC.Function
        end
        ptr=selects[3]
        @test !Lava.trace_pointer_to_alloca(ptr)
        @test Lava.trace_to_non_alloca(ptr)
        @test Lava.is_psb_pointer(ptr)
        @test Lava.get_pointer_storage_class(ptr)==Lava.SC.PhysicalStorageBuffer
        Lava.fold_local_array_gep_selects!(mod)
        @test LLVM.verify(mod) === nothing
        folded=collect(LLVM.instructions(first(LLVM.blocks(LLVM.functions(mod)["choose"]))))
        # Keep one selected load; do not speculate either array access.
        @test count(i->i isa LLVM.LoadInst,folded)==1
        pointer_selects=filter(i->i isa LLVM.SelectInst && i.value_type isa LLVM.PointerType,folded)
        @test length(pointer_selects)==1
        @test only(pointer_selects).name=="device"
        @test count(i->i isa LLVM.SelectInst && i.value_type isa LLVM.IntegerType,folded)==2
    end
end


@testset "different local arrays select safe indices before loads" begin
    LLVM.Context() do ctx
        for volatile in (false,true)
            mod=parse(LLVM.Module,"""
            define i32 @choose(i1 %cond, i64 %index) {
            entry:
              %left = alloca [4 x i32]
              %right = alloca [4 x i32]
              store [4 x i32] [i32 1, i32 2, i32 3, i32 4], ptr %left
              store [4 x i32] [i32 5, i32 6, i32 7, i32 8], ptr %right
              %a = getelementptr [4 x i32], ptr %left, i64 0, i64 %index
              %b = getelementptr [4 x i32], ptr %right, i64 0, i64 0
              %selected = select i1 %cond, ptr %a, ptr %b
              %value = load $(volatile ? "volatile " : "")i32, ptr %selected
              ret i32 %value
            }
            """)
            Lava.lower_local_array_select_loads!(mod)
            @test LLVM.verify(mod) === nothing
            text=string(mod)
            if volatile
                @test occursin("select i1 %cond, ptr",text)
                @test !occursin("local_safe_true_index",text)
            else
                @test !occursin("select i1 %cond, ptr",text)
                @test occursin("select i1 %cond, i64 %index, i64 0",text)
                @test occursin("local_selected_value = select i1",text)
            end
        end
    end
end
