# Opaque pointer selects must retain the storage class of local arrays.
using Test,Lava
import LLVM

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
        Lava.lower_phi_select_function_ptrs!(mod)
        @test LLVM.verify(mod) === nothing
        folded=collect(LLVM.instructions(first(LLVM.blocks(LLVM.functions(mod)["choose"]))))
        # Trunk lowers nested pointer choices to values. Every speculative
        # access using the dynamic index is clamped into its private array.
        text=string(mod)
        @test occursin("icmp slt i64 %index, 0",text)
        @test occursin("icmp sgt i64",text)
        pointer_selects=filter(i->i isa LLVM.SelectInst && i.value_type isa LLVM.PointerType,folded)
        @test length(pointer_selects)==1
        @test only(pointer_selects).name=="device"
        @test any(i->i isa LLVM.SelectInst && i.value_type == LLVM.Int32Type(),folded)
    end
end

@testset "pointer-select reads stay after intervening writes" begin
    LLVM.Context() do _
        mod = parse(LLVM.Module, """
        define i32 @choose(i1 %cond) {
        entry:
          %a = alloca i32
          %b = alloca i32
          store i32 1, ptr %a
          store i32 2, ptr %b
          %p = select i1 %cond, ptr %a, ptr %b
          store i32 99, ptr %a
          %v = load i32, ptr %p
          ret i32 %v
        }
        """)
        Lava.lower_phi_select_function_ptrs!(mod)
        @test LLVM.verify(mod) === nothing
        text = string(mod)
        @test !occursin("select i1 %cond, ptr", text)
        # Both reads must observe the final store, including the selected a.
        storepos = first(findfirst("store i32 99", text))
        for read in ("load i32, ptr %a", "load i32, ptr %b")
            @test first(findfirst(read, text)) > storepos
        end
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
            Lava.lower_phi_select_function_ptrs!(mod)
            @test LLVM.verify(mod) === nothing
            text=string(mod)
            if volatile
                @test occursin("select i1 %cond, ptr",text)
                @test count("load volatile i32",text) == 1
                @test !occursin("vmirror_leaf_load",text)
            else
                @test !occursin("select i1 %cond, ptr",text)
                @test occursin("icmp slt i64 %index, 0",text)
                @test occursin("icmp sgt i64",text)
                @test occursin("vmirror_sel = select i1",text)
                @test count("load i32",text) == 2
            end
        end
    end
end
