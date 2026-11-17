using Test, Lava, LLVM

const PRIVATE_ARRAY_CHOICE_IR = """
@__spirv_BuiltInGlobalInvocationId = external addrspace(7) global <3 x i32>
@lava_shared_out = internal addrspace(3) global [64 x i32] undef, align 4

define void @main() {
entry:
  %a = alloca [8 x i32], align 4
  %p0 = getelementptr [8 x i32], ptr %a, i64 0, i64 0
  %p2 = getelementptr [8 x i32], ptr %a, i64 0, i64 2
  store i32 17, ptr %p0
  store i32 29, ptr %p2
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInGlobalInvocationId
  %x = extractelement <3 x i32> %v, i64 0
  %cond = icmp ult i32 %x, 32
  br i1 %cond, label %left, label %right
left:
  %a2 = getelementptr [8 x i32], ptr %a, i64 0, i64 2
  br label %join
right:
  %a4 = getelementptr [8 x i32], ptr %a, i64 0, i64 4
  br label %join
join:
  %p = phi ptr [ %a2, %left ], [ %a4, %right ]
  %previous = getelementptr { i32, i32 }, ptr %p, i64 -1
  %result = load i32, ptr %previous
  %out = getelementptr [64 x i32], ptr addrspace(3) @lava_shared_out, i64 0, i32 %x
  store i32 %result, ptr addrspace(3) %out
  ret void
}
"""

@testset "private array pointer choices become indices" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, PRIVATE_ARRAY_CHOICE_IR)
        @test_throws "in Function storage" Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
    end
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, PRIVATE_ARRAY_CHOICE_IR)
        Lava.index_private_array_choices!(mod, mod.datalayout)
        @test LLVM.verify(mod) === nothing
        instructions = [i for f in mod.functions for b in f.blocks for i in b.instructions]
        @test !any(i -> i isa LLVM.PHIInst && i.value_type isa LLVM.PointerType, instructions)
        @test any(i -> i isa LLVM.PHIInst && i.value_type == LLVM.Int64Type(), instructions)
        @test !occursin("getelementptr { i32, i32 }", string(mod))
        Lava.run_structurize_cfg_pipeline!(mod)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end

    @testset "stores, selects and signed dynamic steps" begin
        LLVM.Context() do ctx
            ir = replace(PRIVATE_ARRAY_CHOICE_IR,
                "%previous = getelementptr { i32, i32 }, ptr %p, i64 -1" => """
                %picked = select i1 %cond, ptr %p, ptr %a4
                %signed_step = sext i32 -1 to i64
                %previous = getelementptr { i32, i32 }, ptr %picked, i64 %signed_step
                store i32 41, ptr %previous
                """)
            # The select must use a pointer dominating the join, not one from only the right edge.
            ir = replace(ir, "ptr %p, ptr %a4" => "ptr %p, ptr %p2")
            mod = parse(LLVM.Module, ir)
            Lava.index_private_array_choices!(mod, mod.datalayout)
            @test LLVM.verify(mod) === nothing
            @test !occursin("phi ptr", string(mod))
            @test !occursin("select i1 %cond, ptr", string(mod))
            @test occursin("store i32 41", string(mod))
            @test occursin("private_choice_index", string(mod))
            Lava.run_structurize_cfg_pipeline!(mod)
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
            @test Lava.validate_spirv(bytes) === nothing
        end
    end

    LLVM.Context() do ctx
        ir = replace(PRIVATE_ARRAY_CHOICE_IR, "[ %a4, %right ]" => "[ undef, %right ]")
        mod = parse(LLVM.Module, ir)
        Lava.index_private_array_choices!(mod, mod.datalayout)
        @test LLVM.verify(mod) === nothing
        @test !occursin("phi ptr", string(mod))
        @test occursin("[ undef, %right ]", string(mod))
    end

    # A runtime tuple index may already be an integer, while its GEP still
    # strides by the pre-SROA struct type over the flattened integer alloca.
    LLVM.Context() do ctx
        ir = replace(PRIVATE_ARRAY_CHOICE_IR,
            "%p = phi ptr [ %a2, %left ], [ %a4, %right ]" => """
            %index = phi i64 [ 1, %left ], [ 2, %right ]
            %p = getelementptr { i32, i32 }, ptr %a, i64 %index
            """)
        mod = parse(LLVM.Module, ir)
        Lava.index_private_array_choices!(mod, mod.datalayout)
        @test LLVM.verify(mod) === nothing
        @test !occursin("getelementptr { i32, i32 }", string(mod))
        Lava.run_structurize_cfg_pipeline!(mod)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end

    @testset "aggregate array element choices" begin
        for direct_field in (false, true)
            LLVM.Context() do ctx
                ir = replace(PRIVATE_ARRAY_CHOICE_IR, "[8 x i32]" => "[8 x { i32, i32 }]")
                # Keep scalar memory operations on the first field of each element.
                ir = replace(ir, "store i32 17, ptr %p0" => "%p0f = getelementptr { i32, i32 }, ptr %p0, i64 0, i32 0\n  store i32 17, ptr %p0f",
                                "store i32 29, ptr %p2" => "%p2f = getelementptr { i32, i32 }, ptr %p2, i64 0, i32 0\n  store i32 29, ptr %p2f",
                                "%result = load i32, ptr %previous" => "%previous_field = getelementptr { i32, i32 }, ptr %previous, i64 0, i32 0\n  %result = load i32, ptr %previous_field")
                direct_field && (ir = replace(ir,
                    "ptr %previous, i64 0, i32 0" => "ptr %p, i64 -1, i32 1"))
                mod = parse(LLVM.Module, ir)
                Lava.index_private_array_choices!(mod, mod.datalayout)
                @test LLVM.verify(mod) === nothing
                @test !occursin("phi ptr", string(mod))
                @test !occursin("ptr %p, i64 -1", string(mod))
                @test all(i -> Lava.is_zero_index(i.operands[2]),
                          (i for f in mod.functions for b in f.blocks for i in b.instructions if
                           i isa LLVM.GetElementPtrInst &&
                           LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(i)) isa LLVM.StructType))
                Lava.run_structurize_cfg_pipeline!(mod)
                bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
                @test Lava.validate_spirv(bytes) === nothing
            end
        end
    end

    # Offsets into different arrays cannot be merged into one rooted address.
    LLVM.Context() do ctx
        ir = replace(PRIVATE_ARRAY_CHOICE_IR,
            "%a = alloca [8 x i32], align 4" => "%a = alloca [8 x i32], align 4\n  %b = alloca [8 x i32], align 4",
            "%a4 = getelementptr [8 x i32], ptr %a" => "%a4 = getelementptr [8 x i32], ptr %b")
        mod = parse(LLVM.Module, ir)
        before = string(mod)
        Lava.index_private_array_choices!(mod, mod.datalayout)
        @test string(mod) == before
        @test LLVM.verify(mod) === nothing
    end

    @testset "shared pointer-choice subgraphs" begin
        # Two prior choices are shared at each level. A tree walk revisits
        # exponentially many paths although there are only forty choices.
        LLVM.Context() do ctx
            choices = join(("%p$j = select i1 %choose, ptr %p$(j-1), ptr %p$(j-2)" for j in 2:41), "\n")
            ir = """
            define i32 @shared_choices(i1 %choose) {
            entry:
              %a = alloca [8 x i32]
              %p0 = getelementptr [8 x i32], ptr %a, i64 0, i64 0
              %p1 = getelementptr [8 x i32], ptr %a, i64 0, i64 2
              store i32 17, ptr %p0
              store i32 29, ptr %p1
              $choices
              %value = load i32, ptr %p41
              ret i32 %value
            }
            """
            mod = parse(LLVM.Module, ir)
            fn = first(mod.functions)
            instructions = [i for b in fn.blocks for i in b.instructions]
            root = first(i for i in instructions if i isa LLVM.AllocaInst)
            choice = last(i for i in instructions if i isa LLVM.SelectInst)
            roots = Dict{LLVM.Value, Union{LLVM.Value, Nothing}}()
            @test Lava.private_array_choice_root(choice, mod.datalayout, Set{LLVM.Value}(), roots) == root
            @test length(roots) == 42
            Lava.index_private_array_choices!(mod, mod.datalayout)
            @test LLVM.verify(mod) === nothing
            @test !any(i -> i isa LLVM.SelectInst && i.value_type isa LLVM.PointerType,
                       (i for f in mod.functions for b in f.blocks for i in b.instructions))
        end
    end

    @testset "cyclic choices remain unchanged" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, """
            define i32 @cyclic_choice(i1 %choose) {
            entry:
              %a = alloca [8 x i32]
              %p0 = getelementptr [8 x i32], ptr %a, i64 0, i64 0
              store i32 17, ptr %p0
              br label %loop
            loop:
              %p = phi ptr [ %p0, %entry ], [ %next, %loop ]
              %next = select i1 %choose, ptr %p, ptr %p0
              %value = load i32, ptr %p
              br i1 %choose, label %exit, label %loop
            exit:
              ret i32 %value
            }
            """)
            before = string(mod)
            Lava.index_private_array_choices!(mod, mod.datalayout)
            @test string(mod) == before
            @test LLVM.verify(mod) === nothing
        end
    end
end
