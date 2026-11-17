using Test, Lava, LLVM

function private_alias_reader(ir)
    name = gensym(:private_alias_reader)
    return Core.eval(@__MODULE__, :(function $name(index::Int64)
        Base.llvmcall(($ir, "read_alias"), Float32, Tuple{Int64}, index)
    end))
end

@testset "private pointer mirrors preserve the allocation bounds" begin
    ir = """
    define float @read_alias(i64 %index) {
    entry:
      %a = alloca [4 x float], align 4
      %b = alloca float, align 4
      store [4 x float] [float 1.0, float 2.0, float 3.0, float 4.0], ptr %a
      store float 9.0, ptr %b
      %p = getelementptr [1 x float], ptr %a, i64 0, i64 %index
      %choose = icmp ult i64 %index, 3
      %selected = select i1 %choose, ptr %p, ptr %b
      %value = load float, ptr %selected
      ret float %value
    }
    """
    variants = [
        ("one-element view", ir, Float32[1, 2, 3, 9]),
        ("vector view", replace(ir, "getelementptr [1 x float]" => "getelementptr <1 x float>"), Float32[1, 2, 3, 9]),
        ("narrow signed index", replace(ir, "%p = getelementptr" => "%narrow = trunc i64 %index to i32\n      %p = getelementptr",
             "i64 0, i64 %index" => "i64 0, i32 %narrow"), Float32[1, 2, 3, 9]),
        ("complete multi-index offset", replace(ir, "getelementptr [1 x float], ptr %a, i64 0, i64 %index" =>
             "getelementptr [1 x float], ptr %a, i64 %index, i64 1"), Float32[2, 3, 4, 9]),
        ("unselected negative offset", replace(ir, "i64 0, i64 %index" => "i64 0, i64 -1",
             "%choose = icmp ult i64 %index, 3" => "%choose = icmp ult i64 %index, 0"), fill(9f0, 4)),
        ("wide access", replace(ir, "%b = alloca float" => "%b = alloca [2 x float]",
             "store float 9.0, ptr %b" => "store [2 x float] [float 9.0, float 9.0], ptr %b",
             "%value = load float, ptr %selected" => "%pair = load <2 x float>, ptr %selected, align 4\n      %value = extractelement <2 x float> %pair, i32 1"), Float32[2, 3, 4, 9]),
    ]
    for (name, source, expected) in variants
        @testset "$name" begin
            original = private_alias_reader(source)
            @test [Base.invokelatest(original, i) for i in Int64(0):Int64(3)] == expected
            LLVM.Context() do ctx
                mod = parse(LLVM.Module, source)
                Lava.lower_phi_select_function_ptrs!(mod)
                @test LLVM.verify(mod) === nothing
                # Wide accesses above require only four-byte alignment, even
                # though their vector type's default alignment is eight.
                loads = [ins for fn in mod.functions for bb in fn.blocks for ins in bb.instructions if ins isa LLVM.LoadInst]
                @test all(ins -> LLVM.alignment(ins) <= 4, loads)
                lowered = private_alias_reader(string(mod))
                @test [Base.invokelatest(lowered, i) for i in Int64(0):Int64(3)] == expected
            end
        end
    end
end
