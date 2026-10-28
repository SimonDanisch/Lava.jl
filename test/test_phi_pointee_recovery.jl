# Pointer operands arriving at a phi retain the aggregate type used after it.
using Test, Lava
const LLVM = Lava.LLVM
@testset "recover BDA aggregate pointees through pointer phis" begin
    LLVM.Context() do _
        mod = parse(LLVM.Module, """
        define float @pick(i1 %which, i64 %left_address, i64 %right_address) {
        entry:
          %left = inttoptr i64 %left_address to ptr addrspace(1)
          %right = inttoptr i64 %right_address to ptr addrspace(1)
          br i1 %which, label %yes, label %no
        yes:
          br label %merge
        no:
          br label %merge
        merge:
          %selected = phi ptr addrspace(1) [ %left, %yes ], [ %right, %no ]
          %element = getelementptr [3 x float], ptr addrspace(1) %selected, i64 0, i64 2
          %value = load float, ptr addrspace(1) %element
          ret float %value
        }
        """)
        ptm = Lava.build_pointee_type_map(mod)
        pointers = [inst for block in LLVM.blocks(LLVM.functions(mod)["pick"])
                    for inst in LLVM.instructions(block) if inst isa LLVM.IntToPtrInst]
        @test length(pointers) == 2
        for pointer in pointers
            inferred = Lava.get_pointee_type(ptm, pointer)
            @test inferred isa LLVM.ArrayType
            if inferred isa LLVM.ArrayType
                @test inferred.length == 3
                @test inferred.element_type == LLVM.FloatType()
            end
        end
        @test LLVM.verify(mod) === nothing
    end
end
