using Test, Lava
import LLVM

@testset "integer zero stores into physical float aggregates" begin
    LLVM.Context() do _
        for (ty, bits) in (("float", 32), ("double", 64))
            mod = parse(LLVM.Module, """
            define void @main() {
            entry:
              %base = inttoptr i64 4096 to ptr addrspace(1)
              %matrix = getelementptr [9 x $ty], ptr addrspace(1) %base, i64 1
              store i$bits 0, ptr addrspace(1) %matrix, align $(bits ÷ 8)
              ret void
            }
            """)
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (1, 1, 1))
            @test Lava.validate_spirv(bytes) === nothing
        end
    end
end
