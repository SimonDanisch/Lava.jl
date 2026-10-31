using LLVM, Lava, Test

@testset "Pointer fields read as integer address bits" begin
    # Optimized Julia code may view the same address slot both as ptr and i64.
    # An OpLoad must retain the slot's declared pointer type; conversion to its
    # integer representation belongs after the load.
    for (types, root, indices) in (
            ("%A = type { ptr addrspace(1), [1 x i64] }", "%A", "i64 0, i32 0"),
            ("%A = type { ptr addrspace(1), [1 x i64] }\n%W = type { [1 x %A] }", "%W", "i64 0, i32 0, i64 0, i32 0"),
        )
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                $types
                @out = internal addrspace(3) global [64 x i64] undef
                @floatout = internal addrspace(3) global [64 x float] undef
                define void @main() {
                entry:
                  %a = alloca $root
                  %field = getelementptr $root, ptr %a, $indices
                  %address_buffer = alloca i64
                  store i64 4096, ptr %address_buffer
                  %address = load i64, ptr %address_buffer
                  %device_pointer = inttoptr i64 %address to ptr addrspace(1)
                  %hint = load float, ptr addrspace(1) %device_pointer
                  store ptr addrspace(1) %device_pointer, ptr %field
                  %typed = load ptr addrspace(1), ptr %field
                  %number = load float, ptr addrspace(1) %typed
                  %word = load i64, ptr %field
                  %o = getelementptr [64 x i64], ptr addrspace(3) @out, i64 0, i64 0
                  store i64 %word, ptr addrspace(3) %o
                  %f = getelementptr [64 x float], ptr addrspace(3) @floatout, i64 0, i64 0
                  store float %number, ptr addrspace(3) %f
                  ret void
                }
                """
            )
            @test LLVM.verify(mod) === nothing
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
            @test Lava.validate_spirv(bytes) === nothing
        end
    end
end
