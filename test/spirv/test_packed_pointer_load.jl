# A non-inlined by-reference kernel argument packed as [N x i64] can carry
# a PSB pointer in slot zero. Read the slot as an integer before converting
# the value, rather than loading a pointer directly from the array address.
using Test, Lava, LLVM

@testset "pointer from the first slot of a packed argument" begin
    LLVM.Context() do _
        mod = parse(LLVM.Module, """
        define void @record(ptr %history) {
        entry:
          %s = getelementptr [13 x i64], ptr %history, i64 0, i64 1
          %stride = load i64, ptr %s, align 8
          %base = load ptr addrspace(1), ptr %history, align 8
          %out = getelementptr float, ptr addrspace(1) %base, i64 %stride
          store float 1.0, ptr addrspace(1) %out, align 4
          ret void
        }
        define void @main() {
        entry:
          %history = alloca [13 x i64], align 8
          store i64 4096, ptr %history, align 8
          %s = getelementptr [13 x i64], ptr %history, i64 0, i64 1
          store i64 1, ptr %s, align 8
          call void @record(ptr %history)
          ret void
        }
        """)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (1, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
        d = Lava.disassemble_spirv(bytes)
        @test occursin("OpAccessChain %_ptr_Function_ulong", d)
        @test occursin("OpConvertUToPtr %_ptr_PhysicalStorageBuffer_float", d)
    end
end
