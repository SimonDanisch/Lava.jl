# Inlining a relay can expose an aggregate alloca to a helper that used to receive
# another helper's opaque pointer. Recheck the pointer boundary after each round;
# otherwise the leaf tries to step off a scalar Function pointer.
using Test, Lava, LLVM

@testset "aggregate pointers exposed by inlining a relay" begin
    LLVM.Context() do _
        source = """
        define float @leaf(ptr %p) noinline {
          %first = load float, ptr %p
          %next = getelementptr i8, ptr %p, i64 24
          %last = load float, ptr %next
          %sum = fadd float %first, %last
          ret float %sum
        }
        define float @relay(ptr %p) alwaysinline {
          %r = call float @leaf(ptr %p)
          ret float %r
        }
        define void @main() {
          %p = alloca {float, float, float, float, float, float, float, float}
          store float 1.0, ptr %p
          %last = getelementptr {float, float, float, float, float, float, float, float}, ptr %p, i32 0, i32 6
          store float 7.0, ptr %last
          %r = call float @relay(ptr %p)
          store float %r, ptr %p
          ret void
        }
        """
        mod = parse(LLVM.Module, source)
        Lava.force_inline_all!(mod, mod.functions["main"])
        bytes, locations = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        Lava.validate_spirv(bytes, string(mod), locations)
        @test !occursin("OpFunctionCall", Lava.disassemble_spirv(bytes))
        LLVM.dispose(mod)
    end
end
