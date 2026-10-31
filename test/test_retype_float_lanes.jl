# Floats in an alloca that one wide integer store fills.
#
# `MVector{2, Float32}(Inf32, Inf32)` becomes an `alloca i64` that a single
# `store i64` fills with both lanes; every other access reads or writes one
# `float` through a GEP. `retype_uniform_typed_allocas!` picks `float` as the
# element type, and before it only decomposed a wider access for integer elements,
# so it gave up on this alloca. The emitter then kept the `ulong` variable and
# stored a `float` into it, which spirv-val rejects ("OpStore Pointer <id>'s type
# does not match Object <id>'s type"); PrintSim's `find_face_touches!` kernel, which
# keeps its candidates' distances in such a vector, hit it. The wide access now
# goes through integers of its width and of the element's, bitcast to the element
# type at either end.

using Test
using Lava
using LLVM

const _IR_FLOAT_LANES = """
define void @main(ptr addrspace(1) %out, ptr addrspace(1) %wide, i64 %k, float %d) {
entry:
  %a = alloca i64, align 16
  store i64 9187343241974906880, ptr %a, align 4
  %p = getelementptr inbounds float, ptr %a, i64 %k
  store float %d, ptr %p, align 4
  %x = load float, ptr %a, align 4
  %q = getelementptr inbounds i8, ptr %a, i64 4
  %y = load float, ptr %q, align 4
  %s = fadd float %x, %y
  store float %s, ptr addrspace(1) %out, align 4
  %w = load i64, ptr %a, align 4
  store i64 %w, ptr addrspace(1) %wide, align 8
  ret void
}
"""

@testset "floats in an alloca a wide integer store fills" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_FLOAT_LANES)
        Lava.retype_uniform_typed_allocas!(mod, LLVM.datalayout(mod))
        @test LLVM.verify(mod) === nothing
        f = LLVM.functions(mod)["main"]
        insts = collect(LLVM.instructions(first(LLVM.blocks(f))))
        allocas = filter(i -> i isa LLVM.AllocaInst, insts)
        @test length(allocas) == 1
        @test string(LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(only(allocas)))) == "[2 x float]"
        # Nothing reads or writes the alloca as anything but a float any more.
        private(p) = !(p isa LLVM.Argument) && LLVM.addrspace(LLVM.value_type(p)) == 0
        stored = [LLVM.value_type(LLVM.operands(i)[1]) for i in insts if i isa LLVM.StoreInst && private(LLVM.operands(i)[2])]
        loaded = [LLVM.value_type(i) for i in insts if i isa LLVM.LoadInst]
        @test length(stored) == 3 && all(t -> string(t) == "float", stored)
        @test length(loaded) == 4 && all(t -> string(t) == "float", loaded)
    end
end
