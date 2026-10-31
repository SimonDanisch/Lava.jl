# A value narrower than an integer variable, written into part of it.
#
# SROA leaves a pair of `Bool`s that is also read whole as one `i16` alloca written a
# byte at a time: the lanes of GPUArrays' `mapreduce`, in `istril` and `istriu`. A
# SPIR-V variable of a scalar type has no byte inside it to point at, and the emitter
# spliced the store to byte 1 into byte 0, so the reduction read `false`. Before that,
# the splice itself failed spirv-val: its mask was a 32-bit constant for a 16-bit
# variable. `lower_subword_access_on_integer_allocas!` shifts and masks in LLVM, and the
# emitter's own splice masks at the variable's width.
#
# Not a GPU test: the passes, emission and spirv-val. Run on the device in Mantle's
# `test_lane_reductions.jl`.

using Test
using Lava
using LLVM

const _IR_SUBWORD_LANES = """
@lava_shared_out = internal addrspace(3) global [1 x i16] undef, align 2
@__spirv_BuiltInLocalInvocationIndex = external addrspace(7) global i32

define void @main() {
entry:
  %lanes = alloca i16, align 2
  %x = load i32, ptr addrspace(7) @__spirv_BuiltInLocalInvocationIndex, align 4
  %lo = trunc i32 %x to i8
  store i8 %lo, ptr %lanes, align 2
  %hi.p = getelementptr inbounds i8, ptr %lanes, i64 1
  store i8 1, ptr %hi.p, align 1
  %both = load i16, ptr %lanes, align 2
  %hi = load i8, ptr %hi.p, align 1
  %hi16 = zext i8 %hi to i16
  %sum = add i16 %both, %hi16
  store i16 %sum, ptr addrspace(3) @lava_shared_out, align 2
  ret void
}
"""

@testset "a byte of an integer variable is that byte" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_SUBWORD_LANES)
        Lava.lower_subword_access_on_integer_allocas!(mod)
        ir = string(mod)
        @test !occursin(r"(store|load) i8", ir)                # whole-variable accesses only
        @test occursin(r"and i16 %subword_slot\d*, -256", ir)  # byte 0 written keeps byte 1,
        @test occursin(r"and i16 %subword_slot\d*, 255", ir)   # byte 1 written keeps byte 0,
        @test occursin(r"or i16 %subword_kept\d*, 256", ir)    # and is 1 << 8,
        @test occursin(r"lshr i16 %subword_slot\d*, 8", ir)    # and is read from there
        @test LLVM.verify(mod) === nothing
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end

# The emitter's splice of a narrower value into a variable, reached when nothing in
# LLVM lowered it: its mask has the variable's width.
const _IR_NARROW_INTO_SHORT = """
@lava_shared_out = internal addrspace(3) global [1 x i16] undef, align 2
@__spirv_BuiltInLocalInvocationIndex = external addrspace(7) global i32

define void @main() {
entry:
  %lanes = alloca i16, align 2
  store i16 -1, ptr %lanes, align 2
  %x = load i32, ptr addrspace(7) @__spirv_BuiltInLocalInvocationIndex, align 4
  %lo = trunc i32 %x to i8
  store i8 %lo, ptr %lanes, align 2
  %both = load i16, ptr %lanes, align 2
  store i16 %both, ptr addrspace(3) @lava_shared_out, align 2
  ret void
}
"""

@testset "a narrower value spliced into a variable is masked at its width" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_NARROW_INTO_SHORT)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end
