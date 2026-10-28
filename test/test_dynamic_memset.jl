# A memset of a runtime length is a loop of typed stores again.
#
# LLVM's loop-idiom pass turns a zeroing loop over a range whose start is a loop
# variable, `for r in k+1:n; M[r, k] = 0; end` on an `MMatrix` (a Householder
# sequence, once `n` is past what the unroller expands), into `llvm.memset` with
# a runtime length, and `lower_memset!` refused it: "Cannot lower memset with
# non-constant length", so the kernel did not compile. `lower_dynamic_memsets!`
# makes it a loop again, storing the element type of the private array, or for
# a device pointer the widest integer that divides every possible length.
#
# A constant-length memset of a private array of one scalar type is now written
# in that type too: i32 words into a `[9 x i64]` made `retype_uniform_typed_allocas!`
# split every other access of the array into 32-bit halves.
#
# The values are checked on a device in Mantle's `test_smatrix_in_kernel.jl`.

using Test, Lava, LLVM

const _IR_DYNAMIC_MEMSET = """
define void @zero_tail(i64 %k, i64 %n, ptr addrspace(1) %dev) {
entry:
  %a = alloca [16 x double], align 8
  %off = shl i64 %k, 3
  %p = getelementptr inbounds i8, ptr %a, i64 %off
  %len = shl i64 %n, 3
  call void @llvm.memset.p0.i64(ptr align 8 %p, i8 0, i64 %len, i1 false)
  %words = shl i64 %n, 2
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 %dev, i8 -1, i64 %words, i1 false)
  ret void
}
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
"""

const _IR_CONSTANT_MEMSET = """
define void @zero_matrix() {
entry:
  %a = alloca [9 x i64], align 16
  call void @llvm.memset.p0.i64(ptr align 16 %a, i8 0, i64 72, i1 false)
  ret void
}
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
"""

@testset "memset of a runtime length" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_DYNAMIC_MEMSET)
        @test Lava.lower_dynamic_memsets!(mod)
        LLVM.verify(mod)
        text = string(mod)
        @test !occursin("llvm.memset", text)
        # The private array is written in doubles, `n` of them.
        @test occursin("store double 0.000000e+00", text)
        # The device pointer in 4-byte words of all ones (the length is a multiple of 4).
        @test occursin(r"store i32 (-1|%\w+), ptr addrspace\(1\)", text)
        @test count(r"phi i64", text) == 2
    end
end

@testset "memset of a constant length into a private array" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_CONSTANT_MEMSET)
        Lava.lower_unsupported_intrinsics!(mod)
        LLVM.verify(mod)
        text = string(mod)
        @test !occursin("llvm.memset", text)
        @test count("store i64 0", text) == 9
        @test !occursin("store i32", text)
    end
end
