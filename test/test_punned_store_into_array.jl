# A store punned to an integer reaches the first scalar of the array it lands on.
#
# Hikari builds a lobe of a material's type with every field zero for a BSDF slot no
# layer fills, and LLVM writes that zero as `store i32 0` at each byte offset of the
# struct. A store landing at the start of a `[1 x [4 x float]]` that is not the first
# member of its struct resolves to a pointer to that array. The store side drilled to
# a same-width leaf only through structs, then fell back to one level of array, and
# stored the integer into a `float[4]`: spirv-val refuses "OpStore Pointer's type
# does not match Object's type". It now drills to the leaf of any aggregate, as a
# load from one does. A narrower integer, which no leaf matches in width, is spliced
# into the leaf by the store's fallback, which drilled one level: into a nested
# array it stored the integer into the inner array.
#
# Not a GPU test: emission and spirv-val. The device side is Mantle's
# `test_punned_store_into_array.jl`.

using Test
using Lava
using LLVM

const _IR_PUNNED_STORE_INTO_ARRAY = """
@lava_shared_out = internal addrspace(3) global [1 x float] undef, align 4
@__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>

define internal void @blank(ptr noalias nocapture noundef nonnull sret({ float, { [1 x [4 x float]], [1 x [4 x float]] } }) align 4 dereferenceable(36) %ret) noinline {
top:
  store i32 0, ptr %ret, align 4
  %p4 = getelementptr inbounds i8, ptr %ret, i64 4
  store i32 0, ptr %p4, align 4
  %p20 = getelementptr inbounds i8, ptr %ret, i64 20
  store i32 0, ptr %p20, align 4
  ret void
}

define void @main() {
entry:
  %w = alloca { float, { [1 x [4 x float]], [1 x [4 x float]] } }, align 4
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
  %x = extractelement <3 x i32> %v, i64 0
  %f = uitofp i32 %x to float
  %q = getelementptr inbounds i8, ptr %w, i64 24
  store float %f, ptr %q, align 4
  call void @blank(ptr %w)
  %p = getelementptr inbounds i8, ptr %w, i64 20
  %y = load float, ptr %p, align 4
  %r = load float, ptr %q, align 4
  %s = fadd float %y, %r
  store float %s, ptr addrspace(3) @lava_shared_out, align 4
  ret void
}
"""

const _IR_NARROW_STORE_INTO_ARRAY = """
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1-n8:16:32:64"
target triple = "spirv64-unknown-unknown-unknown"

@lava_shared_out = internal addrspace(3) global [1 x i32] undef, align 4
@__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>

define internal void @low(ptr noalias nocapture noundef nonnull sret({ float, { [1 x [2 x i64]], [1 x [2 x i64]] } }) align 8 dereferenceable(40) %ret, i32 %x) noinline {
top:
  %p24 = getelementptr inbounds i8, ptr %ret, i64 24
  store i32 %x, ptr %p24, align 8
  ret void
}

define void @main() {
entry:
  %w = alloca { float, { [1 x [2 x i64]], [1 x [2 x i64]] } }, align 8
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
  %x = extractelement <3 x i32> %v, i64 0
  %q = getelementptr inbounds i8, ptr %w, i64 24
  store i64 -1, ptr %q, align 8
  call void @low(ptr %w, i32 %x)
  %r = load i64, ptr %q, align 8
  %t = trunc i64 %r to i32
  store i32 %t, ptr addrspace(3) @lava_shared_out, align 4
  ret void
}
"""

@testset "a punned store into a nested array reaches its first scalar" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_PUNNED_STORE_INTO_ARRAY)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_NARROW_STORE_INTO_ARRAY)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
        # Spliced into the leaf: the slot is read, its high half kept and the value or-ed in.
        @test occursin("OpBitwiseOr %ulong", Lava.disassemble_spirv(bytes))
    end
end
