# A memcpy lowered to words claims the alignment its pointers have.
#
# `lower_memcpy!` copies what it cannot type in 4-byte words, and marked every word
# `align 4`. AcceleratedKernels' chunk totals are `_Lane{Int64}`s, a value and a `valid`
# flag, and their seven bytes of padding after the flag are copied that way, the first
# word at byte 9 of the element. NVIDIA took the claim and wrote that word at byte 8, over
# the flag: every chunk total read as empty, and a scan along a matrix's rows with an
# operator without a neutral element dropped every chunk but the last's.
#
# The offset of a field after padding, the other half of the same scans: the type-pun
# decomposition put the third field of `{ float, i64, float }` at byte 12 (the fields'
# sizes added up), inside the `i64`, where the datalayout has it at 16, so `fill!` with
# such a struct wrote the `i64`'s high half where the last `Float32` belongs.
#
# Not a GPU test: the passes on LLVM IR. On the device, Mantle's
# `test_lane_reductions.jl` and `test_struct_alignment_systematic.jl`.

using Test
using Lava
using LLVM

const _IR_MEMCPY_AT_ODD_OFFSET = """
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define void @main(ptr addrspace(1) %element) {
entry:
  %padding = alloca [7 x i8], align 1
  store [7 x i8] zeroinitializer, ptr %padding, align 1
  %after_flag = getelementptr inbounds i8, ptr addrspace(1) %element, i64 9
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 1 %after_flag, ptr align 1 %padding, i64 7, i1 false)
  ret void
}
"""

@testset "a memcpy's words claim only the alignment their pointers have" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_MEMCPY_AT_ODD_OFFSET)
        Lava.lower_unsupported_intrinsics!(mod)
        accesses = [i for b in LLVM.blocks(mod.functions["main"]) for i in LLVM.instructions(b)
                    if i isa Union{LLVM.LoadInst, LLVM.StoreInst} && i isa LLVM.Instruction]
        words = filter(i -> (i isa LLVM.StoreInst ? i.operands[1].value_type : i.value_type) == LLVM.Int32Type(),
                       accesses)
        @test !isempty(words)                                   # still copied in words,
        @test all(i -> LLVM.alignment(i) == 1, words)           # each byte-aligned
        @test LLVM.verify(mod) === nothing
    end
    @test Lava.chunk_alignment(16, 0, 4) == 4
    @test Lava.chunk_alignment(16, 8, 4) == 4
    @test Lava.chunk_alignment(16, 9, 4) == 1
    @test Lava.chunk_alignment(16, 6, 4) == 2
    @test Lava.chunk_alignment(1, 4, 4) == 1
end

# `i64:64`, as the SPIR-V target lays it out: under LLVM's default datalayout an `i64` is
# four-aligned, and the sizes added up are the offsets.
const _IR_FIELD_AFTER_PADDING = """
target datalayout = "e-i64:64"

@lava_shared_out = internal addrspace(3) global [1 x i32] undef, align 4

define void @main() {
entry:
  %s = alloca { float, i64, float }, align 8
  store { float, i64, float } { float 1.5, i64 7, float 2.5 }, ptr %s, align 8
  %c = getelementptr inbounds { float, i64, float }, ptr %s, i64 0, i32 2
  %bits = load i32, ptr %c, align 4
  store i32 %bits, ptr addrspace(3) @lava_shared_out, align 4
  ret void
}
"""

@testset "a field after padding is where the datalayout puts it" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_FIELD_AFTER_PADDING)
        Lava.decompose_typepun_gep_loads!(mod, mod.datalayout)
        ir = string(mod)
        @test !occursin("lshr", ir)                        # nothing is cut out of the i64
        @test occursin(r"load float, ptr %typepun_gep", ir) # the third field is read whole
        @test LLVM.verify(mod) === nothing
    end
end
