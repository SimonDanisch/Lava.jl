# A device address loaded as i64 whose only typed use sits behind a byte GEP. LLVM 19+
# (Julia 1.13) LICM hoists a loop's invariant byte offset, `gep i8, p, -4` from 1-based
# indexing, and leaves the element GEP `gep [1 x float], %hoisted, %i` in the loop
# (Hikari's `lookup_table_read`). The inttoptr had no pointee ("Cannot determine pointee
# type for inttoptr result"), and the -4 then became AccessChain index -1 into a
# one-element array instead of a step back by 4 bytes.
using LLVM, Lava, Test

@testset "inttoptr whose typed use is behind a byte GEP" begin
    LLVM.Context() do _
        mod = parse(LLVM.Module, """
            @out = internal addrspace(3) global [64 x float] undef
            define void @main() {
            entry:
              %address_buffer = alloca i64
              store i64 4096, ptr %address_buffer
              %address = load i64, ptr %address_buffer
              %table = inttoptr i64 %address to ptr addrspace(1)
              %before = getelementptr i8, ptr addrspace(1) %table, i64 -4
              br label %loop
            loop:
              %i = phi i64 [ 1, %entry ], [ %next, %loop ]
              %acc = phi float [ 0.0, %entry ], [ %sum, %loop ]
              %element = getelementptr [1 x float], ptr addrspace(1) %before, i64 %i
              %bits = load i32, ptr addrspace(1) %element
              %punned = bitcast i32 %bits to float
              %value = load float, ptr addrspace(1) %element
              %both = fadd float %punned, %value
              %sum = fadd float %acc, %both
              %next = add i64 %i, 1
              %done = icmp eq i64 %next, 16
              br i1 %done, label %exit, label %loop
            exit:
              %o = getelementptr [64 x float], ptr addrspace(3) @out, i64 0, i64 0
              store float %sum, ptr addrspace(3) %o
              ret void
            }
            """)
        @test LLVM.verify(mod) === nothing
        table = only(i for b in LLVM.blocks(LLVM.functions(mod)["main"])
                     for i in LLVM.instructions(b) if i isa LLVM.IntToPtrInst)
        @test Lava.infer_pointee_from_users(table) == LLVM.ArrayType(LLVM.FloatType(), 1)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
        # The -4 is added to the address, 2^64 - 4 as the unsigned constant.
        @test occursin(r"OpIAdd %ulong %\d+ %ulong_18446744073709551612", Lava.disassemble_spirv(bytes))
    end
end
