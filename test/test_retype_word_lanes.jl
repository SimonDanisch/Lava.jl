using Test, Lava, LLVM

# A mutable word buffer mixes wide copies and indirect narrow field stores.
# Cover odd field indices, signed zero, denormals, infinities and NaN payloads
# as uninterpreted bits, including a heterogeneous view of the same buffer.
const PRIVATE_WORD_LANES_IR = """
define i64 @update_word(i64 %index, i64 %bits) {
entry:
  %a = alloca [4 x i64], align 16
  %p1 = getelementptr [4 x i64], ptr %a, i64 0, i64 1
  %p2 = getelementptr [4 x i64], ptr %a, i64 0, i64 2
  %p3 = getelementptr [4 x i64], ptr %a, i64 0, i64 3
  store i64 81985529216486895, ptr %a
  store i64 1393753992385309920, ptr %p1
  store i64 9223372036854775807, ptr %p2
  store i64 -9223372036854775808, ptr %p3
  %field = getelementptr i32, ptr %a, i64 %index
  %narrow = trunc i64 %bits to i32
  store i32 %narrow, ptr %field, align 4
  %v0 = load i64, ptr %a
  %v1 = load i64, ptr %p1
  %v2 = load i64, ptr %p2
  %v3 = load i64, ptr %p3
  %s0 = mul i64 %v0, 3
  %s1 = mul i64 %v1, 5
  %s2 = mul i64 %v2, 7
  %s3 = mul i64 %v3, 11
  %s01 = xor i64 %s0, %s1
  %s23 = xor i64 %s2, %s3
  %result = xor i64 %s01, %s23
  ret i64 %result
}
"""

function word_lane_reader(ir)
    name = gensym(:word_lane_reader)
    return Core.eval(@__MODULE__, :(function $name(index::Int64, bits::UInt64)
        Base.llvmcall(($ir, "update_word"), UInt64, Tuple{Int64, UInt64}, index, bits)
    end))
end

@testset "private word retyping preserves narrow lanes and wide copies" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, PRIVATE_WORD_LANES_IR)
        Lava.retype_uniform_typed_allocas!(mod, mod.datalayout)
        @test LLVM.verify(mod) === nothing
        allocas = [i for b in mod.functions["update_word"].blocks for i in b.instructions if i isa LLVM.AllocaInst]
        @test length(allocas) == 1
        @test string(LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(only(allocas)))) == "[8 x i32]"
        original = word_lane_reader(PRIVATE_WORD_LANES_IR)
        rewritten = word_lane_reader(string(mod))
        for index in Int64(0):Int64(7), bits in UInt64[0, 1, 0x7f800000, 0x7fc00001, 0x80000000, 0xffffffff, 0x12345678]
            @test Base.invokelatest(rewritten, index, bits) == Base.invokelatest(original, index, bits)
        end
    end
    LLVM.Context() do ctx
        # A heterogeneous float projection must preserve integer bits as well.
        ir = replace(PRIVATE_WORD_LANES_IR, "%v0 = load i64" => """
            %fp = load float, ptr %field, align 4
            %samebits = bitcast float %fp to i32
            store i32 %samebits, ptr %field, align 4
            %v0 = load i64
            """)
        mod = parse(LLVM.Module, ir)
        Lava.retype_uniform_typed_allocas!(mod, mod.datalayout)
        @test LLVM.verify(mod) === nothing
        original = word_lane_reader(PRIVATE_WORD_LANES_IR)
        preserved = word_lane_reader(string(mod))
        for index in Int64(0):Int64(7), bits in UInt64[0, 1, 0x7f800000, 0x7fc00001, 0x80000000, 0xffffffff, 0x12345678]
            @test Base.invokelatest(preserved, index, bits) == Base.invokelatest(original, index, bits)
        end
    end
    LLVM.Context() do ctx
        ir = """
        @__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>
        @word_out = internal addrspace(3) global [64 x i64] undef, align 8
        """ * PRIVATE_WORD_LANES_IR * """
        define void @main() {
        entry:
          %id = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId
          %x = extractelement <3 x i32> %id, i32 0
          %index = and i32 %x, 7
          %wide = zext i32 %index to i64
          %v = call i64 @update_word(i64 %wide, i64 2143289345)
          %p = getelementptr [64 x i64], ptr addrspace(3) @word_out, i32 0, i32 %x
          store i64 %v, ptr addrspace(3) %p
          ret void
        }
        """
        mod = parse(LLVM.Module, ir)
        Lava.retype_uniform_typed_allocas!(mod, mod.datalayout)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end
