# Tier 1b: integers of a width SPIR-V has no type for are carried in a native one.
#
# SimplifyCFG's switch-to-lookup builds a bitmap as wide as its table, whatever the
# datalayout's native widths: Hikari's shader materials tested a closure lobe
# against four of seven values as `lshr i7 -50, (trunc i8 %x to i7)`. The emitter
# mapped `i7` to an 8-bit integer and the `trunc` to an `OpUConvert` between two
# 8-bit integers, which spirv-val refuses. `widen_odd_integers!` carries every odd
# width in the next native one with the bits above it zero.
#
# Each function below is run on the host through `llvmcall` as written, where LLVM
# computes in the odd width, and after the pass, for every pair of 8-bit inputs: the
# IR the pass emits, not a transcription of it. Then the same IR in a kernel goes
# through Lava's whole pipeline and spirv-val. The device side is Mantle's
# `test_odd_integers.jl`.

using Test
using Lava
import LLVM
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm

const ODD_INTEGER_FUNCTIONS = Dict(
    # Hikari's bitmap: whether `x` is not one of 1, 2, 3, 6.
    "bitmap" => """
        %c = icmp ugt i8 %x, 6
        %cast = trunc i8 %x to i7
        %down = lshr i7 -50, %cast
        %bit = and i7 %down, 1
        %m = icmp eq i7 %bit, 0
        %s = select i1 %c, i1 true, i1 %m
        %r = zext i1 %s to i32
        ret i32 %r""",
    # Every operation that can set the bits above the width, then read as signed.
    "wrapping" => """
        %a = trunc i8 %x to i7
        %b = trunc i8 %y to i7
        %s = add i7 %a, %b
        %m = mul i7 %s, %b
        %h = shl i7 %m, 3
        %d = sub i7 %h, %a
        %r = sext i7 %d to i32
        ret i32 %r""",
    # Reading as signed: shifts, division, compares, conversion to float.
    "signed" => """
        %a = trunc i8 %x to i7
        %b = trunc i8 %y to i7
        %q = ashr i7 %a, 2
        %dl = and i7 %b, 15
        %dv = or i7 %dl, 1
        %v = sdiv i7 %a, %dv
        %w = srem i7 %b, %dv
        %lt = icmp slt i7 %a, %b
        %f = sitofp i7 %a to float
        %fi = fptosi float %f to i32
        %q32 = sext i7 %q to i32
        %v32 = sext i7 %v to i32
        %w32 = sext i7 %w to i32
        %lt32 = zext i1 %lt to i32
        %r1 = mul i32 %q32, 1000003
        %r2 = mul i32 %v32, 10007
        %r3 = mul i32 %w32, 101
        %r4 = mul i32 %fi, 7
        %s1 = add i32 %r1, %r2
        %s2 = add i32 %s1, %r3
        %s3 = add i32 %s2, %r4
        %r = add i32 %s3, %lt32
        ret i32 %r""",
    # Unsigned reads and conversions, from and to other odd widths.
    "unsigned" => """
        %a = trunc i8 %x to i5
        %b = zext i8 %y to i12
        %c = zext i5 %a to i12
        %p = mul i12 %b, %c
        %dv = or i12 %b, 1
        %u = udiv i12 %p, %dv
        %m = urem i12 %p, %dv
        %ge = icmp uge i12 %p, 2000
        %o = xor i12 %p, -1
        %f = uitofp i12 %o to float
        %fu = fptoui float %f to i32
        %t = trunc i12 %u to i3
        %t32 = zext i3 %t to i32
        %m32 = zext i12 %m to i32
        %ge32 = zext i1 %ge to i32
        %r1 = mul i32 %fu, 1000003
        %r2 = mul i32 %m32, 101
        %r3 = mul i32 %t32, 11
        %s1 = add i32 %r1, %r2
        %s2 = add i32 %s1, %r3
        %r = add i32 %s2, %ge32
        ret i32 %r""",
    # Around a loop, and a switch on one.
    "control" => """
        %a = trunc i8 %x to i7
        %n = trunc i8 %y to i3
        br label %loop
    loop:
        %i = phi i3 [ 0, %entry ], [ %i1, %loop ]
        %acc = phi i7 [ %a, %entry ], [ %acc2, %loop ]
        %acc2 = mul i7 %acc, 3
        %i1 = add i3 %i, 1
        %done = icmp eq i3 %i1, %n
        br i1 %done, label %exit, label %loop
    exit:
        switch i7 %acc2, label %other [ i7 3, label %three
                                        i7 -10, label %minus ]
    three:
        br label %join
    minus:
        br label %join
    other:
        br label %join
    join:
        %k = phi i32 [ 1, %three ], [ 2, %minus ], [ 0, %other ]
        %z = zext i7 %acc2 to i32
        %z4 = shl i32 %z, 4
        %r = or i32 %z4, %k
        ret i32 %r""",
)

"`body` as `define i32 @f(i8 %x, i8 %y)`, after `widen_odd_integers!` when `widen`."
function odd_integer_ir(body; widen::Bool)
    ir = "define i32 @f(i8 %x, i8 %y) alwaysinline {\nentry:\n" * body * "\n}\n"
    widen || return ir
    LLVM.@dispose ctx = LLVM.Context() begin
        mod = parse(LLVM.Module, ir)
        @assert Lava.widen_odd_integers!(mod)
        LLVM.verify(mod)
        string(mod)
    end
end

"Whether the IR has an integer of an odd width left."
hasoddint(ir) = occursin(r"\bi(?!(1|8|16|32|64)\b)\d+\b", ir)

for (name, body) in ODD_INTEGER_FUNCTIONS
    @eval $(Symbol("odd_", name))(x::UInt8, y::UInt8) =
        Base.llvmcall(($(odd_integer_ir(body; widen = false)), "f"), Int32, Tuple{UInt8, UInt8}, x, y)
    @eval $(Symbol("wide_", name))(x::UInt8, y::UInt8) =
        Base.llvmcall(($(odd_integer_ir(body; widen = true)), "f"), Int32, Tuple{UInt8, UInt8}, x, y)
end

"How many of the 2^16 input pairs `wide` answers differently from `odd`."
oddinteger_errors(odd, wide) = count(odd(x, y) != wide(x, y) for x in 0x00:0xff, y in 0x00:0xff)

@testset "Odd-width integers" begin
    @testset "$name keeps its values" for name in sort(collect(keys(ODD_INTEGER_FUNCTIONS)))
        @test !hasoddint(odd_integer_ir(ODD_INTEGER_FUNCTIONS[name]; widen = true))
        odd, wide = getfield(@__MODULE__, Symbol("odd_", name)), getfield(@__MODULE__, Symbol("wide_", name))
        @test oddinteger_errors(odd, wide) == 0
    end

    # The bitmap and the wrapping arithmetic in a kernel: the whole pipeline, validated.
    @testset "a kernel with odd-width integers validates" begin
        function odd_integer_kernel(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = odd_bitmap(A[i], B[i]) + odd_wrapping(A[i], B[i]) + odd_control(A[i], B[i])
            return nothing
        end
        tt = Tuple{Lava.LavaDeviceArray{UInt8, 1}, Lava.LavaDeviceArray{UInt8, 1}, Lava.LavaDeviceArray{Int32, 1}}
        compile_and_disasm(odd_integer_kernel, tt)   # validated by spirv-val
        @test true
    end
end
