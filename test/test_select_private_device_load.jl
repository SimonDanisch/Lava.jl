# A load through a select between a private and a device pointer is a branch.
#
# SPIR-V has no pointer that is either a Function or a PhysicalStorageBuffer one, so a
# select between the two cannot be emitted; the emitter made one anyway, out of an
# `OpConvertPtrToU` on the Function pointer, which spirv-val rejects ("Pointer storage
# class must be PhysicalStorageBuffer: ConvertPtrToU"). Hikari's node interpreter in a
# closest-hit shader made it: `o == NO_SLOT ? ins.k[i] : stack[o]`, with the instruction
# read from device memory and the stack private, both loads merged by LLVM into one
# through a select. Reading both would read the device address the condition rules
# out, so `branch_mixed_pointer_select_loads!` branches instead, before structurization.
#
# Not a GPU test: the passes, emission and spirv-val.

using Test
using Lava
using LLVM

const _IR_SELECT_PRIVATE_DEVICE_LOAD = """
@lava_shared_out = internal addrspace(3) global [1 x float] undef, align 4
@__spirv_BuiltInLocalInvocationIndex = external addrspace(7) global i32

define void @main() {
entry:
  %stack = alloca [4 x float], align 4
  store float 1.0, ptr %stack, align 4
  %x = load i32, ptr addrspace(7) @__spirv_BuiltInLocalInvocationIndex, align 4
  %o = trunc i32 %x to i8
  %const = icmp eq i8 %o, 0
  %o64 = zext i8 %o to i64
  %slot = getelementptr float, ptr %stack, i64 %o64
  %address = add i64 %o64, 4096
  %instruction = inttoptr i64 %address to ptr
  %k = getelementptr i8, ptr %instruction, i64 28
  %p = select i1 %const, ptr %k, ptr %slot
  %value = load float, ptr %p, align 4
  store float %value, ptr addrspace(3) @lava_shared_out, align 4
  ret void
}
"""

@testset "a load through a select between a private and a device pointer is a branch" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_SELECT_PRIVATE_DEVICE_LOAD)
        Lava.branch_mixed_pointer_select_loads!(mod)
        ir = string(mod)
        @test !occursin(r"select i1 [^,]+, ptr", ir)    # no pointer select is left
        @test occursin("phi float", ir)                 # the two loads are merged
        @test LLVM.verify(mod) === nothing
        Lava.run_structurize_cfg_pipeline!(mod)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end

# The same choice made by the edge that came in: a phi of a device pointer and a private
# one, which structurization makes a select after this pass has run. GPUArrays'
# `generic_trimatmul!` reads `unit ? oA : A[i, i]` for a complex matrix, `oA` an argument
# copied to a private alloca.
const _IR_PHI_PRIVATE_DEVICE_LOAD = """
@lava_shared_out = internal addrspace(3) global [1 x float] undef, align 4
@__spirv_BuiltInLocalInvocationIndex = external addrspace(7) global i32

define void @main() {
entry:
  %copy = alloca [2 x float], align 8
  store float 1.0, ptr %copy, align 4
  %x = load i32, ptr addrspace(7) @__spirv_BuiltInLocalInvocationIndex, align 4
  %unit = icmp eq i32 %x, 0
  %x64 = zext i32 %x to i64
  %address = add i64 %x64, 4096
  %device = inttoptr i64 %address to ptr
  br i1 %unit, label %private, label %element

private:
  br label %join

element:
  br label %join

join:
  %p = phi ptr [ %copy, %private ], [ %device, %element ]
  %value = load float, ptr %p, align 4
  store float %value, ptr addrspace(3) @lava_shared_out, align 4
  ret void
}
"""

@testset "a load through a phi of a private and a device pointer is a branch" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, _IR_PHI_PRIVATE_DEVICE_LOAD)
        Lava.branch_mixed_pointer_select_loads!(mod)
        ir = string(mod)
        @test !occursin("load float, ptr %p,", ir)      # nothing loads through the phi
        @test occursin("phi float", ir)                 # the two loads are merged
        @test LLVM.verify(mod) === nothing
        Lava.run_structurize_cfg_pipeline!(mod)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end
