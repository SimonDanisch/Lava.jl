# Emission/validation only: the illustrative addresses are never executed.
using Test,Lava,LLVM

const POINTER_TABLE_PHI_IR = """
@__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>
define void @main() {
entry:
  %id = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
  %x = extractelement <3 x i32> %id, i64 0
  %condition = icmp eq i32 %x, 0
  %a = inttoptr i64 4096 to ptr addrspace(1)
  %b = inttoptr i64 8192 to ptr addrspace(1)
  %ap = load ptr addrspace(1), ptr addrspace(1) %a, align 8
  %bp = load ptr addrspace(1), ptr addrspace(1) %b, align 8
  %av = load float, ptr addrspace(1) %ap, align 4
  %bv = load float, ptr addrspace(1) %bp, align 4
  br i1 %condition, label %left, label %join
left:
  br label %join
join:
  %holder = phi ptr addrspace(1) [ %a, %entry ], [ %b, %left ]
  %pointer = load ptr addrspace(1), ptr addrspace(1) %holder, align 8
  %value = load float, ptr addrspace(1) %pointer, align 4
  store float %value, ptr addrspace(1) %pointer, align 4
  ret void
}
"""

const POINTER_TABLE_CHAINED_PHI_IR = replace(POINTER_TABLE_PHI_IR,
    "  %holder = phi ptr addrspace(1) [ %a, %entry ], [ %b, %left ]" => """
  %selected = phi ptr addrspace(1) [ %a, %entry ], [ %b, %left ]
  br i1 %condition, label %second, label %final
second:
  br label %final
final:
  %holder = phi ptr addrspace(1) [ %selected, %join ], [ %a, %second ]
  %tag = load i8, ptr addrspace(1) %selected, align 1""")

@testset "A pointer-table PHI retains its nested pointer type" begin
    for ir in (POINTER_TABLE_PHI_IR,POINTER_TABLE_CHAINED_PHI_IR)
        LLVM.Context() do ctx
            mod=parse(LLVM.Module,ir)
            bytes,_=Lava.emit_spirv_from_llvm(mod,"main",(32,1,1))
            @test Lava.validate_spirv(bytes)===nothing
        end
    end
end
