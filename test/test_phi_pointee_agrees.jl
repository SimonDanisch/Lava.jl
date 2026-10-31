# A pointer phi gets the pointee every access through it is emitted against.
#
# `defer_phi!` typed a pointer phi with its own search over the phi's users, which
# took the first one in use-list order. The loads and GEPs through the phi are
# emitted against the pointee type map, where a direct load of the phi says what
# lies at the phi itself. Here the search reached a byte-offset GEP first, read
# as `i32` 16 bytes in, and typed the phi `uint*`, while the offset-0 load stayed
# `OpLoad %float` through it:
#
#     %146 = OpPhi %_ptr_PhysicalStorageBuffer_uint ...
#     %6274 = OpLoad %float %146 Aligned 4
#
# which spirv-val rejects ("OpLoad Result Type does not match Pointer's type").
# Hikari's per-material shading kernel hit it reading a `[2 x [4 x float]]` whose
# second half LLVM loads as `i32`, under an area light with a spectral material
# (BK7 glass, F11 coated diffuse, BK7 coated conductor), on every driver.
#
# Emission and validation only: the addresses are never dereferenced.
using Test, Lava, LLVM

const PHI_POINTEE_IR = """
@__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>
define void @main() {
entry:
  %id = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
  %x = extractelement <3 x i32> %id, i64 0
  %condition = icmp eq i32 %x, 0
  %xi = zext i32 %x to i64
  %abase = inttoptr i64 4096 to ptr addrspace(1)
  %bbase = inttoptr i64 8192 to ptr addrspace(1)
  %a = getelementptr inbounds [2 x [4 x float]], ptr addrspace(1) %abase, i64 %xi
  %b = getelementptr inbounds [2 x [4 x float]], ptr addrspace(1) %bbase, i64 %xi
  %out = inttoptr i64 12288 to ptr addrspace(1)
  br i1 %condition, label %left, label %join
left:
  br label %join
join:
  %p = phi ptr addrspace(1) [ %a, %entry ], [ %b, %left ]
  %f = load float, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds i8, ptr addrspace(1) %p, i64 16
  %i = load i32, ptr addrspace(1) %q, align 4
  %fi = bitcast float %f to i32
  %s = add i32 %fi, %i
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}
"""

@testset "a pointer phi is typed as the accesses through it are emitted" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, PHI_POINTEE_IR)
        bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (32, 1, 1))
        @test Lava.validate_spirv(bytes) === nothing
    end
end
