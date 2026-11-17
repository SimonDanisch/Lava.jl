# Tier 1: a step off a pointer to an ELEMENT of a Function-storage array.
#
# `M[r, k]` of an `SMatrix` held in a private array, with a runtime column `k`,
# reads the column through one pointer to its first element and steps off it:
#
#     %col = getelementptr [9 x double], ptr %M, i64 0, i64 %k3
#     %r1  = getelementptr double, ptr %col, i64 1
#
# StaticArrays' `mul_loop`, which every `SMatrix * SMatrix` from 5x5 up runs
# (`for j = 2:K` is not unrolled), reads both operands that way. A logical
# pointer has no arithmetic, so the step must become an index of the array, and
# the emitter folded it only when the element pointer came from a ONE-index GEP.
# This two-index form recorded no array origin, the step fell through to "use
# the base pointer", and every row read row 1: `Q * Diagonal(λ) * Q'` of 6x6
# gave diag [20, 20, 0, 0, 0, 0] for 20·I, `A * B` was wrong from 5x5 up, and a
# column `h[:, k]` of a 3x3 read `h[1, k]` three times. spirv-val accepts all of
# it. Three other places in `emit_gep!`/`emit_byte_offset_gep!` emitted the base
# pointer for a step they could not express; they now refuse to compile.
#
# The values are checked on a device in Mantle's `test_smatrix_in_kernel.jl`.
# Here: the loads go through as many distinct access chains as they read
# elements, and the refusal, on hand-written IR.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, compile_and_disasm
using StaticArrays
using LLVM

"Column `k` of a 3x3 read as a whole: `ntuple(r -> h[r, k])` with a runtime `k`."
function smatrix_column!(out, inp)
    k = Int(KernelInterface.get_global_id().x)
    k <= 3 || return nothing
    h = SMatrix{3,3,Float64}(ntuple(j -> @inbounds(inp[j]), Val(9)))
    @inbounds out[k] = SVector{3,Float64}(ntuple(r -> h[r, k], Val(3)))
    return nothing
end

"A 6x6 product of two in-kernel `SMatrix`es, StaticArrays' `mul_loop`."
function smatrix_product6!(out, inp)
    A = SMatrix{6,6,Float64}(ntuple(j -> @inbounds(inp[j]), Val(36)))
    B = SMatrix{6,6,Float64}(ntuple(j -> @inbounds(inp[36 + j]), Val(36)))
    C = A * B
    for j in 1:36
        @inbounds out[j] = C[j]
    end
    return nothing
end

"""
The distinct pointers `double`s are loaded through that are access chains into
Function storage. A dropped step shows as fewer pointers than elements read:
every element of a column became one load, which spirv-opt merges.
"""
function function_load_pointers(d::AbstractString)
    chains = Set{String}()
    for m in eachmatch(r"(%\w+) = OpAccessChain %_ptr_Function_double ", d)
        push!(chains, m[1])
    end
    return Set(m[1] for m in eachmatch(r"OpLoad %double (%\w+)", d) if m[1] in chains)
end

const _IR_ELEMENT_STEP = """
@__spirv_BuiltInGlobalInvocationId = external addrspace(7) global <3 x i32>
@lava_shared_out = internal addrspace(3) global [64 x double] undef, align 8

define void @main() {
entry:
  %a = alloca [9 x double], align 8
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInGlobalInvocationId, align 16
  %x = extractelement <3 x i32> %v, i64 0
  %m = urem i32 %x, 3
  %k = zext i32 %m to i64
  %k3 = mul i64 %k, 3
  %col = getelementptr inbounds [9 x double], ptr %a, i64 0, i64 %k3
  %p1 = getelementptr double, ptr %col, i64 1
  %p2 = getelementptr double, ptr %col, i64 2
  %v0 = load double, ptr %col, align 8
  %v1 = load double, ptr %p1, align 8
  %v2 = load double, ptr %p2, align 8
  %s = fadd double %v0, %v1
  %t = fadd double %s, %v2
  %o = getelementptr inbounds [64 x double], ptr addrspace(3) @lava_shared_out, i64 0, i64 %k
  store double %t, ptr addrspace(3) %o, align 8
  ret void
}
"""

# A runtime step off a scalar alloca: there is no array to fold it into.
const _IR_SCALAR_STEP = """
@__spirv_BuiltInGlobalInvocationId = external addrspace(7) global <3 x i32>
@lava_shared_out = internal addrspace(3) global [64 x double] undef, align 8

define void @main() {
entry:
  %a = alloca double, align 8
  %v = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInGlobalInvocationId, align 16
  %x = extractelement <3 x i32> %v, i64 0
  %k = zext i32 %x to i64
  %p = getelementptr double, ptr %a, i64 %k
  %l = load double, ptr %p, align 8
  %o = getelementptr inbounds [64 x double], ptr addrspace(3) @lava_shared_out, i64 0, i64 %k
  store double %l, ptr addrspace(3) %o, align 8
  ret void
}
"""

@testset "steps off a Function-storage array element" begin
    D = Lava.LavaDeviceArray{Float64,1}

    @testset "a column of an SMatrix at a runtime index" begin
        d, _ = compile_and_disasm(smatrix_column!, Tuple{Lava.LavaDeviceArray{SVector{3,Float64},1}, D})
        # Three rows, three access chains (one before the fix).
        @test length(function_load_pointers(d)) == 3
    end

    @testset "a 6x6 SMatrix product" begin
        d, _ = compile_and_disasm(smatrix_product6!, Tuple{D, D})
        # Each trip of `mul_loop` reads a column of A and a row of B: twelve
        # elements (three access chains before the fix).
        @test length(function_load_pointers(d)) >= 12
    end

    @testset "the step folds into the array index (IR)" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_ELEMENT_STEP)
            bytes, _ = Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
            @test Lava.validate_spirv(bytes) === nothing
            d = Lava.disassemble_spirv(bytes)
            @test length(function_load_pointers(d)) == 3
        end
    end

    @testset "a step with no array to fold into does not compile (IR)" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_SCALAR_STEP)
            @test_throws "in Function storage" Lava.emit_spirv_from_llvm(mod, "main", (64, 1, 1))
        end
    end
end
