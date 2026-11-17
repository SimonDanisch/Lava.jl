# Tier 1: a PhysicalStorageBuffer pointer stepped by a NEGATIVE CONSTANT is
# address arithmetic, not `OpPtrAccessChain`.
#
# `flag[i] == 1 ? a[i] : b[i]` over two views (a device array and an offset,
# `getindex` adding the offset) is, after LLVM, a phi of the two element
# addresses `a.data + a.offset + i` and `b.data + b.offset + i`, stepped back by
# the -1 of 1-based indexing:
#
#     %p = phi ptr [ %pa, ... ], [ %pb, ... ]
#     %q = getelementptr double, ptr %p, i64 -1
#
# Emitted as `OpPtrAccessChain %p %ulong_18446744073709551615`, every lane read
# 0 on NVIDIA (RTX 4000 Ada, driver 595); the same chain off a pointer whose own
# offset the driver cannot see reads correctly, and so does `ifelse`, whose
# select of two raw pointers has no phi. The emitter folds chains of access
# chains into one from the root (`emit_psb_ptr_arithmetic!`), so the -1 only
# stays visible when the base is a phi or select of them, and it now becomes
# `OpConvertPtrToU` + `OpIAdd` + `OpConvertUToPtr`, which read correctly.
#
# The values are checked on a device in Mantle's `test_branch_between_views.jl`.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, compile_and_disasm

"A device array seen from an offset, indexed from 1."
struct OffsetView{A}
    data::A
    offset::Int32
end
@inline Base.getindex(v::OffsetView, i::Integer) = @inbounds v.data[v.offset + i]

function select_between_views!(out, flag, a, b)
    i = Int32(KernelInterface.get_global_id().x)
    @inbounds out[i] = flag[i] == Int32(1) ? a[i] : b[i]
    return nothing
end

"""
The `OpPtrAccessChain`s whose Element is a negative constant (64- or 32-bit,
signed or not).
"""
function negative_element_chains(d::AbstractString)
    negative = Set{String}()
    for m in eachmatch(r"(%\w+) = OpConstant %(\w+) (-?\d+)", d)
        v = parse(BigInt, m[3])
        bits = occursin("long", m[2]) ? 64 : 32
        (v < 0 || v >= big(2)^(bits - 1)) && push!(negative, m[1])
    end
    return [m.match for m in eachmatch(r"OpPtrAccessChain %\w+ %\w+ (%\w+)", d) if m[1] in negative]
end

@testset "a negative constant step off a phi of PSB pointers" begin
    D = Lava.LavaDeviceArray{Float64,1}
    V = OffsetView{D}
    d, _ = compile_and_disasm(select_between_views!, Tuple{D, Lava.LavaDeviceArray{Int32,1}, V, V})
    @test isempty(negative_element_chains(d))
    # The step is still there, as arithmetic on the address.
    check(d, "OpConvertPtrToU")
    # Steps that are not negative constants keep their access chains.
    check(d, "OpPtrAccessChain")
end
