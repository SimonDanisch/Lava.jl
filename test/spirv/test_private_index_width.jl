# Tier 1: an index into a private array is computed in 32 bits.
#
# The closed-form 3x3 eigensolver writes its eigenvector columns at `k` and `l`,
# chosen by comparing eigenvalue gaps. In memory that `MMatrix` is `[9 x i64]`
# retyped to 32-bit words, and the word written for `E[3, k]` is at
# `(select(c, 0, 48) + 16) >> 2`. Computed in i64 — the type Julia's index
# arithmetic has — NVIDIA (RTX 4000 Ada, driver 595) stored it to the wrong
# element: column `k` lost its last entry and the cross product built from it
# was wrong. The same arithmetic in 32 bits, or as `(x >> 2) + 4`, stored it
# right. The access chain takes 32 bits of the index either way, so the emitter
# now does the arithmetic of an index in 32 bits (`narrow_index!`).
#
# The values are checked on a device in Mantle's `test_smatrix_in_kernel.jl`.

using Test
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: compile_and_disasm
using StaticArrays

cross3(a, b) = SVector(a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

function columns_by_comparison_kernel!(out, λs, cols)
    t = Int(KernelInterface.get_global_id().x)
    t <= length(λs) || return nothing
    @inbounds λ, a, b = λs[t], cols[2t - 1], cols[2t]
    k, l = 1, 3
    if λ[3] - λ[2] > λ[2] - λ[1]
        k, l = l, k
    end
    E = MMatrix{3,3,Float64}(undef)
    for r in 1:3
        E[r, k] = a[r]
        E[r, l] = b[r]
    end
    c = cross3(SVector(E[1, 3], E[2, 3], E[3, 3]), SVector(E[1, 1], E[2, 1], E[3, 1]))
    for r in 1:3
        E[r, 2] = c[r]
    end
    @inbounds out[t] = SMatrix{3,3,Float64}(E)
    return nothing
end

"""
The access chains into Function storage whose index is a 64-bit value
truncated (`OpUConvert %uint` of an `%ulong` result) after a right shift.
"""
function wide_shifted_function_indices(d::AbstractString)
    defs = Dict{String,String}()
    for m in eachmatch(r"(%\w+) = (Op\w+ %\w+[^\n]*)", d)
        defs[m[1]] = m[2]
    end
    wide = String[]
    for m in eachmatch(r"OpAccessChain %_ptr_Function_\w+ %\w+ (%\w+)", d)
        def = get(defs, m[1], "")
        conv = match(r"^OpUConvert %uint (%\w+)", def)
        conv === nothing && continue
        startswith(get(defs, conv[1], ""), "OpShiftRight") && push!(wide, m.match)
    end
    return wide
end

@testset "private array indices in 32 bits" begin
    V = Lava.LavaDeviceArray{SVector{3,Float64},1}
    d, _ = compile_and_disasm(columns_by_comparison_kernel!,
                              Tuple{Lava.LavaDeviceArray{SMatrix{3,3,Float64,9},1}, V, V})
    @test isempty(wide_shifted_function_indices(d))
end
