# Diagonal * matrix must not be ambiguous with the dense GEMM.
#
# `mul!(C::LavaArray{T,2}, ::AbstractVecOrMat, ::AbstractVecOrMat, α, β)` in
# gemm.jl and GPUArrays' `mul!(::AbstractGPUVecOrMat,
# ::Diagonal{<:Any,<:AbstractGPUArray}, …)` are both applicable when the left
# operand is a Diagonal, and neither is more specific. That is a MethodError,
# not a wrong answer:
#
#   mul!(::LavaArray{Float32,2}, ::Diagonal{Float32,LavaArray{Float32,1}},
#        ::LavaArray{Float32,2}, ::Float32, ::Float32) is ambiguous
#
# Lava's method only became applicable when the GEMM landed, so this covers the
# disambiguation rather than the multiply itself. GPUArrays' behaviour is the one
# to keep: a diagonal operand is a scaling, and routing it through the dense GEMM
# would materialise the zeros.

using Test, Lava, LinearAlgebra

@testset "Diagonal mul! disambiguation" begin
    n, m = 6, 4
    hd = rand(Float32, n); hB = rand(Float32, n, m); hC = rand(Float32, n, m)
    α, β = 2.0f0, 3.0f0

    D = Diagonal(Lava.LavaArray(copy(hd)))
    B = Lava.LavaArray(copy(hB))
    C = Lava.LavaArray(copy(hC))
    mul!(C, D, B, α, β)
    @test Array(C) ≈ α .* Diagonal(hd) * hB .+ β .* hC

    # β = 0 must ignore C's existing contents rather than scale them.
    C0 = Lava.LavaArray(copy(hC))
    mul!(C0, D, B, 1.0f0, 0.0f0)
    @test Array(C0) ≈ Diagonal(hd) * hB

    # The dense path must be unaffected by the added method.
    A = Lava.LavaArray(rand(Float32, n, n))
    C2 = Lava.LavaArray(zeros(Float32, n, m))
    mul!(C2, A, B, 1.0f0, 0.0f0)
    @test Array(C2) ≈ Array(A) * hB
end
