"""
The staged GEMM against the register-blocked one, on shapes that expose its
assumptions.

`GEMM_STAGED` has been off by default for most of its life, which is why the bug
this file exists for went unnoticed: **the staged kernel does not split K.** It
walks the whole of it and writes one plane. But `coopmat_gemm_shape` picks
`splitk` from the shape, and at 64x64x64 it picks 4 — so the caller allocates
four partial planes and sums them, three of which the staged kernel never wrote.
The result came back with 0.83 relative error, and only on small shapes: every
shape the kernel had ever been benchmarked on chooses `splitk == 1`.

So the tests here are deliberately small. A GEMM test suite made of the shapes a
GEMM is fast at is a suite that cannot find this.
"""

using Test, Lava, DNNKernels, KernelAbstractions
const KA = KernelAbstractions

"Relative error of `matmul!` against a Float32 CPU reference, over a slice."
function gemmerr(backend, ws, M, N, K; staged::Bool, withbias::Bool)
    hA = rand(Float16, M, K) .- Float16(0.5)
    hB = rand(Float16, K, N) .- Float16(0.5)
    hbias = Float16.(rand(M) .- 0.5)
    A = KA.allocate(backend, Float16, M, K); copyto!(A, hA)
    B = KA.allocate(backend, Float16, K, N); copyto!(B, hB)
    bias = nothing
    if withbias
        bias = KA.allocate(backend, Float16, M)
        copyto!(bias, hbias)
    end
    old = Lava.GEMM_STAGED[]
    out = KA.allocate(backend, Float16, M, N)
    fill!(out, zero(Float16))
    try
        Lava.GEMM_STAGED[] = staged
        DNNKernels.reset!(ws)
        DNNKernels.matmul!(out, A, B, bias; ws)
        KA.synchronize(backend)
    finally
        Lava.GEMM_STAGED[] = old
    end
    rows = 1:min(M, 24)
    ref = Float32.(hA[rows, :]) * Float32.(hB)
    withbias && (ref = ref .+ Float32.(hbias[rows]))
    return maximum(abs.(Float32.(Array(out)[rows, :]) .- ref)) / maximum(abs.(ref))
end

@testset "staged GEMM" begin
    backend = LavaBackend()
    ws = DNNKernels.Workspace(backend)

    # Small shapes first, because they are the ones whose plan splits K.
    shapes = [(64, 64, 64), (64, 64, 32), (64, 64, 128), (128, 128, 64),
              (128, 64, 64), (64, 128, 64), (192, 192, 64),
              (576, 4096, 576), (2304, 4096, 576)]

    @testset "both paths, with and without bias" begin
        for (M, N, K) in shapes, staged in (false, true), withbias in (false, true)
            # fp16 destination and fp16 accumulate through the tensor cores, so
            # the tolerance is the format's.
            @test gemmerr(backend, ws, M, N, K; staged, withbias) < 2.0f-2
        end
    end

    @testset "the guard names the shapes the staged kernel may take" begin
        # It may only run where the plan wants a single plane; `splitk` comes
        # from `coopmat_gemm_shape`, not from the kernel.
        old = Lava.GEMM_STAGED[]
        try
            Lava.GEMM_STAGED[] = true
            for (M, N, K) in shapes
                splitk = Lava.coopmat_gemm_shape(M, N, K)[2]
                ok = Lava.staged_gemm_applicable(M, N, K, 1, splitk)
                ok && @test splitk == 1
            end
            @test !Lava.staged_gemm_applicable(64, 64, 64, 1, 4)   # split: refused
            @test !Lava.staged_gemm_applicable(64, 64, 64, 2, 1)   # batched: refused
            @test !Lava.staged_gemm_applicable(48, 64, 64, 1, 1)   # M off the block
        finally
            Lava.GEMM_STAGED[] = old
        end
    end
end
