"""
The staged GEMM against the register-blocked one, on shapes that expose its
assumptions — and **every tiling in `GEMM_TILINGS`, individually**.

That last part is not thoroughness for its own sake. `(3, 2, 2, 4, 32, 8)` — a
96 x 128 block — silently lost 4 of every 32 k-terms per row, while 96 x 64 and
160 x 128 were correct, and nothing about the tiling predicted which. The cause
was not the tiling at all but the shape of the kernel's k-loop: see
`test_shared_index_division.jl`, where `while` loses 92% of a staged block that
a counted `for` keeps intact, on identical instructions.

So a tiling is trusted because it was run, never because it looks like one that
was.

The other bug this file exists for: **the staged kernel does not split K.** It
walks the whole of it and writes one plane. But `coopmat_gemm_shape` picks
`splitk` from the shape, and at 64x64x64 it picks 4 — so the caller allocates
four partial planes and sums them, three of which the staged kernel never wrote.
The result came back with 0.83 relative error, and only on small shapes: every
shape the kernel had ever been benchmarked on chooses `splitk == 1`. A GEMM test
suite made of the shapes a GEMM is fast at is a suite that cannot find this.
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

"""
Every k-term accumulated, or not.

`A` and `B` all ones makes each output element the count of products that
actually landed, so a result of 28 where K is 32 says *which* is wrong rather
than merely *that* it is — and an integer count survives fp16 exactly, where a
relative error on random data can hide four missing terms in the tolerance.
"""
function stagedcount(backend, cfg, K; blocks = 2)
    M = Lava.gemm_bm(cfg) * blocks
    N = Lava.gemm_bn(cfg) * blocks
    A = KA.allocate(backend, Float16, M, K); fill!(A, one(Float16))
    B = KA.allocate(backend, Float16, K, N); fill!(B, one(Float16))
    C = KA.allocate(backend, Float16, M, N); fill!(C, Float16(-1))
    wg = Lava.gemm_wg(cfg)
    Lava.GEMM_STAGED_KERNELS[cfg](backend, wg)(
        C, A, B, nothing, Val(M), Val(N), Val(K);
        ndrange = (M ÷ Lava.gemm_bm(cfg)) * (N ÷ Lava.gemm_bn(cfg)) * wg)
    KA.synchronize(backend)
    g = Float32.(Array(C))
    (correct = all(==(Float32(K)), g), K = K, seen = sort(unique(g)))
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

    @testset "both staging widths accumulate every k-term, at every K" begin
        # `GEMM_VEC2` picks between scalar and `vec2` staging buffers, and both
        # are generated for every tiling — so both have to be swept, not just
        # whichever is currently the default.
        old = Lava.GEMM_VEC2[]
        try
            for v2 in (false, true)
                Lava.GEMM_VEC2[] = v2
                for cfg in Lava.GEMM_TILINGS,
                    K in (32, 64, 96, 128, 288, 576, 2304)
                    K % Lava.gemm_bk(cfg) == 0 || continue
                    v2 && !haskey(Lava.GEMM_STAGED_V2_KERNELS, cfg) && continue
                    @test gemmerr(backend, ws, Lava.gemm_bm(cfg) * 2,
                                  Lava.gemm_bn(cfg) * 2, K;
                                  staged = true, withbias = false) < 2.0f-2
                end
            end
        finally
            Lava.GEMM_VEC2[] = old
        end
    end

    @testset "every shipped tiling accumulates every k-term, at every K" begin
        # The K sweep is the point. The hazard in `test_shared_index_division.jl`
        # depends on the trip count as well as the geometry — the same kernel is
        # exact at K = 32 and loses terms at K = 64 — so checking one size proves
        # nothing about the others. 32 to 2304 is one iteration to seventy-two.
        for cfg in Lava.GEMM_TILINGS,
            K in (32, 64, 96, 128, 160, 288, 320, 576, 1152, 2304)
            K % Lava.gemm_bk(cfg) == 0 || continue
            r = stagedcount(backend, cfg, K)
            r.correct || @info "tiling $cfg lost k-terms" K = r.K seen = r.seen
            @test r.correct
        end
    end

    @testset "the tiling chooser only returns blocks that divide the shape" begin
        old = Lava.GEMM_TILING[]
        try
            Lava.GEMM_TILING[] = nothing
            for (M, N, K) in vcat(shapes, [(288, 16384, 1152), (1152, 16384, 288),
                                           (2304, 4096, 576), (48, 64, 64)])
                c = Lava.gemm_tiling(M, N, K)
                c === nothing && continue
                @test M % Lava.gemm_bm(c) == 0
                @test N % Lava.gemm_bn(c) == 0
                @test K % Lava.gemm_bk(c) == 0
                @test c in Lava.GEMM_TILINGS
            end
        finally
            Lava.GEMM_TILING[] = old
        end
    end

    @testset "the guard names the calls the staged kernel may take" begin
        old = (Lava.GEMM_STAGED[], Lava.GEMM_TILING[])
        try
            Lava.GEMM_STAGED[] = true
            Lava.GEMM_TILING[] = nothing
            for (M, N, K) in shapes
                splitk = Lava.coopmat_gemm_shape(M, N, K)[2]
                Lava.staged_gemm_tiling(M, N, K, 1, splitk) === nothing || @test splitk == 1
            end
            @test Lava.staged_gemm_tiling(64, 64, 64, 1, 4) === nothing   # split: refused
            @test Lava.staged_gemm_tiling(64, 64, 64, 2, 1) === nothing   # batched: refused
            @test Lava.staged_gemm_tiling(48, 40, 64, 1, 1) === nothing   # no block divides
        finally
            Lava.GEMM_STAGED[], Lava.GEMM_TILING[] = old
        end
    end
end

@testset "the aliasing rule keeps the 96-row block off its bad stride" begin
    # `gemm_aliasing` exists because a *weighted mean over shapes* said the wrong
    # thing: 96 x 128 is the faster block on four of SAM 2's six `addmm` shapes
    # and collapses on the fifth, and the fifth was heavy enough to lose it the
    # average. The collapse is exactly at `K % 256 == 0` — measured at seventeen
    # values of K on either side of it, with everything but K held fixed.
    #
    # Asserted as *selection*, not as speed: a timing assertion on this card is a
    # coin flip (the clock idles at 210 MHz of 2265) and would fail for reasons
    # that have nothing to do with the rule.
    old = (Lava.GEMM_STAGED[], Lava.GEMM_TILING[])
    try
        Lava.GEMM_STAGED[], Lava.GEMM_TILING[] = true, nothing

        c96 = (3, 2, 2, 4, 32, 8)
        @test Lava.gemm_bm(c96) == 96 && !ispow2(Lava.gemm_bm(c96))
        @test Lava.gemm_aliasing(c96, 2304)          # 256 * 9
        @test Lava.gemm_aliasing(c96, 1024)
        @test !Lava.gemm_aliasing(c96, 2208)         # one step below, and fine
        @test !Lava.gemm_aliasing(c96, 2400)         # one step above, and fine
        # A power-of-two block is never refused, whatever K does.
        for c in Lava.GEMM_TILINGS, K in (256, 1024, 2304, 4096)
            ispow2(Lava.gemm_bm(c)) && @test !Lava.gemm_aliasing(c, K)
        end

        # The rule as the picker applies it: no shape whose K is a multiple of
        # 256 may come back with a non-power-of-two block.
        for K in (256, 512, 1024, 1536, 2048, 2304, 2560, 3072), M in (576, 1152, 2304)
            c = Lava.gemm_tiling(M, 4096, K)
            c === nothing && continue
            @test ispow2(Lava.gemm_bm(c))
        end
        # ...and where it is allowed, the 96-row block is what gets picked, since
        # it now leads the table.
        for K in (576, 1152, 1728, 2880), M in (576, 1152, 2304)
            @test Lava.gemm_tiling(M, 4096, K) == c96
        end

        # SAM 2's own six, as the encoder runs them.
        @test Lava.gemm_tiling(2304, 4096,  576) == c96
        @test Lava.gemm_tiling( 576, 4096, 2304) == (2, 2, 2, 4, 32, 8)   # the bad K
        @test Lava.gemm_tiling(1728, 4096,  576) == c96
        @test Lava.gemm_tiling( 576, 4096,  576) == c96
        @test Lava.gemm_tiling( 288, 16384, 1152) == c96
        @test Lava.gemm_tiling(1152, 16384,  288) == c96
    finally
        Lava.GEMM_STAGED[], Lava.GEMM_TILING[] = old
    end
end
