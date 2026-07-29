# The cooperative-matrix GEMM is only valid on a 32-lane subgroup.
#
# The block kernel computes its subgroup index as `lane ÷ 32` (gemm.jl) and
# GEMM_WORKGROUP is sized as "2 subgroups" on the same assumption. Cooperative
# matrix operations are subgroup-scoped, so on a device with a different wave
# width those lanes do not form a subgroup: on wave64 hardware exactly HALF the
# output tile is written — bit-exact where written, zero elsewhere — which is a
# silently wrong answer, not a crash. It reproduced down to a single 16x16x16
# tile with nbatch=1.
#
# `coopmat_gemm_available()` therefore requires the device subgroup size to match,
# and `mul!` falls back to `gemmlaunch!`, which is correct at any wave width.
#
# Written against whatever the running device reports, so it is meaningful on
# wave32 hardware (where the coopmat path is taken) and wave64 (where it is not).

using Test, Lava, LinearAlgebra, KernelAbstractions
const KA = KernelAbstractions

@testset "coopmat GEMM requires a matching subgroup size" begin
    ctx = Lava.vk_context()
    sg = Lava.device_subgroup_size(ctx)
    @test sg > 0

    # Availability may never be true on a mismatched wave width, whatever shapes
    # the device reports.
    if Lava.coopmat_gemm_available(ctx)
        @test sg == Lava.GEMM_SUBGROUP
    else
        @test sg != Lava.GEMM_SUBGROUP ||
              !Lava.coopmat_shape(ctx, Float16, Lava.GEMM_TILE, Lava.GEMM_TILE, Lava.GEMM_TILE)
    end

    # Whichever path `mul!` picks, the answer must be right — this is the case
    # that silently returned a half-written tile.
    M, N, K = 64, 64, 32
    hA = Float16.(reshape(sin.(range(0, 3, M * K)), M, K))
    hB = Float16.(reshape(cos.(range(0, 4, K * N)), K, N))
    A = Lava.LavaArray(hA); B = Lava.LavaArray(hB)
    C = Lava.LavaArray(zeros(Float32, M, N))
    mul!(C, A, B, 1.0f0, 0.0f0)
    KA.synchronize(LavaBackend())
    got = Array(C)
    want = Float32.(hA) * Float32.(hB)

    @test count(!iszero, got) == length(got)          # nothing left unwritten
    @test maximum(abs, got .- want) / maximum(abs, want) < 1e-2
end
