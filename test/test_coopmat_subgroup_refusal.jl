# A cooperative-matrix pipeline must refuse to build where 32 lanes cannot be pinned.
#
# `get_compute_pipeline` pins a module declaring `CooperativeMatrixKHR` to
# `COOPMAT_SUBGROUP` whenever the device default is something else, and errors
# where the device will not accept the pin. Both halves need a test, and only one
# of them is reachable by running the kernel:
#
#   * that the guard does NOT fire here is asserted directly, because on RDNA 3.5
#     `minSubgroupSize == 32` and every coopmat kernel in the suite depends on
#     that being true. If this half ever fails, the whole cooperative-matrix path
#     on this device is running at the wrong width.
#
#   * that the guard DOES fire is otherwise unreachable on any hardware here:
#     it needs a device with cooperative matrices AND a subgroup that cannot be
#     pinned to 32, which neither the discrete GPU nor lavapipe is (lavapipe has
#     8-lane subgroups but no coopmat). Without a positive control the guard is a
#     branch nobody has executed, which is how it would rot. So the capability
#     cache is driven to describe a wave64-only device and the pipeline is asked
#     for again.
#
# Driving the CACHE rather than the kernel is deliberate: the guard reads
# `subgroup_size_control(ctx)` and `device_subgroup_size(ctx)`, both of which are
# per-device `Dict`s keyed by `ctx.id`, so a test can state the device's answer
# without a second device and without touching the emitter.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "coopmat pipelines and the 32-lane pin" begin
    ctx = Lava.vk_context()

    if !Lava.coopmat_gemm_available(ctx)
        @info "no cooperative matrices on this device; the guard is unreachable"
        @test_skip Lava.coopmat_gemm_available(ctx)
    else
        # ── the half that must hold on this machine ──────────────────────────
        @test Lava.can_require_subgroup_size(ctx, Lava.COOPMAT_SUBGROUP)

        # A real coopmat kernel builds and computes correctly at the pinned width.
        # `mul!` on fp16 operands with an fp32 destination is the shipped route
        # onto `coopmat_gemm!`.
        M = N = K = 128
        A = Lava.LavaArray(Float16.(randn(Float32, M, K) .* 0.1f0))
        B = Lava.LavaArray(Float16.(randn(Float32, K, N) .* 0.1f0))
        C = Lava.LavaArray(zeros(Float32, M, N))
        ref = Float32.(Array(A)) * Float32.(Array(B))
        Lava.LinearAlgebra.mul!(C, A, B)
        KA.synchronize(LavaBackend())
        got = Array(C)
        @test maximum(abs, got) > 1e-3                       # it wrote something
        @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3

        # ── the positive control ─────────────────────────────────────────────
        # Describe a device whose subgroups are 64 and cannot be narrowed, which
        # is what an RDNA part without `VK_EXT_subgroup_size_control` would look
        # like, and assert the refusal rather than a wrong answer.
        saved_ctl  = get(Lava.SUBGROUP_SIZE_CONTROL, ctx.id, nothing)
        saved_size = Lava.device_subgroup_size(ctx)
        try
            Lava.SUBGROUP_SIZE_CONTROL[ctx.id] = Lava.SubgroupSizeControl(64, 64, true)
            Lava.DEVICE_SUBGROUP_SIZE[ctx.id]  = 64
            @test !Lava.can_require_subgroup_size(ctx, Lava.COOPMAT_SUBGROUP)

            # Every cache that could hand back the pipeline built a moment ago,
            # which would make the refusal unreachable and this assertion vacuous.
            empty!(Lava.PIPELINE_CACHE)
            empty!(Lava.PIPELINE_INSERTION_ORDER)
            empty!(Lava.LAUNCH_PLAN_CACHE)
            empty!(Lava.LINKED_KERNEL_CACHE)

            C2 = Lava.LavaArray(zeros(Float32, M, N))
            err = try
                Lava.LinearAlgebra.mul!(C2, A, B)
                KA.synchronize(LavaBackend())
                nothing
            catch e
                e
            end
            @test err !== nothing
            # And it must be THIS refusal, not any error that happens to be thrown.
            @test occursin("cooperative-matrix", sprint(showerror, err))
        finally
            saved_ctl === nothing ? delete!(Lava.SUBGROUP_SIZE_CONTROL, ctx.id) :
                                    (Lava.SUBGROUP_SIZE_CONTROL[ctx.id] = saved_ctl)
            Lava.DEVICE_SUBGROUP_SIZE[ctx.id] = saved_size
            empty!(Lava.PIPELINE_CACHE)
            empty!(Lava.PIPELINE_INSERTION_ORDER)
            empty!(Lava.LAUNCH_PLAN_CACHE)
            empty!(Lava.LINKED_KERNEL_CACHE)
        end
    end
end
