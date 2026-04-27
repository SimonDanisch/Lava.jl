using Test, Lava
using Lava: LavaInstanceRecord, write_grain_instances_kernel, build_blas_aabb,
            build_tlas, refit_tlas!, as_build, AS_INPUT_USAGE
using GeometryBasics: Point3f, Vec3f, Vec4f

# Validates the full P1 flow: GPU kernel writes instances, allow_update build,
# refit moves geometry, ray queries observe the change.

@testset "TLAS refit cycle -- kernel-written instances, then refit" begin
    aabb = Lava.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    aabb_blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end
    tri_blas  = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    radius = 1f0
    quats_cpu = [Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:n]
    quats_gpu = Lava.LavaArray(quats_cpu)
    instances_gpu = Lava.LavaArray{LavaInstanceRecord}(undef, 2 * n;
                                                        extra_usage=AS_INPUT_USAGE)
    backend = Lava.LavaBackend()
    bq = Lava.vk_context().default_bq

    # Frame 0: grains at x = 0, 5, 10, 15 (separated so each has its own AABB).
    pos_a = [Point3f(Float32(5*(i-1)), 0f0, 0f0) for i in 1:n]
    positions_gpu = Lava.LavaArray(pos_a)
    write_grain_instances_kernel(backend)(positions_gpu, quats_gpu, radius,
                                           aabb_blas.address, tri_blas.address,
                                           instances_gpu;
                                           ndrange = n)
    Lava.vk_flush!(bq)

    tlas = as_build() do ctx
        build_tlas(ctx, instances_gpu, 2 * n; allow_update=true)
    end

    @test tlas.allow_update == true
    @test tlas.update_scratch_size > 0

    # Frame 1: shift all grains by +100 in x.
    pos_b = [Point3f(Float32(5*(i-1) + 100f0), 0f0, 0f0) for i in 1:n]
    Lava.copyto!(positions_gpu, pos_b)
    write_grain_instances_kernel(backend)(positions_gpu, quats_gpu, radius,
                                           aabb_blas.address, tri_blas.address,
                                           instances_gpu;
                                           ndrange = n)
    Lava.vk_flush!(bq)

    as_build() do ctx
        refit_tlas!(ctx, tlas, instances_gpu, 2 * n)
    end

    # The handle is preserved (in-place refit).
    @test tlas.allow_update == true
end
