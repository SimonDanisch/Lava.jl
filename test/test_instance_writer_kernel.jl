using Test, Lava
using Lava: LavaInstanceRecord, write_grain_instances_kernel,
            write_meshscatter_instances_kernel, build_blas_aabb,
            as_build, AS_INPUT_USAGE
using GeometryBasics: Point3f, Vec3f, Vec4f

@testset "write_grain_instances_kernel -- 4 grains, identity rotations" begin
    # Build two trivial BLASes so we have non-zero device addresses.
    aabb = Lava.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    aabb_blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end
    tri_blas  = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    positions_cpu = [Point3f(Float32(i), 0f0, 0f0) for i in 1:n]
    quats_cpu     = [Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:n]   # identity quaternions
    positions_gpu = Lava.LavaArray(positions_cpu)
    quats_gpu     = Lava.LavaArray(quats_cpu)
    instances_gpu = Lava.LavaArray{LavaInstanceRecord}(undef, 2 * n;
                                                        extra_usage=AS_INPUT_USAGE)

    radius = 0.5f0

    # Launch via KA backend pattern: kernel(LavaBackend())(args...; ndrange=n)
    k = write_grain_instances_kernel(Lava.LavaBackend())
    k(positions_gpu, quats_gpu, radius,
      aabb_blas.address, tri_blas.address,
      instances_gpu;
      ndrange = n)

    bq = Lava.vk_context().default_bq
    Lava.vk_flush!(bq)

    instances_cpu = Array(instances_gpu)

    # Per grain i: 2 records.
    for i in 1:n
        rec_phys = instances_cpu[2i - 1]
        rec_rend = instances_cpu[2i]
        # Both share transform: identity-rot x radius scale x translate by i.
        expected_t = (radius, 0f0, 0f0, Float32(i),
                       0f0, radius, 0f0, 0f0,
                       0f0, 0f0, radius, 0f0)
        @test rec_phys.transform == expected_t
        @test rec_rend.transform == expected_t
        # custom_index = i-1 in low 24 bits.
        @test (rec_phys.custom_index_and_mask & 0x00FFFFFF) == UInt32(i - 1)
        @test (rec_rend.custom_index_and_mask & 0x00FFFFFF) == UInt32(i - 1)
        # Mask bits.
        @test (rec_phys.custom_index_and_mask >> 24) == UInt32(0x02)
        @test (rec_rend.custom_index_and_mask >> 24) == UInt32(0x04)
        # BLAS addresses.
        @test rec_phys.blas_address == aabb_blas.address
        @test rec_rend.blas_address == tri_blas.address
    end
end

@testset "write_meshscatter_instances_kernel -- basic" begin
    # Single BLAS for all N instances (one record per instance).
    aabb = Lava.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    positions_cpu = [Point3f(Float32(i), 0f0, 0f0) for i in 1:n]
    rotations_cpu = [Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:n]  # identity quats
    positions_gpu = Lava.LavaArray(positions_cpu)
    rotations_gpu = Lava.LavaArray(rotations_cpu)
    instances_gpu = Lava.LavaArray{LavaInstanceRecord}(undef, n;
                                                        extra_usage=AS_INPUT_USAGE)
    scale = 2.0f0
    mask  = UInt8(0x04)

    k = Lava.write_meshscatter_instances_kernel(Lava.LavaBackend())
    k(positions_gpu, rotations_gpu, scale, blas.address, mask, instances_gpu;
      ndrange = n)

    bq = Lava.vk_context().default_bq
    Lava.vk_flush!(bq)

    instances_cpu = Array(instances_gpu)

    for i in 1:n
        rec = instances_cpu[i]
        # Transform: identity rotation * scale 2 = diag(2,2,2), translation = (i, 0, 0).
        expected_t = (2f0, 0f0, 0f0, Float32(i),
                       0f0, 2f0, 0f0, 0f0,
                       0f0, 0f0, 2f0, 0f0)
        @test rec.transform == expected_t
        @test (rec.custom_index_and_mask & 0x00FFFFFF) == UInt32(i - 1)
        @test (rec.custom_index_and_mask >> 24) == UInt32(0x04)
        @test rec.blas_address == blas.address
    end
end
