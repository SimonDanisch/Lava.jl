using Test, Lava, Raycore
using Lava: LavaInstanceRecord, build_blas_aabb, as_build, AS_INPUT_USAGE,
            write_grain_instances_kernel
using GeometryBasics: Point3f, Vec3f, Vec4f

@testset "HWTLAS sync! with instance batch" begin
    aabb = Lava.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 8
    radius = 1f0
    quats = Lava.LavaArray([Vec4f(0,0,0,1) for _ in 1:n])
    positions = Lava.LavaArray([Point3f(Float32(3*(i-1)),0,0) for i in 1:n])
    instance_buf = Lava.LavaArray{LavaInstanceRecord}(undef, 2 * n; extra_usage=AS_INPUT_USAGE)

    backend = Lava.LavaBackend()
    bq = Lava.vk_context().default_bq

    # Use the KA backend pattern for @kernel-defined kernels.
    write_grain_instances_kernel(backend)(
        positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    Lava.vk_flush!(bq)

    tlas = Lava.HWTLAS(backend)
    Raycore.push_instances!(tlas, blas, instance_buf; n=2*n, instance_mask=UInt8(0x02))
    Raycore.sync!(tlas)

    @test tlas.dirty == false
    @test tlas.hw_tlas !== nothing
    @test tlas.hw_tlas.allow_update == true
    @test tlas.hw_tlas.update_scratch_size > 0
end

@testset "HWTLAS sync! errors on multiple batches" begin
    aabb = Lava.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    instance_buf1 = Lava.LavaArray{LavaInstanceRecord}(undef, 4; extra_usage=AS_INPUT_USAGE)
    instance_buf2 = Lava.LavaArray{LavaInstanceRecord}(undef, 4; extra_usage=AS_INPUT_USAGE)

    backend = Lava.LavaBackend()
    tlas = Lava.HWTLAS(backend)
    Raycore.push_instances!(tlas, blas, instance_buf1; n=4, instance_mask=UInt8(0x02))
    Raycore.push_instances!(tlas, blas, instance_buf2; n=4, instance_mask=UInt8(0x04))

    @test_throws ErrorException Raycore.sync!(tlas)
end
