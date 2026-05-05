using Test, Lava, Raycore
using Lava: LavaInstanceRecord, build_blas_aabb, as_build, AS_INPUT_USAGE,
            write_grain_instances_kernel
using GeometryBasics: Point3f, Vec3f, Vec4f

@testset "Raycore.sync!(HWTLAS) refit cycles" begin
    aabb = Lava.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    radius = 1f0
    quats = Lava.LavaArray([Vec4f(0,0,0,1) for _ in 1:n])
    positions = Lava.LavaArray([Point3f(Float32(3*(i-1)),0,0) for i in 1:n])
    instance_buf = Lava.LavaArray{LavaInstanceRecord}(undef, 2 * n; extra_usage=AS_INPUT_USAGE)
    backend = Lava.LavaBackend()
    bq = Lava.vk_context().default_bq

    write_grain_instances_kernel(backend)(
        positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    Lava.vk_flush!(bq)

    tlas = Lava.HWTLAS(backend)
    push!(tlas, blas, instance_buf; n=2*n, instance_mask=UInt8(0x02))
    Raycore.sync!(tlas)
    pinned_hw_tlas = tlas.hw_tlas

    # Refit cycle: write new positions, mark transforms dirty, sync! (refit path).
    new_positions = Lava.LavaArray([Point3f(Float32(3*(i-1) + 100f0),0,0) for i in 1:n])
    write_grain_instances_kernel(backend)(
        new_positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    Lava.vk_flush!(bq)

    tlas.transforms_dirty = true
    Raycore.sync!(tlas)
    @test tlas.dirty == false
    @test tlas.transforms_dirty == false
    # Refit reuses the same hw_tlas object (no rebuild allocation).
    @test tlas.hw_tlas === pinned_hw_tlas
end
