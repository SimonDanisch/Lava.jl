using Test, Lava
using GeometryBasics: Point3f, Vec3f

@testset "GeometryType - types and constructors" begin
    tri = TrianglesGeometry(; vertex_format=UInt32(103),
                              vertex_addr=UInt64(0x1000),
                              vertex_stride=UInt64(12),
                              max_vertex=UInt32(2),
                              index_type=UInt32(1),
                              index_addr=UInt64(0x2000))
    @test tri isa GeometryType
    @test tri isa TrianglesGeometry
    @test tri.vertex_addr == UInt64(0x1000)

    ab = AABBsGeometry(; aabb_addr=UInt64(0x3000), aabb_stride=UInt64(24))
    @test ab isa GeometryType
    @test ab isa AABBsGeometry
    @test ab.aabb_stride == UInt64(24)
end

@testset "AABB struct" begin
    a = AABB(Point3f(0, 0, 0), Point3f(1, 2, 3))
    @test a.min == Point3f(0, 0, 0)
    @test a.max == Point3f(1, 2, 3)
end

@testset "Ray struct" begin
    r = Ray(Point3f(0, 0, 0), Vec3f(0, 0, 1), 0f0, 1f3)
    @test r.origin == Point3f(0, 0, 0)
    @test r.direction == Vec3f(0, 0, 1)
    @test r.tmin == 0f0
    @test r.tmax == 1f3
end
