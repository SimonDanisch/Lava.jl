using Test, Lava

@testset "Ray Query - Phase A1 constants" begin
    @test Lava.Op.OpRayQueryInitializeKHR == UInt16(4473)
    @test Lava.Op.OpRayQueryProceedKHR == UInt16(4477)
    @test Lava.Op.OpRayQueryConfirmIntersectionKHR == UInt16(4474)
    @test Lava.Op.OpRayQueryTerminateKHR == UInt16(4475)
    @test Lava.Op.OpRayQueryGetIntersectionTypeKHR == UInt16(4479)
    @test Lava.Op.OpRayQueryGetIntersectionTKHR == UInt16(6018)
    @test Lava.Op.OpRayQueryGetIntersectionInstanceIdKHR == UInt16(6020)
    @test Lava.Op.OpRayQueryGetIntersectionInstanceCustomIndexKHR == UInt16(6019)
    @test Lava.Op.OpRayQueryGetIntersectionPrimitiveIndexKHR == UInt16(6023)
    @test Lava.Op.OpRayQueryGetIntersectionBarycentricsKHR == UInt16(6024)
    @test Lava.Op.OpTypeRayQueryKHR == UInt16(4472)
    @test Lava.Cap.RayQueryKHR == UInt32(4472)
end
