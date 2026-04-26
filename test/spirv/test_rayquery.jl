using Test, Lava
using GeometryBasics: Point3f, Vec3f
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, compile_and_disasm

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

@testset "Ray Query - Phase A2 enable flag emits capability" begin
    # Set ray_query_available=true so the loud-error guard doesn't fire.
    # B1 will probe the real extension and set this permanently on supporting devices.
    ctx = Lava.vk_context()
    saved = ctx.ray_query_available
    ctx.ray_query_available = true
    local d
    try
        function noop_kernel(out)
            @inbounds out[1] = 1.0f0
            return nothing
        end
        d, _ = compile_and_disasm(noop_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}};
                                  stage=:compute, enable_ray_query=true)
    finally
        ctx.ray_query_available = saved
    end
    check(d, "OpCapability RayQueryKHR")
    check(d, "OpCapability RayTracingKHR")
    check(d, "OpExtension \"SPV_KHR_ray_query\"")
    check(d, "OpExtension \"SPV_KHR_ray_tracing\"")
end

@testset "Ray Query - Phase A2 errors loudly without device support" begin
    # Save and restore the field so this test does not poison subsequent tests.
    ctx = Lava.vk_context()
    saved = ctx.ray_query_available
    ctx.ray_query_available = false
    try
        function noop_kernel2(out)
            @inbounds out[1] = 1f0
            return nothing
        end
        @test_throws ErrorException compile_and_disasm(noop_kernel2,
            Tuple{Lava.LavaDeviceArray{Float32,1}};
            stage=:compute, enable_ray_query=true)
    finally
        ctx.ray_query_available = saved
    end
end

@testset "Ray Query - Phase A3 type and TLAS descriptor declared" begin
    ctx = Lava.vk_context()
    saved = ctx.ray_query_available
    ctx.ray_query_available = true
    local d
    try
        function noop_kernel3(out)
            @inbounds out[1] = 1.0f0
            return nothing
        end
        d, _ = compile_and_disasm(noop_kernel3, Tuple{Lava.LavaDeviceArray{Float32,1}};
                                  stage=:compute, enable_ray_query=true)
    finally
        ctx.ray_query_available = saved
    end
    check(d, "OpTypeAccelerationStructureKHR")
    check(d, "OpTypeRayQueryKHR")
    check(d, "DescriptorSet 0")
    check(d, "Binding 0")
end

@testset "Ray Query - Phase A4 OpRayQueryInitializeKHR" begin
    ctx = Lava.vk_context()
    saved = ctx.ray_query_available
    ctx.ray_query_available = true
    local d
    try
        function init_kernel(out)
            ray = Ray(o=Point3f(0, 0, 0), d=Vec3f(0, 0, 1), t_min=0f0, t_max=1f3)
            Lava.lava_ray_query_init(ray; mask=UInt32(0xFF))
            @inbounds out[1] = 1f0
            return nothing
        end
        d, _ = compile_and_disasm(init_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}};
                                  stage=:compute, enable_ray_query=true)
    finally
        ctx.ray_query_available = saved
    end
    check(d, "OpRayQueryInitializeKHR")
end
