# `DeviceCompute` — the SM/CU count, warps per SM, and shared-memory ceiling.
#
# Written against whatever the running device reports rather than a vendor's
# numbers, so it means the same thing everywhere. The one thing it does assert
# unconditionally is the failure mode that produced this code:
#
# `vkGetPhysicalDeviceProperties2` fills its `pNext` chain only if the
# application opted into it — core in Vulkan 1.1, or the
# `VK_KHR_get_physical_device_properties2` instance extension before that.
# Without either, the driver populates the *base* struct correctly and silently
# drops the chain, and the function returns void so there is no status to check.
# Measured on an RTX 4000 Ada: a 1.0 instance with no extension reads
# `shader_sm_count = 0` where a 1.1+ instance reads 48.
#
# Lava's instance asks for 1.4, so an advertised extension must yield a non-zero
# count. If that ever regresses, every launch heuristic downstream quietly falls
# back to its default instead of failing, which is why this is a test and not a
# comment.

using Test, Lava

@testset "DeviceCompute" begin
    ctx = Lava.vk_context()

    @testset "shared memory limit is core Vulkan, always real" begin
        m = Lava.max_shared_memory(ctx)
        @test m isa Int
        @test m > 0
        # Cross-check against an independent read, so a mis-wired field shows up
        # as a mismatch rather than as a plausible number.
        limits = Lava.Vulkan.get_physical_device_properties(ctx.physical_device).limits
        @test m == Int(limits.max_compute_shared_memory_size)
        # A workgroup cannot be given more than this; kernels that size
        # `@localmem` against a budget must stay at or under it.
        @test m >= 16 * 1024   # the Vulkan-mandated minimum
    end

    @testset "core count: nothing, or a positive number" begin
        c = Lava.shader_core_count(ctx)
        @test c === nothing || (c isa Int && c > 0)
        w = Lava.shader_warps_per_sm(ctx)
        @test w === nothing || (w isa Int && w > 0)

        # `nothing`, not `0` — the value is used as a denominator, and a zero
        # there is a silently empty grid rather than a loud failure.
        @test c !== 0
        @test w !== 0
        @test something(c, 16) isa Int
    end

    @testset "an advertised extension must actually fill the chain" begin
        nv = Lava.has_extension(ctx.physical_device, "VK_NV_shader_sm_builtins")
        amd = Lava.has_extension(ctx.physical_device, "VK_AMD_shader_core_properties2")
        if nv || amd
            @test Lava.shader_core_count(ctx) !== nothing
            nv && @test Lava.shader_warps_per_sm(ctx) !== nothing
        else
            @info "device reports no SM/CU count extension; count is expected to be unknown" device=ctx.device_name
            @test Lava.shader_core_count(ctx) === nothing
        end
    end

    @testset "the unknown device is representable" begin
        d = Lava.DeviceCompute()
        @test d.sm_count == 0
        @test d.warps_per_sm == 0
        @test d == Lava.DeviceCompute(0, 0, 0)
    end
end
