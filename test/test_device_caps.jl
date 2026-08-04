# `DeviceCaps` — what a kernel asks the device before it picks a tiling.
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
import KernelAbstractions as KA

@testset "DeviceCaps" begin
    ctx = Lava.vk_context()
    c = Lava.caps(ctx)

    @testset "shared memory limit is core Vulkan, always real" begin
        m = Lava.max_shared_memory(ctx)
        @test m isa Int
        @test m > 0
        # Cross-check against an independent read, so a mis-wired field shows up
        # as a mismatch rather than as a plausible number.
        limits = Lava.Vulkan.get_physical_device_properties(ctx.physical_device).limits
        @test m == Int(limits.max_compute_shared_memory_size)
        @test m == c.sharedbudget
        # A workgroup cannot be given more than this; kernels that size
        # `@localmem` against a budget must stay at or under it.
        @test m >= 16 * 1024   # the Vulkan-mandated minimum
    end

    @testset "workgroup limit is queried, not assumed" begin
        # This is the regression the field exists for: it was `Ref(1024)` with a
        # docstring claiming to be the query below. A device whose real limit is
        # not 1024 would have been told it was.
        limits = Lava.Vulkan.get_physical_device_properties(ctx.physical_device).limits
        @test c.workgrouplimit == Int(limits.max_compute_work_group_invocations)
        @test Lava.workgroup_limit(ctx) == c.workgrouplimit
        @test c.workgrouplimit >= 128   # the Vulkan-mandated minimum
    end

    @testset "core count: nothing, or a positive number" begin
        cores = Lava.shader_core_count(ctx)
        @test cores === nothing || (cores isa Int && cores > 0)
        w = Lava.shader_warps_per_sm(ctx)
        @test w === nothing || (w isa Int && w > 0)

        # `nothing`, not `0` — the value is used as a denominator, and a zero
        # there is a silently empty grid rather than a loud failure.
        @test cores !== 0
        @test w !== 0
        @test something(cores, 16) isa Int
        # The struct keeps the raw 0; only the accessors translate it.
        @test c.cores == something(cores, 0)
        @test c.warps == something(w, 0)
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

    @testset "the coopmat width is the PINNED one, not the device default" begin
        # The distinction this field exists to make. On this card they are equal
        # and the bug is invisible; on RDNA 3.5 the device default is 64 while
        # `get_compute_pipeline` pins every coopmat module to 32, so a workgroup
        # sized in units of `subgroup` asks for twice the threads the kernel
        # indexes and writes half its tile.
        @test c.coopmatsubgroup == Lava.COOPMAT_SUBGROUP
        @test c.subgroup == Lava.device_subgroup_size(ctx)
        @test c.subgroup > 0
        # A coopmat kernel's workgroup is a multiple of `coopmatsubgroup`; that
        # multiple has to fit in the device's limit for any of this to launch.
        @test c.coopmatsubgroup <= c.workgrouplimit
    end

    @testset "coopmat availability agrees with the GEMM's own probe" begin
        @test c.coopmat == Lava.coopmat_gemm_available(ctx)
        @test c.tile == Lava.GEMM_TILE
    end

    @testset "queried once, cached on the context" begin
        # Not a performance claim — an identity one. Two reads must be the same
        # object, because a kernel that reads `caps` twice while deciding a
        # tiling must not see two different devices.
        @test Lava.caps(ctx) === c
        @test ctx.caches.caps === c
    end

    @testset "a modified copy leaves the device's own answer alone" begin
        # How a test asks what a kernel would decide on a device it does not
        # have. The copy is a value; nothing about it can reach `ctx`.
        w64 = Lava.DeviceCaps(c; subgroup = 64, coopmat = false)
        @test w64.subgroup == 64
        @test w64.coopmat == false
        @test w64.tile == c.tile
        @test w64.workgrouplimit == c.workgrouplimit
        @test w64.coopmatsubgroup == c.coopmatsubgroup   # pinned, so unchanged
        @test Lava.caps(ctx) === c                        # the device is untouched
        @test Lava.DeviceCaps(c) == c                     # no keywords = no change
    end
end
