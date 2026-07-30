# The VkPipelineCache must eliminate DRIVER compilation, not just look fast.
#
# Lava has two caches and they answer different questions:
#
#   frozen / disk kernel cache  Julia  -> SPIR-V   (test_frozen_cache.jl, test_disk_cache.jl)
#   VkPipelineCache             SPIR-V -> binary   (here)
#
# Only the second one avoids the driver's shader compiler, and it was the one
# with no coverage: `pipeline_cache.jl` had no registered test at all, just an
# assertion-free MWE.
#
# Timing cannot verify this. A fast second run is equally explained by Lava's
# in-memory `PIPELINE_CACHE` dict, which never touches disk. So this uses
# `pipelineCreationCacheControl` (enabled in vk_device!'s Vulkan 1.3 feature set):
# `no_pipeline_compilation` sets FAIL_ON_PIPELINE_COMPILE_REQUIRED, and the driver
# then REFUSES to build a binary, returning VK_PIPELINE_COMPILE_REQUIRED, instead
# of quietly compiling one. A run that completes did zero driver compilation
# because the driver said so.
#
# The negative control is not optional. A green run against an instrument that
# never fires proves nothing — the same trap `verify_gpu_av` exists to avoid, and
# GPU-AV really has been observed attaching without ever firing. So a kernel the
# driver has never seen must be REFUSED here, or this test is vacuous.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function pcnc_warm!(d)
    i = @index(Global)
    @inbounds d[i] = d[i] + 1.0f0
end

# Body deliberately unlike anything else in the suite, so its pipeline cannot
# already be in the cache from another test.
@kernel function pcnc_novel!(d)
    i = @index(Global)
    @inbounds d[i] = d[i] * 3.0f0 - 7.0f0 + sqrt(abs(d[i]) + 1.5f0)
end

@testset "VkPipelineCache avoids driver compilation" begin
    backend = LavaBackend()

    # ── Cold: compile normally, which populates the VkPipelineCache ──
    a = Lava.LavaArray(zeros(Float32, 64))
    pcnc_warm!(backend, 64)(a; ndrange = 64)
    KA.synchronize(backend)
    @test Array(a) == ones(Float32, 64)

    # ── The instrument must FIRE. A kernel the driver has never compiled has to
    #    be refused; without this the rest of the testset is vacuous. ──
    Lava.PIPELINE_COMPILES_REFUSED[] = 0
    b = Lava.LavaArray(ones(Float32, 64))
    refused = try
        Lava.no_pipeline_compilation() do
            pcnc_novel!(backend, 64)(b; ndrange = 64)
            KA.synchronize(backend)
        end
        false
    catch e
        occursin("PIPELINE_COMPILE_REQUIRED", sprint(showerror, e))
    end
    @test refused                                     # instrument fires
    @test Lava.PIPELINE_COMPILES_REFUSED[] == 1       # ...and is counted

    # ── Warm, same session: the cached pipeline needs no compilation ──
    Lava.PIPELINE_COMPILES_REFUSED[] = 0
    a2 = Lava.LavaArray(zeros(Float32, 64))
    Lava.no_pipeline_compilation() do               # throws if it would compile
        pcnc_warm!(backend, 64)(a2; ndrange = 64)
        KA.synchronize(backend)
    end
    @test Array(a2) == ones(Float32, 64)
    @test Lava.PIPELINE_COMPILES_REFUSED[] == 0

    # ── Across a device reset: the ONLY thing that can satisfy creation now is
    #    the on-disk blob, since the VkPipelineCache is rebuilt from scratch. ──
    ctx = Lava.vk_context()
    path = Lava.lava_pipeline_cache_path(ctx.device_name, string(ctx.driver_version))
    Lava.save_pipeline_cache!(ctx)
    @test isfile(path)
    @test filesize(path) > Lava.PIPELINE_CACHE_HEADER_BYTES

    Lava.vk_reset_device!()

    Lava.PIPELINE_COMPILES_REFUSED[] = 0
    a3 = Lava.LavaArray(zeros(Float32, 64))
    Lava.no_pipeline_compilation() do
        pcnc_warm!(LavaBackend(), 64)(a3; ndrange = 64)
        KA.synchronize(LavaBackend())
    end
    @test Array(a3) == ones(Float32, 64)
    @test Lava.PIPELINE_COMPILES_REFUSED[] == 0     # zero driver compilation
end
