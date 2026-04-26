# GPU-AV regression test
#
# Phase-I of the Hikari GPU stability investigation
# (`docs/specs/2026-04-25-unaligned-bda-investigation.md`).
#
# Runs a minimal Hikari HW RT render under the Vulkan validation layer with
# GPU-assisted instrumentation enabled, then asserts that no validation
# messages were captured.  This pins the unaligned-BDA-store fix in Lava's
# SPIR-V emitter (`psb_needs_decomposition`): historically the emitter
# returned false for `access_align <= 4` AND the byte-offset check used
# `if offset > 0` which silently dropped negative offsets emitted by SROA's
# end-relative pointer arithmetic.  Both gaps allowed `store i32 align 4`
# to land on a 2-aligned BDA, which AMD tolerates but other vendors may
# not — and which corrupts GPU memory state enough to trigger RADV GPUVM
# faults after many Hikari renders.  See the investigation doc for the
# full root-cause trace.
#
# # Why this test is gated
#
# GPU-AV slows compute dispatches by ~100x.  The test takes minutes per
# render, so it's gated on `LAVA_GPU_AV=1` and excluded from the default
# `runtests.jl` flow.  CI should run a dedicated job with that env var set.

using Test
using Lava

if get(ENV, "LAVA_GPU_AV", "0") != "1"
    @info "test_gpuav_clean: skipped (set LAVA_GPU_AV=1 to enable; see test header)"
else
    using Hikari, Raycore, Adapt
    using GeometryBasics
    using GeometryBasics: normal_mesh, Tesselation, Sphere, Point3f, Point2f, Vec3f
    using LinearAlgebra: I

    @testset "GPU-AV clean — minimal Hikari HW RT render" begin
        backend = Lava.LavaBackend()
        ctx = Lava.vk_context()

        # Tiny scene + tiny film keeps the GPU-AV-instrumented run finite.
        scene = Hikari.Scene(; backend=backend, hw_accel=true)
        sphere = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.35), 0.35f0), 8))
        push!(scene, sphere, Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
        push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
        Hikari.sync!(scene)

        film = Hikari.Film(Point2f(8, 8))
        gpu_film = Adapt.adapt(backend, film)
        camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5), Point3f(0, 0, 0.35),
                                          film; fov=50f0)

        empty!(Lava.VALIDATION_MESSAGES)
        vp = Hikari.VolPath(samples=1, max_depth=1, hw_accel=true)
        vp(scene, gpu_film, camera)
        close(vp)

        # `Lava.VALIDATION_MESSAGES` collects setup-noise warnings (validation
        # layer adjusts settings on init) alongside real errors.  Filter to
        # actual validation issues — the setup noise is harmless.
        real_messages = filter(Lava.VALIDATION_MESSAGES) do m
            !occursin("adjusting settings", m) &&
            !occursin("VALIDATION-SETTINGS", m) &&
            !occursin("Loader Message", m)
        end

        # Log every captured message verbatim before asserting, so a
        # regression points at the offending shader text immediately.
        for m in real_messages
            @info "GPU-AV captured" m
        end

        # Contract: zero validation messages from a real Hikari HW RT render.
        # If this fails, the SPIR-V emitter has regressed on alignment-tracking
        # for byte-offset GEPs — re-investigate via:
        #   ENV["LAVA_SPIRV_DUMP_DIR"] = "/tmp/lava_spv_debug"
        #   ENV["LAVA_GPU_AV"] = "1"
        # then disassemble + scan via the script at the bottom of
        # docs/specs/2026-04-25-unaligned-bda-investigation.md.
        @test isempty(real_messages)
        # Also assert the device didn't actually go lost during the render.
        @test !Lava.device_lost(ctx)
    end
end
