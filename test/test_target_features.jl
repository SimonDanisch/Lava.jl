"""
`TargetFeatures`: what the device a module is compiled FOR lets it declare.

The emitter has to know what the hardware it is emitting for supports, because
a capability declared on a device that lacks it is a **validation error**, not a
slow path. As a process global pushed when the runtime bound a device, the
record answers for the bound device rather than the one being
compiled for. It is part of `LavaCompilerParams` now: a compile job carries it,
so it is part of what the compile cache keys on, and there is no global to swap.

The all-`false` default is load-bearing in its own right: it is what lets an
emitter test compile without a device, and it is the module that is valid
everywhere.
"""

using Test, Lava

function tf_compute_kernel(out)
    @inbounds out[1] = 1f0
    return nothing
end

@testset "TargetFeatures" begin
    @testset "the record itself" begin
        @test Lava.TargetFeatures() == Lava.TargetFeatures(; ray_query = false)
        @test Lava.TargetFeatures(; ray_query = true).ray_query
        # Part of the job, never a global.
        @test Lava.LavaCompilerParams().features == Lava.TargetFeatures()
        @test Lava.lava_compiler_config(; features = Lava.TargetFeatures(; ray_query = true)).params.features.ray_query
        # There is no such global, not merely an unused one: no record, no
        # setter for it.
        @test !isdefined(Lava, :targetfeatures)
        @test !isdefined(Lava, :targetfeatures!)
        @test !isdefined(Lava, :TARGET_FEATURES)
        # Content-hashed, so a compiler configuration holding it is the same
        # configuration in every session.
        @test hash(Lava.TargetFeatures(; ray_query = true)) == hash(Lava.TargetFeatures(; ray_query = true))
        @test hash(Lava.TargetFeatures(; ray_query = true)) != hash(Lava.TargetFeatures())
    end

    # A runtime compiles a launch from `lava_kernel_job(...; features = <its
    # device's record>)`, and the kernel cache keys on the job's configuration,
    # so the same kernel for a device with the opposite flag is a different
    # configuration and a different entry. From Mantle's
    # `test/vulkan/test_target_features_push.jl`, which asked it of a live
    # context's record; the record is the only input.
    @testset "a compute job is keyed on its record" begin
        tt = Tuple{Lava.LavaDeviceArray{Float32,1}}
        job(features) = Lava.lava_kernel_job(identity, tt; workgroup_size = (64, 1, 1), features)
        on = Lava.TargetFeatures(; ray_query = true)
        off = Lava.TargetFeatures(; ray_query = false)
        @test job(on).config != job(off).config
        @test job(on).config == job(Lava.TargetFeatures(; ray_query = true)).config
    end

    @testset "ray_query is refused rather than emitted" begin
        # `enable_ray_query = true` is a claim about hardware. With the job's
        # record saying the device has none, the compile is refused HERE, where
        # the message can say why, not by the driver at pipeline creation.
        @test_throws "does not support VK_KHR_ray_query" Lava.lava_compile_gpu(
            tf_compute_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}};
            enable_ray_query = true, features = Lava.TargetFeatures(; ray_query = false))
    end
end
