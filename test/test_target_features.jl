"""
`TargetFeatures`: what the device a module is compiled FOR lets it declare.

The emitter has to know two things about the hardware it is emitting for, because
a capability declared on a device that lacks it is a **validation error**, not a
slow path. The record used to be a process global the runtime pushed when it
bound a device, which answered for the bound device rather than the one being
compiled for. It is part of `LavaCompilerParams` now: a compile job carries it,
the frozen keys mix it in, and there is no global to swap.

The assertions are about the SPIR-V, not about the plumbing. Compiling the same
raygen shader with `ser = true` and with `ser = false` has to produce modules that
differ in exactly one way, and without any cache clear between them: two records
are two entries.

The all-`false` default is load-bearing in its own right: it is what lets an
emitter test compile without a device, and it is the module that is valid
everywhere.
"""

using Test, Lava

# One raygen shader, compiled twice. `lava_rt_launch_id_x` is what makes it a
# raygen stage rather than a compute kernel; SER is only ever declared there.
function tf_raygen_kernel(out)
    i = Lava.lava_rt_launch_id_x() + UInt32(1)
    @inbounds i <= length(out) && (out[i] = Float32(1))
    return nothing
end

@testset "TargetFeatures" begin
    @testset "the record itself" begin
        @test Lava.TargetFeatures() == Lava.TargetFeatures(; ser = false, ray_query = false)
        @test Lava.TargetFeatures(; ser = true).ser
        @test !Lava.TargetFeatures(; ser = true).ray_query
        # Part of the job, never a global.
        @test Lava.LavaCompilerParams().features == Lava.TargetFeatures()
        @test Lava.lava_compiler_config(; features = Lava.TargetFeatures(; ser = true)).params.features.ser
        @test !isdefined(Lava, :targetfeatures)
        @test !isdefined(Lava, :TARGET_FEATURES)
        # Content-hashed, so a frozen key that mixes it in is stable across sessions.
        @test hash(Lava.TargetFeatures(; ser = true)) == hash(Lava.TargetFeatures(; ser = true))
        @test hash(Lava.TargetFeatures(; ser = true)) != hash(Lava.TargetFeatures())
    end

    # The point of the whole exercise: the emitted module changes with the job's
    # record, and the two compiles do not need a cache clear between them.
    @testset "SER is declared only when the job's record says so" begin
        tt = Tuple{Lava.LavaDeviceArray{Float32,1}}
        declares_ser(sh) =
            occursin("ShaderInvocationReorder", Lava.disassemble_spirv(sh.spirv_bytes))
        Lava.frozen_rt_clear!()
        try
            on = Lava.lava_compile_rt_shader(tf_raygen_kernel, tt; stage = :raygen,
                                             features = Lava.TargetFeatures(; ser = true))
            off = Lava.lava_compile_rt_shader(tf_raygen_kernel, tt; stage = :raygen,
                                              features = Lava.TargetFeatures(; ser = false))
            @test declares_ser(on)
            @test !declares_ser(off)
            # Both are real modules, so "no capability" is not "no output".
            @test !isempty(on.spirv_bytes)
            @test !isempty(off.spirv_bytes)
            # Two records, two frozen entries.
            @test Lava.frozen_rt_key(tf_raygen_kernel, tt, :raygen, :f32, 8, Lava.TargetFeatures(; ser = true)) !=
                  Lava.frozen_rt_key(tf_raygen_kernel, tt, :raygen, :f32, 8, Lava.TargetFeatures())
            @test Lava.frozen_key(tf_raygen_kernel, tt, (64, 1, 1), Lava.TargetFeatures(; ray_query = true)) !=
                  Lava.frozen_key(tf_raygen_kernel, tt, (64, 1, 1), Lava.TargetFeatures())
        finally
            Lava.frozen_rt_clear!()
        end
    end

    @testset "ray_query is refused rather than emitted" begin
        # `enable_ray_query = true` is a claim about hardware. With the job's
        # record saying the device has none, the compile is refused HERE, where
        # the message can say why, not by the driver at pipeline creation.
        @test_throws "does not support VK_KHR_ray_query" Lava.lava_compile_gpu(
            tf_raygen_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}};
            enable_ray_query = true, features = Lava.TargetFeatures(; ray_query = false))
    end
end
