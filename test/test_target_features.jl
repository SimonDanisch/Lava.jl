"""
`TargetFeatures` — what the bound device lets a SPIR-V module declare.

The emitter has to know two things about the hardware it is emitting for, because
a capability declared on a device that lacks it is a **validation error**, not a
slow path. It used to learn them by reading `VK_CONTEXT_REF[]` and reaching into
a `VkContext`, which was the reason `compiler/` named a Vulkan type at all. Now
the runtime pushes them and the compiler reads a record with no Vulkan in it.

The assertions are about the SPIR-V, not about the plumbing. Compiling the same
raygen shader with `ser = true` and with `ser = false` has to produce modules that
differ in exactly one way — a test that only checked `targetfeatures().ser`
round-trips would pass against an emitter that ignored it entirely.

The all-`false` default is load-bearing in its own right: it is what lets an
emitter test compile without a device, and it is the module that is valid
everywhere.
"""

using Test, Lava

# One raygen shader, compiled twice. `lava_rt_launch_id_x` is what makes it a
# raygen stage rather than a compute kernel — SER is only ever declared there.
function tf_raygen_kernel(out)
    i = Lava.lava_rt_launch_id_x() + UInt32(1)
    @inbounds i <= length(out) && (out[i] = Float32(1))
    return nothing
end

@testset "TargetFeatures" begin
    @testset "the record itself" begin
        # Both default to false: with no device bound, an emitter test gets the
        # conservative module rather than one shaped by hardware that is absent.
        @test Lava.TargetFeatures() == Lava.TargetFeatures(; ser = false, ray_query = false)
        @test Lava.TargetFeatures(; ser = true).ser
        @test !Lava.TargetFeatures(; ser = true).ray_query

        old = Lava.targetfeatures()
        try
            f = Lava.TargetFeatures(; ser = true, ray_query = true)
            @test Lava.targetfeatures!(f) === f
            @test Lava.targetfeatures() === f
        finally
            Lava.targetfeatures!(old)
        end
        @test Lava.targetfeatures() === old
    end

    # The OTHER half of this contract — that the runtime pushes what the device
    # actually reports — is asserted in `Mantle/test/vulkan/test_target_features_push.jl`,
    # because it needs a device to report anything. This file is the compiler's
    # side: given a record, what does it emit.

    # The point of the whole exercise: the emitted module changes.
    @testset "SER is declared only when the record says so" begin
        tt = Tuple{Lava.LavaDeviceArray{Float32,1}}
        declares_ser(sh) =
            occursin("ShaderInvocationReorder", Lava.disassemble_spirv(sh.spirv_bytes))

        old = Lava.targetfeatures()
        try
            # `frozen_rt_clear!` between the two: the RT entries memoise on
            # (function, signature, stage) and know nothing about the feature
            # record, so without it the second compile hands back the first
            # module and the test passes for the wrong reason.
            Lava.targetfeatures!(Lava.TargetFeatures(; ser = true, ray_query = old.ray_query))
            Lava.frozen_rt_clear!()
            on = Lava.lava_compile_rt_shader(tf_raygen_kernel, tt; stage = :raygen)

            Lava.targetfeatures!(Lava.TargetFeatures(; ser = false, ray_query = old.ray_query))
            Lava.frozen_rt_clear!()
            off = Lava.lava_compile_rt_shader(tf_raygen_kernel, tt; stage = :raygen)

            @test declares_ser(on)
            @test !declares_ser(off)
            # Both are real modules, so "no capability" is not "no output".
            @test !isempty(on.spirv_bytes)
            @test !isempty(off.spirv_bytes)
        finally
            Lava.targetfeatures!(old)
            Lava.frozen_rt_clear!()
        end
    end

    @testset "ray_query is refused rather than emitted" begin
        # `enable_ray_query = true` is a claim about hardware. With the record
        # saying the device has none, the compile is refused HERE, where the
        # message can say why — not by the driver at pipeline creation.
        old = Lava.targetfeatures()
        try
            Lava.targetfeatures!(Lava.TargetFeatures(; ser = old.ser, ray_query = false))
            @test_throws "does not support VK_KHR_ray_query" Lava.lava_compile_gpu(
                tf_raygen_kernel, Tuple{Lava.LavaDeviceArray{Float32,1}};
                enable_ray_query = true)
        finally
            Lava.targetfeatures!(old)
        end
    end
end
