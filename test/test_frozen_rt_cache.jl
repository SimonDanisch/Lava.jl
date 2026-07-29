# Ray-tracing shaders must be frozen too.
#
# `frozen_load`/`frozen_store` were wired into `launch.jl` only — the compute
# path — so `lava_compile_rt_shader` rebuilt every raygen/miss/chit from scratch
# in every session. That is the expensive half of an hw_accel=true scene: crown
# spent ~1063 s starting up against ~7.5 s of actual rendering, and it paid that
# on every run.
#
# `ctx.pipeline_cache` does not cover it. That caches the driver's SPIR-V → ISA
# step, which cannot begin until the SPIR-V exists; everything above it
# (GPUCompiler, the LLVM passes, structurize, the emitter) ran regardless.
#
# The round-trip below is deliberately GPU-free: it exercises the key, the
# serialisation and the memo, which is where the bug was.

using Test, Lava

@testset "frozen RT cache round-trip" begin
    old_version, old_recording = Lava.FROZEN_VERSION[], Lava.FROZEN_RECORDING[]
    try
        Lava.FROZEN_VERSION[] = "rt_test_v1"
        Lava.FROZEN_RECORDING[] = true
        Lava.frozen_clear!()
        Lava.frozen_reset_stats!()

        f = sin                       # any function: the key is types, not code
        tt = Tuple{Float32}
        pinfo = Lava.PushConstantInfo("wrapper_main", 8, 16, [0 => 8, 8 => 8], Int[0, 0])
        shader = Lava.LavaRTShader(UInt8[0x03, 0x02, 0x23, 0x07], :raygen, pinfo,
                                   "some llvm ir that must NOT be stored")

        @test Lava.frozen_rt_load(f, tt, :raygen, :f32, 8) === nothing   # cold
        Lava.frozen_rt_store(f, tt, :raygen, :f32, 8, shader)
        @test Lava.frozen_stats().stores == 1

        # A fresh session has no memo, only the file. This is the path that
        # regressed: the RT memo key has five fields and cannot live in
        # `FROZEN_MEM`, whose key type is fixed at three — storing into it threw
        # `MethodError: Cannot convert Tuple{DataType,DataType,Symbol,Symbol,Int64}`
        # at the first cache hit, so every replay failed to render.
        empty!(Lava.FROZEN_RT_MEM)
        got = Lava.frozen_rt_load(f, tt, :raygen, :f32, 8)
        @test got !== nothing
        @test got.spirv_bytes == shader.spirv_bytes
        @test got.stage === :raygen
        @test isempty(got.ir)                       # IR is dropped on store
        @test Lava.frozen_stats().misses == 0

        # Second load comes from the memo — must not throw, must be the same object.
        @test Lava.frozen_rt_load(f, tt, :raygen, :f32, 8) === got

        # Stage and payload are part of the key: a different stage is a miss.
        @test Lava.frozen_rt_load(f, tt, :miss, :f32, 8) === nothing
        @test Lava.frozen_rt_load(f, tt, :raygen, :u32, 8) === nothing
    finally
        Lava.frozen_clear!()
        Lava.FROZEN_VERSION[]   = old_version
        Lava.FROZEN_RECORDING[] = old_recording
    end
end
