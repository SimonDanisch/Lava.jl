# test_compile_overhead.jl
#
# Pins two fixed per-kernel overheads that were pure waste — no runtime trade,
# no codegen effect — and that are both easy to reintroduce because neither
# announces itself. See benchmarks/compile_baseline/FINDINGS.md §1a.
#
#  1. `lava_compile_gpu_from_job` used to call `string(mod)` and write a `.ll`
#     and a `.spv` on EVERY compile, ungated. `dev/tmp_kernels` had reached
#     5675 files / 275 MB. On a fat module `string(mod)` alone is seconds (the
#     RT path says so in its own comment).
#
#  2. `lava_run` used to wait on `spirv-opt` / `spirv-val` with a flat
#     `sleep(0.005)`. Those tools exit in ~1.3 ms on a small module, so the
#     poll granularity — not the tool — was the cost: ~10 ms per kernel across
#     the two spawns, which was the single largest Lava-owned cost for a small
#     kernel.
#
# The timing assertions here are deliberately loose (they assert "not the old
# quantisation", not a target number) so they pin the regression without going
# flaky on a loaded machine.

using Test, Lava

@testset "compile-time overhead" begin

    kernel_tt = Tuple{Lava.LavaDeviceArray{Float32,1}, Lava.LavaDeviceArray{Float32,1}}

    function overhead_probe(A::Lava.LavaDeviceArray{Float32,1},
                            B::Lava.LavaDeviceArray{Float32,1})
        i = Lava.lava_global_invocation_id_x()
        @inbounds B[i] = A[i] * 2.0f0 + 1.0f0
        return nothing
    end

    @testset "kernel dumping is off by default" begin
        dir = mktempdir()
        kernel = withenv("LAVA_DUMP_KERNELS" => nothing,
                         "LAVA_DEBUG_PASSES" => nothing,
                         "LAVA_SPIRV_DUMP_DIR" => nothing,
                         "LAVA_DUMP_KERNELS_DIR" => dir) do
            @test !Lava.kernel_dump_wanted()
            Lava.lava_compile_gpu(overhead_probe, kernel_tt; workgroup_size = (64, 1, 1))
        end
        # Nothing on disk, and no IR string materialised.
        @test isempty(readdir(dir))
        @test isempty(kernel.ir)
        # The kernel itself is unaffected — this is overhead, not codegen.
        @test !isempty(kernel.spirv_bytes)
    end

    @testset "LAVA_DUMP_KERNELS=1 still dumps" begin
        dir = mktempdir()
        kernel = withenv("LAVA_DUMP_KERNELS" => "1",
                         "LAVA_DUMP_KERNELS_DIR" => dir) do
            @test Lava.kernel_dump_wanted()
            Lava.lava_compile_gpu(overhead_probe, kernel_tt; workgroup_size = (64, 1, 1))
        end
        files = readdir(dir)
        @test count(endswith(".ll"), files) == 1
        @test count(endswith(".spv"), files) == 1
        @test all(filesize(joinpath(dir, f)) > 0 for f in files)
        @test !isempty(kernel.ir)
    end

    @testset "the other debug switches still imply a dump" begin
        withenv("LAVA_DUMP_KERNELS" => nothing, "LAVA_DEBUG_PASSES" => "1",
                "LAVA_SPIRV_DUMP_DIR" => nothing) do
            @test Lava.kernel_dump_wanted()
        end
        withenv("LAVA_DUMP_KERNELS" => nothing, "LAVA_DEBUG_PASSES" => nothing,
                "LAVA_SPIRV_DUMP_DIR" => mktempdir()) do
            @test Lava.kernel_dump_wanted()
        end
    end

    @testset "the profiler can still name a kernel without the IR" begin
        # Gating the IR string initially blinded `kernel_source_name`, which had
        # recovered the name by regexing the IR. `test_frozen_kernels_visible.jl`
        # caught it. The name now comes from `source_name` (the mangled entry
        # symbol), which is small, session-portable and kept by both caches —
        # so the profiler works whether or not the IR was materialised.
        kernel = withenv("LAVA_DUMP_KERNELS" => nothing,
                         "LAVA_DEBUG_PASSES" => nothing,
                         "LAVA_SPIRV_DUMP_DIR" => nothing) do
            Lava.lava_compile_gpu(overhead_probe, kernel_tt; workgroup_size = (64, 1, 1))
        end
        @test isempty(kernel.ir)                       # IR really is gated off
        @test !isempty(kernel.source_name)             # …but the name survives
        @test Lava.kernel_source_name(kernel) == "overhead_probe"
        # `entry_name` is the Vulkan entry point and cannot distinguish kernels;
        # that is exactly why `source_name` has to exist.
        @test kernel.entry_name == "main"
    end

    @testset "lava_run reports exit status correctly" begin
        ok = Lava.lava_run(`true`; label = "test-true")
        @test process_exited(ok)
        @test ok.exitcode == 0

        bad = Lava.lava_run(`false`; label = "test-false")
        @test process_exited(bad)
        @test bad.exitcode != 0

        # A child slower than the initial spin must still be waited for, not
        # abandoned — the backoff has to keep polling, not give up.
        slow = Lava.lava_run(`sleep 0.3`; label = "test-slow")
        @test process_exited(slow)
        @test slow.exitcode == 0
    end

    @testset "lava_run does not quantise short waits to 5 ms" begin
        # The old flat `sleep(0.005)` made every spawn cost >= ~5 ms regardless
        # of how fast the child was. Assert we are well under two of those.
        best = minimum(@elapsed(Lava.lava_run(`true`; label = "test-fast")) for _ in 1:15)
        @test best < 0.004
    end
end
