# Kernels compile on several threads at once: Mantle compiles every kernel of a
# plan together (`compilekernels!`), so a scene with a dozen material kernels waits
# for the slowest instead of for all of them in turn. Lava's own state is shared by
# those compiles: the compiler-config cache is locked, the compile counters atomic,
# and the phase log locked with its kernel tag scoped to the compiling task. The tag
# was one global: a compile that started while another ran took over its records,
# so per-kernel phase tables named the wrong kernels.
#
# Device-free, like the rest of this suite. Needs more than one Julia thread to
# overlap anything; with one, the tasks run in turn and this checks only that the
# answer is right.

using Test, Lava, GPUCompiler

# New code every run, as in `test_compile_cache.jl`: a kernel found in the object
# cache compiles nothing and records no phase.
const PARCOMPILE_SALT = Float32(rand(10_000:99_999))

function parcompile_kernel!(out::LavaDeviceArray{Float32,1}, a::LavaDeviceArray{Float32,1},
                            ::Val{K}) where {K}
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(out)
        x = a[i]
        acc = 0f0
        for k in 1:K
            x = muladd(x, Float32(K), Float32(k))
            acc += sin(x) + PARCOMPILE_SALT
        end
        out[i] = acc
    end
    return nothing
end

parcompile_types(K) = Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, Val{K}}

@testset "kernels compiled on several threads at once" begin
    ks = 1:8
    jobs = [Lava.lava_kernel_job(parcompile_kernel!, parcompile_types(K); workgroup_size = (64, 1, 1))
            for K in ks]
    was = Lava.enable_phase_capture!(true)
    Lava.reset_phase_log!()
    parallel = fetch.([Threads.@spawn Lava.compile_or_lookup(job) for job in jobs])
    records = copy(Lava.phase_records())
    Lava.enable_phase_capture!(was)

    @testset "give the SPIR-V a sequential compile gives" begin
        sequential = [Lava.lava_compile_gpu(parcompile_kernel!, parcompile_types(K);
                                            workgroup_size = (64, 1, 1)) for K in ks]
        @test all(((p, s),) -> p.spirv_bytes == s.spirv_bytes, zip(parallel, sequential))
    end

    @testset "record each phase under the kernel that ran it" begin
        # Every compile emits once: one `emit_spirv_from_llvm` record per kernel, and
        # none tagged with another compile's name or none at all.
        emits = [r.kernel for r in records if r.group == "stage" && r.label == "emit_spirv_from_llvm"]
        @test length(emits) == length(jobs)
        @test allunique(emits)
        @test !any(isempty, emits)
    end
end

# Kernels with callees in common, inferred on several threads at once, came out less
# precise: on the docs build a shading kernel had dynamic dispatches it does not have
# inferred alone, cached for the rest of the session. Inference holds
# `Lava.INFERENCE_LOCK`; only what follows it runs alongside other compiles.
@testset "a kernel is inferred while no other is" begin
    job = Lava.lava_kernel_job(parcompile_kernel!, parcompile_types(12); workgroup_size = (64, 1, 1))
    interp = GPUCompiler.get_interpreter(job)
    world = GPUCompiler.get_inference_world(interp)
    inferred() = Core.Compiler.haskey(Core.Compiler.WorldView(Core.Compiler.code_cache(interp), world, world),
                                      job.source)
    compile = lock(Lava.INFERENCE_LOCK) do
        t = Threads.@spawn Lava.compile_or_lookup(job)
        sleep(3)
        @test !inferred()
        t
    end
    @test fetch(compile).spirv_bytes isa Vector{UInt8}
    @test inferred()
end

# The first compiles of a session on several threads at once: GPUCompiler registers
# LLVM's targets in its first `emit_llvm` behind a plain flag, and LLVM's registry is
# a linked list, so two threads both prepended a target, it became its own successor,
# and every target lookup after that (Julia's own JIT's too) looped forever. Lava
# registers them all on one thread when it loads, which leaves GPUCompiler nothing to
# race on. In a process of its own: here, kernels have compiled already.
@testset "LLVM's targets are registered when Lava loads" begin
    script = """
        using Lava, LLVM
        loaded = length(collect(LLVM.targets()))
        LLVM.InitializeAllTargetInfos()
        LLVM.InitializeAllTargets()
        print(loaded, " ", length(collect(LLVM.targets())))
        """
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`, String)
    loaded, registered = parse.(Int, split(out))
    @test loaded == registered
end
