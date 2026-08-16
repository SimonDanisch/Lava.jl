# bench_compile_time.jl — where Lava's kernel compile time actually goes.
#
# Run from the umbrella project, NOT from Lava's own env (the tests and
# benchmarks are written against the umbrella; see test/runtests.jl preflight):
#
#     julia --project=<VideoEdit root> dev/Lava/benchmarks/bench_compile_time.jl
#
# or, from a live session:
#
#     include("dev/Lava/benchmarks/bench_compile_time.jl")
#     BenchCompileTime.record_baseline()          # writes compile_baseline/
#     BenchCompileTime.check_against_baseline()   # SPIR-V diff + phase compare
#
# ── Two axes, measured separately ────────────────────────────────────────────
#
# Axis A — per-kernel compile speed (LLVM passes + SPIR-V emitter).
# Axis B — how often that cost is paid at all (cache miss / world-age
#          invalidation).
#
# They are NOT interchangeable and the second one is much larger in practice.
# See `compile_baseline/FINDINGS.md` for the numbers this harness produced.
#
# ── Three confounds this harness controls for, all of which produced wrong
#    numbers before they were pinned ─────────────────────────────────────────
#
#  1. GPUCompiler version. `GPUCompiler.compile(:llvm)` is phase 1 of the
#     pipeline and dominates several of these measurements, so its version is a
#     direct confound for "Lava got slower". Recorded in the baseline; refuse to
#     compare across versions.
#
#  2. Julia JIT of Lava itself. The FIRST compile in a session spends most of
#     its wall clock JIT-compiling Lava's own emitter, not compiling the kernel.
#     Measured: 0.466 s for a kernel that costs 0.019 s once warm — a 24x lie.
#     Every measurement below runs `warmup!()` first.
#
#  3. World age. Defining ANY method anywhere invalidates the kernel cache, so
#     an A/B loop that defines a helper between the two halves measures a full
#     recompile against a cache hit. Define everything before measuring.
#
# ── A note on the IR-scaling knob ────────────────────────────────────────────
#
# `Val{DEPTH}` recursion scales IR linearly, which is what separates "slow per
# unit of IR" from "superlinear in IR size". But deep recursive inlining is
# itself quadratic for Julia's inference, so the `GPUCompiler.compile` column in
# the DEPTH sweep is partly an artifact of the knob. Read the Lava-owned columns
# (passes, emit, string, writes, validate) for Lava's own scaling, and use the
# staged GEMM row as the un-doctored large-kernel data point.

module BenchCompileTime

using Lava
using LinearAlgebra
using Printf

const BASELINE_DIR = joinpath(@__DIR__, "compile_baseline")

# ── The benchmark kernel ─────────────────────────────────────────────────────
#
# One source, parameterised on (TILE, DEPTH), hitting the paths the pipeline
# works hardest on:
#
#   * a tiled `@localmem` K-loop            → unroll_loops!
#   * struct argument access (ProbeParams)  → lift_geps, retype_allocas
#   * two `@noinline` helpers               → the multi-OpFunction path
#                                             (force_inline_all=false default)
#   * mixed Int32/Float32 indexing          → the int/float conversion paths
#
# TILE sets the shared-memory loop trip count. DEPTH sets how many fully-inlined
# copies of the body the loop carries, and is the knob that scales IR size.

struct ProbeParams
    scale::Float32
    bias::Float32
    n::Int32
end

# `@noinline` so these survive as their own OpFunctions rather than being
# folded into the entry — that is the path force_inline_all=false takes.
@noinline function probe_stage(acc::Float32, x::Float32, y::Float32, k::Int32)
    return muladd(acc, 1.0009f0, muladd(x, y, Float32(k) * 0.5f0))
end

@noinline function probe_reduce(acc::Float32, v::Float32)
    return acc + v * v
end

@inline probe_chain(::Val{0}, acc::Float32, x::Float32, y::Float32, k::Int32) = acc
@inline function probe_chain(::Val{N}, acc::Float32, x::Float32, y::Float32, k::Int32) where {N}
    acc = probe_stage(acc, x, y, k)
    return probe_chain(Val(N - 1), acc, x * 1.0001f0, y, k + Int32(1))
end

function probe_kernel(A::Lava.LavaDeviceArray{Float32,1},
                      B::Lava.LavaDeviceArray{Float32,1},
                      out::Lava.LavaDeviceArray{Float32,1},
                      p::ProbeParams, ::Val{TILE}, ::Val{DEPTH}) where {TILE, DEPTH}
    ptr  = Lava.lava_alloc_shared(Val(:probe_tile), Float32, Val(TILE))
    tile = Lava.LavaSharedArray{Float32}(ptr, TILE)
    li = Lava.lava_local_invocation_id_x()
    gi = Lava.lava_global_invocation_id_x()
    acc = 0.0f0
    @inbounds begin
        tile[li] = A[gi]
        Lava.lava_workgroup_barrier()
        for k in Int32(1):Int32(TILE)
            acc = probe_reduce(acc, probe_chain(Val(DEPTH), acc, tile[k], B[gi], k))
        end
        out[gi] = acc * p.scale + p.bias
    end
    return nothing
end

probe_tt(TILE, DEPTH) = Tuple{Lava.LavaDeviceArray{Float32,1},
                              Lava.LavaDeviceArray{Float32,1},
                              Lava.LavaDeviceArray{Float32,1},
                              ProbeParams, Val{TILE}, Val{DEPTH}}

# The CPU reference, so a compile-time "win" that changed codegen shows up as a
# wrong number rather than as a silent regression.
function probe_reference(A::Vector{Float32}, B::Vector{Float32}, p::ProbeParams,
                         TILE::Int, DEPTH::Int, wg::Int)
    out = similar(A)
    for gi in eachindex(A)
        block = (gi - 1) ÷ wg
        acc = 0.0f0
        for k in 1:TILE
            tile_k = A[block * wg + k]
            x, y, kk = tile_k, B[gi], Int32(k)
            c = acc
            for _ in 1:DEPTH
                c = muladd(c, 1.0009f0, muladd(x, y, Float32(kk) * 0.5f0))
                x *= 1.0001f0
                kk += Int32(1)
            end
            acc = acc + c * c
        end
        out[gi] = acc * p.scale + p.bias
    end
    return out
end

# ── Measurement ──────────────────────────────────────────────────────────────

"""
    warmup!()

Pay the Julia-JIT-of-Lava cost (confound 2) before anything is timed. Compiles
one small kernel through the full pipeline and throws the result away.
"""
function warmup!()
    Lava.lava_compile_gpu(probe_kernel, probe_tt(8, 2); workgroup_size = (64, 1, 1))
    return nothing
end

"""
    stage_timings(f) -> (result, Dict(label => seconds), Dict(tool => SubprocessStat))

Run `f()` with phase capture on and return the per-stage table it produced.
"""
function stage_timings(f)
    Lava.reset_phase_log!()
    was = Lava.enable_phase_capture!(true)
    result = try
        f()
    finally
        Lava.enable_phase_capture!(was)
    end
    stages = Dict(r.label => r.seconds for r in Lava.phase_table(; group = "stage"))
    subs = copy(Lava.phase_subprocesses())
    return result, stages, subs
end

"""
    measure_ir_bytes(TILE, DEPTH) -> Int

Size of the post-pass LLVM IR, obtained from an untimed compile with kernel
dumping switched on. Separate from the timed compile on purpose — see
`compile_one`. Returns 0 if the dump produced nothing.
"""
function measure_ir_bytes(TILE::Int, DEPTH::Int; workgroup_size = (64, 1, 1))
    dir = mktempdir()
    kernel = withenv("LAVA_DUMP_KERNELS" => "1", "LAVA_DUMP_KERNELS_DIR" => dir) do
        Lava.lava_compile_gpu(probe_kernel, probe_tt(TILE, DEPTH); workgroup_size)
    end
    return sizeof(kernel.ir)
end

"""
    compile_one(TILE, DEPTH) -> NamedTuple

One cold `lava_compile_gpu` with a full phase breakdown. `lava_compile_gpu` goes
straight to `lava_compile_gpu_from_job`, so there is no cache to defeat — each
call genuinely recompiles.
"""
function compile_one(TILE::Int, DEPTH::Int; workgroup_size = (64, 1, 1))
    # One compile, capture on. Capture is a `push!` per phase — far below the
    # resolution of anything measured here — so there is no reason to pay for a
    # second uninstrumented compile just to get `total`.
    #
    # This runs in the PRODUCTION configuration (kernel dumping off), because
    # that is what a user actually pays. The IR size is therefore not available
    # from it — nothing materialises the IR string — so it comes from a separate,
    # untimed compile with the dump switched on. Keeping the two apart is the
    # point: folding the dump into the timed compile would hide the very cost
    # that gating it removed.
    local res, stages, subs
    t = @elapsed begin
        res, stages, subs = stage_timings() do
            Lava.lava_compile_gpu(probe_kernel, probe_tt(TILE, DEPTH); workgroup_size)
        end
    end
    ir_bytes = measure_ir_bytes(TILE, DEPTH; workgroup_size)
    get_s(k) = get(stages, k, 0.0)
    lava_owned = get_s("run_llvm_passes!") + get_s("emit_spirv_from_llvm") +
                 get_s("string(mod)") + get_s("write .ll") + get_s("write .spv") +
                 get_s("run_spirv_opt") + get_s("validate_spirv") +
                 get_s("wrap_entry_for_vulkan!")
    return (TILE = TILE, DEPTH = DEPTH,
            total = t,
            ir_bytes = ir_bytes,
            spirv_bytes = length(res.spirv_bytes),
            gpucompiler = get_s("GPUCompiler.compile(:llvm)"),
            passes      = get_s("run_llvm_passes!"),
            emit        = get_s("emit_spirv_from_llvm"),
            string_mod  = get_s("string(mod)"),
            write_ll    = get_s("write .ll"),
            write_spv   = get_s("write .spv"),
            spirv_opt   = get_s("run_spirv_opt"),
            validate    = get_s("validate_spirv"),
            lava_owned  = lava_owned,
            subprocess_spawns = sum(st.calls for (t, st) in subs if !startswith(t, "write"); init = 0),
            subprocess_bytes  = sum(st.bytes for (_, st) in subs; init = 0))
end

"""
    depth_sweep(depths = (4, 8, 16, 32, 64, 128, 256); TILE = 16) -> Vector

Axis A: does compile time scale linearly with IR, or superlinearly?
"""
function depth_sweep(depths = (4, 8, 16, 32, 64, 128, 256); TILE::Int = 16)
    warmup!()
    return [compile_one(TILE, d) for d in depths]
end

function print_sweep(rows)
    @printf("%-7s %9s %9s %9s %9s %9s %9s %9s %10s\n",
            "DEPTH", "IR kB", "SPIR-V", "total s", "gpucomp", "passes", "emit", "str(mod)", "lava-own")
    println("-"^88)
    for r in rows
        @printf("%-7d %9.1f %9d %9.3f %9.3f %9.3f %9.3f %9.4f %10.3f\n",
                r.DEPTH, r.ir_bytes / 1024, r.spirv_bytes, r.total, r.gpucompiler,
                r.passes, r.emit, r.string_mod, r.lava_owned)
    end
    return nothing
end

# ── The real large kernel: the staged GEMM, unmodified ───────────────────────

"""
    gemm_case(n = 512) -> NamedTuple

`LinearAlgebra.mul!` on `LavaArray`s: a real, unmodified large kernel
(~149 kB IR / ~49 kB SPIR-V at the time of writing) reached through the actual
KA launch path rather than through `lava_compile_gpu` directly.

Returns the cold-compile breakdown plus the three cache states of axis B.
"""
function gemm_case(n::Int = 512)
    A = Lava.LavaArray(rand(Float32, n, n))
    B = Lava.LavaArray(rand(Float32, n, n))
    C = Lava.LavaArray(zeros(Float32, n, n))
    bq = Lava.LavaBackend().bq
    run!() = (LinearAlgebra.mul!(C, A, B); Lava.vk_flush!(bq))

    run!()                                   # warm: compile + JIT Lava itself
    reference = Array(A) * Array(B)
    correct = isapprox(Array(C), reference; rtol = 1.0f-3)

    # cold: kernel cache cleared, so the full pipeline runs again
    Lava.clear_kernel_cache!()
    local cold_stages, cold_subs
    cold = @elapsed begin
        _, cold_stages, cold_subs = stage_timings(run!)
    end

    # warm: stable world, cache hit
    run!()
    warm = minimum(@elapsed(run!()) for _ in 1:20)

    # SPIR-V size comes from the file `validate_spirv` always writes, NOT from
    # the kernel dump — that is gated off in the production configuration these
    # timings are taken in, so reading it from there reports 0.
    return (n = n,
            correct = correct,
            cold = cold,
            warm = warm,
            stages = cold_stages,
            subs = cold_subs,
            spirv_bytes = get(cold_subs, "write lava_last.spv", Lava.SubprocessStat()).bytes)
end

# ── Axis B: world-age invalidation ───────────────────────────────────────────
#
# `get_compiled_kernel_and_pipeline` delegates to `GPUCompiler.cached_compilation`,
# which keys on `(objectid(ci), world, cfg)`. Defining any method anywhere bumps
# `Base.get_world_counter()`, so the key changes and the kernel is compiled again
# from scratch — SPIR-V emission, spirv-opt, spirv-val, disk writes and all —
# even though the source and the argument types are untouched.
#
# `frozen_cache.jl` does NOT absorb this: `FROZEN_VERSION[]` is `""` by default,
# which disables that cache entirely (it is a `@compile_workload` mechanism).

const WORLD_BUMPS = Ref(0)

"""
    world_bump_launch(run!) -> seconds

Define a fresh method (bumping the world age), then time one launch.
"""
function world_bump_launch(run!)
    WORLD_BUMPS[] += 1
    name = Symbol("bench_world_bump_", WORLD_BUMPS[])
    @eval Main $name() = $(WORLD_BUMPS[])
    return @elapsed Base.invokelatest(run!)
end

"""
    world_age_case(n = 512; reps = 8) -> NamedTuple

Axis B, measured against the same GEMM: warm launch on a stable world, versus a
launch after a world bump, versus whether the compiler pipeline actually re-ran.
"""
function world_age_case(n::Int = 512; reps::Int = 8)
    A = Lava.LavaArray(rand(Float32, n, n))
    B = Lava.LavaArray(rand(Float32, n, n))
    C = Lava.LavaArray(zeros(Float32, n, n))
    bq = Lava.LavaBackend().bq
    run!() = (LinearAlgebra.mul!(C, A, B); Lava.vk_flush!(bq))

    run!()
    warm = minimum(@elapsed(run!()) for _ in 1:20)
    bumped = [world_bump_launch(run!) for _ in 1:reps]

    # Did the compiler pipeline genuinely re-run, or is this only lookup cost?
    _, stages, _ = stage_timings() do
        world_bump_launch(run!)
    end

    return (warm = warm,
            bumped_min = minimum(bumped),
            bumped_median = sort(bumped)[cld(reps, 2)],
            bumped = bumped,
            penalty = minimum(bumped) / warm,
            recompiled = !isempty(stages),
            stages = stages)
end

# ── SPIR-V guardrail ─────────────────────────────────────────────────────────
#
# The important half of the guardrail: a compile-time "win" that silently
# changed codegen is exactly how a runtime regression sneaks in, and a timing
# number cannot see it.
#
# ── Why this is not a byte-diff, and not a raw text diff either ──────────────
#
# Repeat compiles WITHIN one session are bit-identical. ACROSS sessions they are
# not: Julia/GPUCompiler can place the two `@noinline` helpers into the LLVM
# module in either order, `collect_reachable_callees` walks them in that order,
# and every SPIR-V `<id>` downstream shifts. Measured on this harness: a raw
# text diff of an unmodified compiler reported all three guard cases "changed",
# with a 16-line diff that was entirely `%7`/`%8`/`%9` renumbering and one moved
# `OpTypeInt` line.
#
# A guardrail with false positives is worse than none — it trains you to ignore
# it, which is precisely when it needs to be believed. So the comparison is on a
# fingerprint that is invariant to <id> numbering and to instruction ordering.
#
# NOTE (worth knowing on its own): this means Lava's SPIR-V is NOT reproducible
# across sessions, which is an assumption `lava_disk_cache_store` states it
# relies on ("same (specTypes, workgroup_size) yields the same SPIR-V bytes
# across sessions, which lets the driver's persistent VkPipelineCache match by
# bit-identical SPIR-V hash"). That assumption does not currently hold.
#
# ── What the fingerprint catches, and what it does not ───────────────────────
#
# Each instruction is reduced to its opcode plus its literal operands, with all
# `<id>` identity erased, and the multiset of those is sorted. It CATCHES: an
# added, removed or changed instruction; a changed opcode mix; a changed
# constant, decoration, type width or capability. It does NOT catch a pure
# rewiring that keeps every opcode and literal identical and only changes which
# value feeds which operand. That residual is covered by the numerical check
# (`probe_reference`) and by the runtime numbers — a guardrail is a net, not a
# proof.

const GUARD_CASES = [(16, 4), (16, 32), (32, 16)]

function disasm_for(TILE::Int, DEPTH::Int)
    res = Lava.lava_compile_gpu(probe_kernel, probe_tt(TILE, DEPTH); workgroup_size = (64, 1, 1))
    return Lava.disassemble_spirv(res.spirv_bytes)
end

guard_name(TILE, DEPTH) = "probe_tile$(TILE)_depth$(DEPTH).spvasm"
fingerprint_name(TILE, DEPTH) = "probe_tile$(TILE)_depth$(DEPTH).fingerprint"

"""
    spirv_fingerprint(disasm) -> String

Order- and `<id>`-invariant summary of a SPIR-V disassembly. See the section
comment above for exactly what this catches and what it lets through.
"""
function spirv_fingerprint(disasm::AbstractString)
    lines = String[]
    for raw in split(disasm, '\n')
        line = strip(raw)
        isempty(line) && continue
        # Debug decoration carries session-specific names and IDs.
        any(startswith(line, p) for p in ("OpLine", "OpSource", "OpString",
                                          "OpName", "OpMemberName", "OpModuleProcessed")) && continue
        # Drop the `%N = ` result binding, then erase every remaining <id>.
        body = replace(line, r"^%\d+\s*=\s*" => "")
        push!(lines, replace(body, r"%\d+" => "%"))
    end
    sort!(lines)
    return join(lines, '\n') * '\n'
end

"""
    record_spirv_baseline(dir = BASELINE_DIR)

Write both the raw disassembly (for a human to read when something does change)
and the fingerprint (what the check actually compares).
"""
function record_spirv_baseline(dir::AbstractString = BASELINE_DIR)
    mkpath(dir)
    for (TILE, DEPTH) in GUARD_CASES
        d = disasm_for(TILE, DEPTH)
        write(joinpath(dir, guard_name(TILE, DEPTH)), d)
        write(joinpath(dir, fingerprint_name(TILE, DEPTH)), spirv_fingerprint(d))
    end
    return nothing
end

"""
    check_spirv_against_baseline(dir = BASELINE_DIR) -> Vector{String}

Recompile every guard case and compare its fingerprint against the recording.
Returns the names whose codegen genuinely changed (empty means unchanged).
"""
function check_spirv_against_baseline(dir::AbstractString = BASELINE_DIR)
    changed = String[]
    for (TILE, DEPTH) in GUARD_CASES
        path = joinpath(dir, fingerprint_name(TILE, DEPTH))
        isfile(path) || error("no SPIR-V fingerprint at $path — run record_baseline() first")
        spirv_fingerprint(disasm_for(TILE, DEPTH)) == read(path, String) ||
            push!(changed, guard_name(TILE, DEPTH))
    end
    return changed
end

# ── Baseline I/O ─────────────────────────────────────────────────────────────

function environment_report()
    io = IOBuffer()
    println(io, "julia          = ", VERSION)
    println(io, "Lava           = ", pkgversion(Lava))
    for name in ("GPUCompiler", "LLVM", "KernelAbstractions", "GPUArrays")
        m = Base.loaded_modules_array()
        idx = findfirst(x -> string(nameof(x)) == name, m)
        idx === nothing && continue
        println(io, rpad(name, 15), "= ", pkgversion(m[idx]))
    end
    ctx = Lava.vk_context()
    println(io, "device         = ", ctx.device_name)
    return String(take!(io))
end

"""
    record_baseline(dir = BASELINE_DIR)

Write phase timings, SPIR-V disassembly, runtime numbers and the versions they
were taken with. The versions are not decoration: `GPUCompiler.compile(:llvm)`
is phase 1 of what is being measured, so a baseline without its version cannot
be compared against anything.
"""
function record_baseline(dir::AbstractString = BASELINE_DIR)
    mkpath(dir)
    warmup!()

    env = environment_report()
    write(joinpath(dir, "environment.txt"), env)

    sweep = depth_sweep()
    open(joinpath(dir, "depth_sweep.txt"), "w") do io
        println(io, env)
        println(io, "# Axis A: per-kernel compile speed vs IR size (TILE=16)")
        println(io, "# gpucomp = GPUCompiler.compile(:llvm); lava-own = everything Lava controls")
        redirect_stdout(io) do
            print_sweep(sweep)
        end
    end

    gemm = gemm_case()
    world = world_age_case()
    open(joinpath(dir, "axis_b_world_age.txt"), "w") do io
        println(io, env)
        @printf(io, "GEMM 512^3 correct        = %s\n", gemm.correct)
        @printf(io, "GEMM cold compile         = %.3f s\n", gemm.cold)
        @printf(io, "GEMM SPIR-V               = %d bytes\n", gemm.spirv_bytes)
        @printf(io, "warm launch, stable world = %.3f ms\n", world.warm * 1e3)
        @printf(io, "launch after world bump   = %.2f ms (median %.2f)\n",
                world.bumped_min * 1e3, world.bumped_median * 1e3)
        @printf(io, "penalty                   = %.1fx\n", world.penalty)
        @printf(io, "pipeline actually re-ran  = %s\n", world.recompiled)
    end

    record_spirv_baseline(dir)
    return (sweep = sweep, gemm = gemm, world = world)
end

"""
    check_against_baseline(dir = BASELINE_DIR)

The guardrail to run after every change: SPIR-V diff first (correctness), then
the phase table (speed). A change that speeds up compilation AND changes the
disassembly is not a win until the disassembly change is explained.
"""
function check_against_baseline(dir::AbstractString = BASELINE_DIR)
    warmup!()
    changed = check_spirv_against_baseline(dir)
    if isempty(changed)
        println("SPIR-V: unchanged vs baseline (", length(GUARD_CASES), " cases)")
    else
        println("SPIR-V: CHANGED in ", length(changed), " case(s): ", join(changed, ", "))
        println("  A compile-time win that changes codegen is a runtime regression until proven otherwise.")
    end
    print_sweep(depth_sweep())
    return changed
end

end # module
