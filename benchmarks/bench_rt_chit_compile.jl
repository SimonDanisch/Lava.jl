# bench_rt_chit_compile.jl — what a HW-RT scene spends compiling, per shader.
#
#     julia --project=<VideoEdit root> dev/Lava/benchmarks/bench_rt_chit_compile.jl [scene]
#
# Default scene is `shadow_bumpgold_dome_over_velvet`: a bump-mapped gold
# conductor dome, i.e. Crown's `gold_base_bumped` / `mitra_right_back` material
# shape (Conductor wrapped in BumpMapped) in a 4 kB self-contained scene with a
# procedural checkerboard height field and no image-file dependencies.
#
# ── Run this in a FRESH PROCESS, one scene per process ───────────────────────
#
# This is not a style preference, it is the difference between a number and a
# fiction. Rendering scene B after scene A in the same session lets B reuse
# GPUCompiler's inference cache for every shader the two have in common (raygen,
# miss, and any shared material chit), so B reports a fraction of its real cold
# cost. Measured: the same bump-gold scene costs 12.3 s when run after its
# smooth-gold twin, against its true cold cost below. Anything that compares two
# scenes inside one process is measuring cache residency, not compilation.
#
# The RT path is `lava_compile_rt_shader`, which is separate from the compute
# path's `lava_compile_gpu_from_job` — its phases land in group "pass", not
# "stage", and reading only "stage" makes an RT compile look nearly free.

using Lava
using Hikari
using Printf
using Profile

const SCENES = joinpath(@__DIR__, "..", "..", "Hikari", "test", "pbrt", "scenes")

"""
    rt_compile_profile(scene; samples=1) -> NamedTuple

Render `scene` once with `hw_accel=true` and return the compiler phase records
it produced. One call per process — see the header.
"""
function rt_compile_profile(scene::AbstractString; samples::Int = 1,
                            sampling::Bool = true)
    backend = Lava.LavaBackend()
    path = joinpath(SCENES, scene * ".pbrt")
    isfile(path) || error("no such scene: $path")

    Lava.reset_phase_log!()
    Lava.enable_phase_capture!(true)
    # The sampling profiler answers a different question than the phase table:
    # the table says WHICH phase is slow, `@profile` says which function inside
    # it. Both are needed — a phase table alone sent me chasing a 5-line kernel.
    # It is only trustworthy for in-process work: the spirv-opt / spirv-val
    # phases appear as gaps here, which is what the subprocess ledger is for.
    Profile.clear()
    Profile.init(; n = 10_000_000, delay = 0.005)
    wall = if sampling
        @elapsed Profile.@profile Hikari.render_pbrt(path; backend, samples, hw_accel = true)
    else
        @elapsed Hikari.render_pbrt(path; backend, samples, hw_accel = true)
    end
    Lava.enable_phase_capture!(false)

    return (scene = scene, wall = wall, records = copy(Lava.phase_records()),
            subs = copy(Lava.phase_subprocesses()), sampled = sampling)
end

"Aggregate records by (group, label), largest first."
function by_phase(records)
    totals = Dict{Tuple{String,String}, Tuple{Float64,Int}}()
    for r in records
        k = (r.group, r.label)
        (s, n) = get(totals, k, (0.0, 0))
        totals[k] = (s + r.seconds, n + 1)
    end
    return sort(collect(totals); by = p -> -p.second[1])
end

"Aggregate records by the shader/kernel they belong to, largest first."
function by_kernel(records)
    totals = Dict{String, Float64}()
    for r in records
        # Only top-level phases, or nested passes double-count into their parent.
        r.group == "pass" && r.label in ("run_llvm_passes!",) && continue
        totals[r.kernel] = get(totals, r.kernel, 0.0) + r.seconds
    end
    return sort(collect(totals); by = p -> -p.second)
end

function report(p; top::Int = 16)
    captured = sum(r.seconds for r in p.records; init = 0.0)
    nk = length(unique(r.kernel for r in p.records))
    println("scene            : ", p.scene)
    println("wall             : ", @sprintf("%.1f s", p.wall))
    println("captured by timer: ", @sprintf("%.1f s (%.0f%% of wall)", captured,
                                            100 * captured / p.wall))
    println("shaders compiled : ", nk)
    println()
    println("── phases ", "─"^58)
    @printf("%-50s %8s %6s\n", "group/label", "s", "calls")
    for ((g, l), (s, n)) in first(by_phase(p.records), top)
        s < 0.05 && continue
        @printf("%-50s %8.2f %6d\n", string(g, "/", l), s, n)
    end
    println()
    println("── slowest shaders ", "─"^49)
    for (k, s) in first(by_kernel(p.records), 8)
        s < 0.05 && continue
        @printf("%8.2f s  %s\n", s, first(k, 62))
    end
    if !isempty(p.subs)
        println()
        println("── subprocesses ", "─"^52)
        for (tool, st) in sort(collect(p.subs); by = q -> -q.second.seconds)
            @printf("%-28s %6d spawns %8.2f s %12d bytes\n", tool, st.calls, st.seconds, st.bytes)
        end
    end
    return nothing
end

"""
    print_flamegraph(; n = 45)

The `@profile` layer: flat, self-cost-ordered view of where the compiler's own
CPU time went. Read together with the phase table, never instead of it.
"""
function print_flamegraph(; n::Int = 45)
    println()
    println("── @profile, flat (C frames folded) ", "─"^32)
    io = IOBuffer()
    Profile.print(IOContext(io, :displaysize => (10_000, 200));
                  format = :flat, sortedby = :count, C = false, mincount = 20)
    lines = split(String(take!(io)), '\n')
    for l in first(lines, n)
        println(rstrip(l))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    scene = isempty(ARGS) ? "shadow_bumpgold_dome_over_velvet" : ARGS[1]
    p = rt_compile_profile(scene)
    report(p)
    p.sampled && print_flamegraph()
end
