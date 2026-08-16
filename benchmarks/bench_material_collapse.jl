# bench_material_collapse.jl — what collapsing the material type explosion cost
# and bought, measured end to end on one scene per process.
#
#     julia --project=<VulkanDev root> dev/Lava/benchmarks/bench_material_collapse.jl <scene.pbrt> <sw|hw>
#
# Prints one JSON-ish line per run: material type count, cold wall time (which
# for a HW-RT scene is almost entirely shader compilation), and warm frame time
# from three subsequent renders.
#
# ── One scene per process, always ────────────────────────────────────────────
#
# Rendering scene B after scene A lets B reuse GPUCompiler's inference cache for
# every shader they share, so B reports a fraction of its real cold cost. The
# same applies to two REVISIONS of one scene: measuring before/after inside a
# session measures cache residency, not compilation. Hence the CLI shape.

using Lava
using Hikari
using Printf

function main(scene_path::String, mode::String)
    hw = mode == "hw"
    backend = Lava.LavaBackend()

    # Scene build first: the material type count is the quantity the refactor
    # targets, and it is known before any shader is compiled.
    build = @elapsed r = Hikari.load_pbrt(scene_path; backend, samples = 1, hw_accel = hw)
    types = [eltype(v) for v in r.scene.materials.static.data]

    println("scene            = ", basename(scene_path))
    println("mode             = ", mode)
    println("build_seconds    = ", round(build, digits = 2))
    println("material_types   = ", length(types))
    for t in types
        println("    ", replace(string(t), "Hikari." => "", "Raycore." => ""))
    end
    flush(stdout)

    # Cold render: shader compilation dominates. Phase capture attributes it.
    Lava.reset_phase_log!()
    Lava.enable_phase_capture!(true)
    cold = @elapsed Hikari.render_pbrt(scene_path; backend, samples = 1, hw_accel = hw)
    Lava.enable_phase_capture!(false)
    println("cold_seconds     = ", round(cold, digits = 2))

    totals = Dict{String, Float64}()
    for rec in Lava.phase_records()
        totals[rec.group] = get(totals, rec.group, 0.0) + rec.seconds
    end
    for (g, s) in sort!(collect(totals); by = last, rev = true)
        @printf("    phase %-10s %8.2f s\n", g, s)
    end
    flush(stdout)

    # Warm frames: same process, everything compiled. Three of them, because the
    # first warm frame still pays one-off buffer allocations.
    warm = Float64[]
    for _ in 1:3
        push!(warm, @elapsed Hikari.render_pbrt(scene_path; backend, samples = 8, hw_accel = hw))
    end
    println("warm_seconds_8spp= ", join(round.(warm, digits = 3), ", "))
    println("warm_best_8spp   = ", round(minimum(warm), digits = 3))
    return nothing
end

main(ARGS[1], ARGS[2])
