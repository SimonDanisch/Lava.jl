# bench_render_time.jl — steady-state render time, with the scene built once.
#
#     julia --project=<VulkanDev root> dev/Lava/benchmarks/bench_render_time.jl <scene.pbrt> <sw|hw> [spp]
#
# Separate from bench_material_collapse.jl on purpose. `Hikari.render_pbrt`
# parses and builds the scene on every call, and Crown's build alone is ~67 s —
# an order of magnitude more than the render it wraps. Timing repeated
# `render_pbrt` calls therefore measures the pbrt parser, not the renderer.
# Here the scene is built once and only `render!` is timed.

using Lava
using Hikari
using Printf

function main(scene_path::String, mode::String, spp::Int)
    hw = mode == "hw"
    backend = Lava.LavaBackend()

    r = Hikari.load_pbrt(scene_path; backend, samples = spp, hw_accel = hw)
    types = [eltype(v) for v in r.scene.materials.static.data]
    println("scene          = ", basename(scene_path))
    println("mode           = ", mode, "  (per-sample timings)")
    println("material_types = ", length(types))

    st = r.integrator_settings
    integrator = Hikari.VolPath(; samples = spp,
                                  max_depth = st.max_depth,
                                  regularize = st.regularize,
                                  russian_roulette_depth = st.russian_roulette_depth,
                                  max_component_value = st.max_component_value,
                                  hw_accel = hw,
                                  sensor = r.sensor)
    # `render!` draws ONE sample and accumulates, so these are per-sample times.
    # First call compiles; the ones after it are the measurement.
    first = @elapsed Hikari.render!(integrator, r.scene, r.film, r.camera)
    println("first_seconds  = ", round(first, digits = 2), "   (includes shader compilation)")

    times = Float64[]
    for _ in 1:5
        push!(times, @elapsed Hikari.render!(integrator, r.scene, r.film, r.camera))
    end
    sort!(times)
    @printf("sample_seconds = %s\n", join(round.(times, digits = 4), ", "))
    @printf("sample_best    = %.4f\n", times[1])
    @printf("sample_median  = %.4f\n", times[3])
    return nothing
end

main(ARGS[1], ARGS[2], length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 32)
