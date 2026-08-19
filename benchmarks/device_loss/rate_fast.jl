# Fast failure-rate probe. The first render pays ~150 s of shader compilation;
# every one after is ~0.5 s and carries roughly a 10 % chance of losing the
# device. So one process amortises the compile and then rolls the dice N times,
# which makes a rate measurable in minutes instead of hours.
#
# Prints SURVIVED or dies. The caller counts.
using Printf
using Lava

# `destroy_buffer!` on a POOLED chunk destroys nothing — it calls
# `return_to_pool!`, which makes that block available to the next `pool_alloc`.
# A still-referenced block handed straight back out looks exactly like this
# fault, and it is the one path deferring every free cannot touch. One VkBuffer
# per allocation removes it.
if get(ENV, "HUNT_NOPOOL", "false") == "true"
    Lava.vk_reset_device!(debug = Lava.DebugConfig(pool_disabled = true))
end

include(joinpath(@__DIR__, "..", "..", "..", "Hikari", "test", "pbrt", "suite.jl"))

# The bounce loop synchronises every EXIT_CHECK_INTERVAL rounds and reads the
# ray queue's BAR counter to decide whether to stop. That is the ONE place the
# host observes device state mid-render, and so the only remaining source of
# run-to-run variation in a workload that is otherwise identical every time.
if get(ENV, "HUNT_NOEARLYEXIT", "false") == "true"
    Hikari.EARLY_EXIT_ENABLED[] = false
end

const SCENE = "medium_null_interface_homog"
const SPP = parse(Int, get(ENV, "HUNT_SPP", "64"))
const N = parse(Int, get(ENV, "HUNT_N", "15"))

for k in 1:N
    rec = render_scene(SCENE; samples = SPP, hw_accel = false)
    k == 1 && @printf("  (first render done, %d to go)\n", N - 1)
    flush(stdout)
end
@printf("SURVIVED %d RENDERS\n", N)
