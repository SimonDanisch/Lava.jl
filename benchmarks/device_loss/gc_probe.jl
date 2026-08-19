# Keeping every render's objects alive survived 3 renders; dropping them died on
# the 3rd. If the trigger is a free racing work still in flight, then forcing
# collection between renders should make it fail sooner and more often, and
# never collecting should keep it alive.
#
#   HUNT_MODE=gc     drop references and GC.gc(true) between renders
#   HUNT_MODE=keep   hold every render alive
#   HUNT_MODE=drop   drop references, let GC happen whenever (the suite's shape)
using Printf
include(joinpath(@__DIR__, "..", "..", "..", "Hikari", "test", "pbrt", "suite.jl"))

const SCENE = "medium_null_interface_homog"
const SPP = parse(Int, get(ENV, "HUNT_SPP", "64"))
const N = parse(Int, get(ENV, "HUNT_N", "6"))
const MODE = get(ENV, "HUNT_MODE", "gc")

kept = Any[]
for k in 1:N
    t = @elapsed rec = render_scene(SCENE; samples = SPP, hw_accel = false)
    s = sum(p -> Float64(red(p)) + Float64(green(p)) + Float64(blue(p)), rec)
    @printf("%2d  %7.2f s  sum=%.4f\n", k, t, s)
    flush(stdout)
    MODE == "keep" && push!(kept, rec)
    if MODE == "gc"
        GC.gc(true)
        GC.gc(true)   # finalizers queued by the first pass run in the second
    end
end
@printf("SURVIVED %d RENDERS in mode=%s (kept %d)\n", N, MODE, length(kept))
