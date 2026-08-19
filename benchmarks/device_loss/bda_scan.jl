# Lava's `vk_free!` carries the note from the last hunt of this hang: "if it
# recurs, the next thing to check is whether a buffer can be reached by an open
# batch through something `pins` does not count either."
#
# `ctx.diag.freed_bda_scan` is exactly that check — before destroying a buffer it
# scans live arg slabs for its address, which is a reference nothing pinned. Turn
# it on and let the reproducer run: if it fires, the log names the buffer and the
# slab, which is the culprit rather than the symptom.
using Printf
using Lava

ctx = Lava.vk_context()
ctx.diag.freed_bda_scan = true
ctx.diag.free_debug = true
# Logging, not throwing: a throw inside a finalizer is swallowed by Julia's
# finalizer machinery, so the log is what survives to be read.
ctx.diag.destroy_freed_bdas_throws = false
# Observe only: does the fault still happen when the stale BDAs are LEFT in
# place? If yes, the earlier clean run was the scan's slowdown, not its poison.
ctx.diag.freed_bda_scan_poisons = parse(Bool, get(ENV, "HUNT_POISON", "true"))

include(joinpath(@__DIR__, "..", "..", "..", "Hikari", "test", "pbrt", "suite.jl"))

const SCENE = "medium_null_interface_homog"
const SPP = parse(Int, get(ENV, "HUNT_SPP", "64"))
const N = parse(Int, get(ENV, "HUNT_N", "6"))

for k in 1:N
    t = @elapsed rec = render_scene(SCENE; samples = SPP, hw_accel = false)
    s = sum(p -> Float64(red(p)) + Float64(green(p)) + Float64(blue(p)), rec)
    GC.gc(true); GC.gc(true)
    hits = length(ctx.diag.freed_bda_scan_log)
    @printf("%2d  %7.2f s  sum=%.4f  freed-BDA-still-referenced: %d\n", k, t, s, hits)
    flush(stdout)
end

println("\n=== freed BDAs that were still live in an arg slab ===")
for e in ctx.diag.freed_bda_scan_log
    println("  ", e)
end
println("total: ", length(ctx.diag.freed_bda_scan_log))
println("SCAN COMPLETE")
