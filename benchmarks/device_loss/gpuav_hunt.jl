# Six hypotheses are dead and the eliminations point one way: identical work,
# varying outcome, so the varying input is the address layout — an out-of-bounds
# access that usually lands in a live buffer and occasionally in an unmapped
# page.
#
# GPU-AV bounds-checks device accesses and reports the offending shader, which
# is the direct test. Narrowed to the medium kernels because the error message
# in `vk_free!` warns that GPU-AV can itself crash on a workload that kills the
# device, and because this scene's distinguishing feature is its medium.
#
# `verify_gpu_av()` first: a layer that is not actually instrumenting reports
# nothing, which is indistinguishable from a clean run.
using Printf
using Lava

shaders = split(get(ENV, "HUNT_SHADERS", "gpu_vp_sample_medium_kernel!"), ',')
Lava.vk_reset_device!(debug = Lava.DebugConfig(validation = true, gpu_av = true,
                                               gpu_av_shaders = String.(shaders),
                                               pool_disabled = true))
@info "GPU-AV" shaders
verified = Lava.verify_gpu_av()
@info "layer actually fires" verified
verified || @warn "GPU-AV did not verify — a silent run below proves nothing"

include(joinpath(@__DIR__, "..", "..", "..", "Hikari", "test", "pbrt", "suite.jl"))

const SCENE = "medium_null_interface_homog"
const SPP = parse(Int, get(ENV, "HUNT_SPP", "64"))
const N = parse(Int, get(ENV, "HUNT_N", "6"))

for k in 1:N
    t = @elapsed render_scene(SCENE; samples = SPP, hw_accel = false)
    @printf("%2d  %6.1f s\n", k, t)
    flush(stdout)
end
println("GPUAV RUN COMPLETE")
