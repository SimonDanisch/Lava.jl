"""
    PhaseTimer(prefix; threshold=0.5, enabled=ENV["LAVA_PHASE_TIMING"] == "1")

Wall-clock checkpoints for compiler phases. Calling the timer prints the time
since the previous checkpoint, so a slow pass shows up as a large delta; deltas
below `threshold` seconds are noise and stay quiet. `prefix` is prepended to
every line, so nested pipelines can indent themselves under their caller.

    checkpoint = PhaseTimer("[pass] ")
    force_inline_all!(mod, entry_fn); checkpoint("force_inline_all!")

Disabled unless `LAVA_PHASE_TIMING=1`, in which case each call is a no-op.
"""
struct PhaseTimer
    prefix::String
    threshold::Float64
    enabled::Bool
    last::Base.RefValue{Float64}
end

function PhaseTimer(prefix::AbstractString; threshold::Real=0.5,
                    enabled::Bool=get(ENV, "LAVA_PHASE_TIMING", "") == "1")
    return PhaseTimer(prefix, threshold, enabled, Ref(time()))
end

function (timer::PhaseTimer)(label::AbstractString)
    timer.enabled || return
    now = time()
    dt = now - timer.last[]
    dt > timer.threshold && println(timer.prefix, label, " = ", round(dt; digits=2), " s")
    timer.last[] = now
    return
end
