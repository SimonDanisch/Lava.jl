# ── Compiler phase accounting ────────────────────────────────────────────────
#
# Two consumers, one mechanism:
#
#   * `LAVA_PHASE_TIMING=1` prints slow phases as they happen. This is the
#     original behaviour and is what you want when a single compile hangs and
#     you need to see where, live.
#   * `enable_phase_capture!()` accumulates every phase into `PHASE_LOG`
#     regardless of duration, so a benchmark can aggregate across many
#     compiles afterwards. Printing stays off by default here — at capture
#     granularity most phases are microseconds and printing them is noise.
#
# Both are off by default and cost one branch on a `Ref{Bool}` when off.
#
# The subprocess side (`spirv-opt`, `spirv-val`, `spirv-dis`) is accounted
# separately in `PHASE_LOG.subprocesses`, because the interesting quantities
# there are the spawn *count* and the bytes pushed through the filesystem, not
# just wall time — a phase table alone cannot tell you that a compile spawned
# three processes and wrote 400 kB to do it.

"""
One timed compiler phase. `group` separates the pipeline stages that share a
label namespace (`"pass"`, `"emit"`, `"stage"`), `kernel` records which compile
it belonged to so per-kernel and aggregate views are both derivable.
"""
struct PhaseRecord
    group::String
    label::String
    seconds::Float64
    kernel::String
end

"""
Spawn accounting for one external tool. `bytes` is what Lava wrote to disk to
feed the tool (SPIR-V binaries, IR text), which is the cost the tool's own
runtime hides.
"""
struct SubprocessStat
    calls::Int
    seconds::Float64
    bytes::Int
end

SubprocessStat() = SubprocessStat(0, 0.0, 0)

function Base.:(+)(a::SubprocessStat, b::SubprocessStat)
    return SubprocessStat(a.calls + b.calls, a.seconds + b.seconds, a.bytes + b.bytes)
end

"""
The single accumulator for compiler phase timings. Genuinely global: phases are
recorded from deep inside the pass pipeline and the emitter, which have no
channel back to the caller that asked for the measurement.
"""
struct PhaseLog
    records::Vector{PhaseRecord}
    subprocesses::Dict{String, SubprocessStat}
    kernel::Base.RefValue{String}
    capturing::Base.RefValue{Bool}
end

const PHASE_LOG = PhaseLog(PhaseRecord[], Dict{String, SubprocessStat}(), Ref(""), Ref(false))

"""
    enable_phase_capture!(on::Bool = true)

Start (or stop) accumulating compiler phase timings into [`PHASE_LOG`]. Unlike
`LAVA_PHASE_TIMING=1` this records every phase no matter how short and prints
nothing. Returns the previous state so a caller can restore it.
"""
function enable_phase_capture!(on::Bool = true)
    was = PHASE_LOG.capturing[]
    PHASE_LOG.capturing[] = on
    return was
end

"""
    reset_phase_log!()

Drop all accumulated records and subprocess stats.
"""
function reset_phase_log!()
    empty!(PHASE_LOG.records)
    empty!(PHASE_LOG.subprocesses)
    PHASE_LOG.kernel[] = ""
    return nothing
end

"""
    phase_kernel!(name)

Tag subsequent phase records as belonging to the compile of `name`.
"""
function phase_kernel!(name::AbstractString)
    PHASE_LOG.kernel[] = String(name)
    return nothing
end

phase_records() = PHASE_LOG.records
phase_subprocesses() = PHASE_LOG.subprocesses

"""
    record_phase!(group, label, seconds)

Append one phase measurement. Cheap no-op when capture is off.
"""
function record_phase!(group::AbstractString, label::AbstractString, seconds::Float64)
    PHASE_LOG.capturing[] || return nothing
    push!(PHASE_LOG.records, PhaseRecord(String(group), String(label), seconds, PHASE_LOG.kernel[]))
    return nothing
end

"""
    record_subprocess!(tool, seconds, bytes = 0)

Account one external-tool spawn. Always records when capture is on, including
zero-byte spawns, so the *count* stays trustworthy.
"""
function record_subprocess!(tool::AbstractString, seconds::Float64, bytes::Integer = 0)
    PHASE_LOG.capturing[] || return nothing
    key = String(tool)
    prev = get(PHASE_LOG.subprocesses, key, SubprocessStat())
    PHASE_LOG.subprocesses[key] = prev + SubprocessStat(1, seconds, Int(bytes))
    return nothing
end

"""
    timed_phase(f, group, label)

Run `f()` and record its wall time under `group`/`label`. When capture is off
this is `f()` with one extra branch, so it is safe on hot paths.
"""
function timed_phase(f, group::AbstractString, label::AbstractString)
    PHASE_LOG.capturing[] || return f()
    t0 = time()
    result = f()
    record_phase!(group, label, time() - t0)
    return result
end

"""
    phase_table(; group = nothing, kernel = nothing) -> Vector{NamedTuple}

Aggregate the captured records by label, sorted by total time descending.
Filter to one `group` (`"pass"`, `"emit"`, `"stage"`) or one `kernel` first.
"""
function phase_table(; group = nothing, kernel = nothing)
    totals = Dict{Tuple{String, String}, Tuple{Float64, Int}}()
    for r in PHASE_LOG.records
        group === nothing || r.group == group || continue
        kernel === nothing || r.kernel == kernel || continue
        key = (r.group, r.label)
        (t, n) = get(totals, key, (0.0, 0))
        totals[key] = (t + r.seconds, n + 1)
    end
    rows = [(group = g, label = l, seconds = t, calls = n) for ((g, l), (t, n)) in totals]
    sort!(rows; by = r -> -r.seconds)
    return rows
end

"""
    print_phase_table(io = stdout; group = nothing, kernel = nothing, top = 25)

Human-readable view of [`phase_table`](@ref) plus the subprocess ledger.
"""
function print_phase_table(io::IO = stdout; group = nothing, kernel = nothing, top::Int = 25)
    rows = phase_table(; group, kernel)
    total = sum(r.seconds for r in rows; init = 0.0)
    println(io, "phase                                                     calls      s     %")
    println(io, "-"^76)
    for r in first(rows, top)
        pct = total > 0 ? 100 * r.seconds / total : 0.0
        name = string(r.group, "/", r.label)
        length(name) > 52 && (name = name[1:52])
        println(io, rpad(name, 54), lpad(r.calls, 6), lpad(round(r.seconds; digits = 3), 8),
                lpad(round(pct; digits = 1), 6))
    end
    println(io, "-"^76)
    println(io, rpad("TOTAL", 54), lpad("", 6), lpad(round(total; digits = 3), 8))
    if !isempty(PHASE_LOG.subprocesses)
        println(io)
        println(io, "subprocess                       spawns      s     bytes written")
        println(io, "-"^76)
        for (tool, st) in sort(collect(PHASE_LOG.subprocesses); by = p -> -p.second.seconds)
            println(io, rpad(tool, 32), lpad(st.calls, 6), lpad(round(st.seconds; digits = 3), 8),
                    lpad(st.bytes, 14))
        end
    end
    return nothing
end

"""
    PhaseTimer(prefix; group="pass", threshold=0.5, printing=ENV["LAVA_PHASE_TIMING"]=="1")

Wall-clock checkpoints for compiler phases. Calling the timer measures the time
since the previous checkpoint, so a slow pass shows up as a large delta.

Two independent sinks, either or both active:

  * printing — deltas above `threshold` seconds are printed with `prefix`
    prepended, so nested pipelines can indent under their caller. Gated on
    `LAVA_PHASE_TIMING=1`.
  * capture — every delta, regardless of size, is appended to [`PHASE_LOG`].
    Gated on [`enable_phase_capture!`](@ref).

With both off the call is a single branch.

    checkpoint = PhaseTimer("    [pass] ")
    force_inline_all!(mod, entry_fn); checkpoint("force_inline_all!")
"""
struct PhaseTimer
    prefix::String
    group::String
    threshold::Float64
    printing::Bool
    last::Base.RefValue{Float64}
end

function PhaseTimer(prefix::AbstractString; group::AbstractString = "pass",
                    threshold::Real = 0.5,
                    printing::Bool = get(ENV, "LAVA_PHASE_TIMING", "") == "1")
    return PhaseTimer(prefix, group, threshold, printing, Ref(time()))
end

function (timer::PhaseTimer)(label::AbstractString)
    capturing = PHASE_LOG.capturing[]
    (timer.printing || capturing) || return
    now = time()
    dt = now - timer.last[]
    timer.printing && dt > timer.threshold &&
        println(timer.prefix, label, " = ", round(dt; digits = 2), " s")
    capturing && record_phase!(timer.group, label, dt)
    timer.last[] = now
    return
end
