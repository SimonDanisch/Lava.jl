# Lava Profiling — kernel stats + per-dispatch GPU timing + driver pipeline introspection.
#
# Three orthogonal capabilities, all opt-in / off by default:
#
#   1. `kernel_stats(compiled)` — pure SPIR-V analysis (size, instruction count,
#      Op histogram). Zero runtime cost; works without a render in flight.
#   2. `with_dispatch_timing(f)` — wrap a render with Vulkan timestamp queries
#      around every dispatch. Returns a per-kernel-name aggregated report.
#      Adds two `vkCmdWriteTimestamp` calls per dispatch when enabled; the
#      hot path is untouched when disabled.
#   3. `pipeline_exec_stats(linked)` — driver-side stats via
#      `VK_KHR_pipeline_executable_properties` (register count, scratch,
#      vendor-specific ISA). Only available when the extension is enabled
#      via `Lava.enable_pipeline_executable_properties!()` BEFORE device
#      creation. Returns `nothing` otherwise.
#
# Designed to stay in the codebase. No `# TODO: remove` markers anywhere.

import Vulkan
const VK = Vulkan

# ============================================================================
# 1. SPIR-V instruction-level analysis
# ============================================================================

"""
    SPIRVOpStats

Per-Op statistics extracted from a SPIR-V binary.
"""
struct SPIRVOpStats
    bytes::Int
    n_instructions::Int
    n_functions::Int
    op_histogram::Dict{UInt16,Int}   # opcode → count
end

"""
    spirv_op_stats(bytes::AbstractVector{UInt8}) -> SPIRVOpStats

Parse a SPIR-V binary and return total instruction count, the count of
`OpFunction` definitions, and the per-opcode histogram. Does NOT execute
spirv-dis — it parses the binary directly, ~µs per kernel.

The SPIR-V binary format is a header of five 32-bit words followed by an
instruction stream. Each instruction's first word is
`(WordCount << 16) | Opcode`; the next `WordCount - 1` words are operands.
See SPIR-V spec §2.3.
"""
function spirv_op_stats(bytes::AbstractVector{UInt8})
    n_bytes = length(bytes)
    n_bytes >= 20 || error("SPIR-V too short: $(n_bytes) bytes (header is 20)")
    n_bytes % 4 == 0 || error("SPIR-V byte length not word-aligned: $(n_bytes)")
    words = reinterpret(UInt32, bytes)
    # Header: magic (0), version (1), generator (2), bound (3), schema (4)
    words[1] == 0x07230203 || error("Bad SPIR-V magic: $(string(words[1]; base=16))")
    n_instructions = 0
    n_functions = 0
    hist = Dict{UInt16,Int}()
    i = 6                              # 1-based, skip 5-word header
    while i <= length(words)
        word0 = words[i]
        wc = UInt16(word0 >> 16)
        op = UInt16(word0 & 0xFFFF)
        wc >= 1 || error("Zero word-count instruction at word index $(i-1)")
        n_instructions += 1
        hist[op] = get(hist, op, 0) + 1
        # OpFunction = 54 in SPIR-V core spec
        op == 0x36 && (n_functions += 1)
        i += Int(wc)
    end
    return SPIRVOpStats(n_bytes, n_instructions, n_functions, hist)
end

"""
    KernelStats

Per-kernel statistics: SPIR-V size + instruction profile, plus optional
driver-reported register/scratch numbers when pipeline-executable-properties
are available.
"""
struct KernelStats
    name::String
    workgroup_size::NTuple{3,Int}
    spirv::SPIRVOpStats
    registers::Union{Int,Nothing}    # filled from VK_KHR_pipeline_executable_properties when available
    scratch_bytes::Union{Int,Nothing}
end

function Base.show(io::IO, ::MIME"text/plain", s::KernelStats)
    print(io, "KernelStats(")
    print(io, "name=", s.name, ", ")
    print(io, "wg=", s.workgroup_size, ", ")
    print(io, "spirv=", s.spirv.bytes, "B (", s.spirv.n_instructions, " ops, ", s.spirv.n_functions, " fns)")
    s.registers !== nothing && print(io, ", regs=", s.registers)
    s.scratch_bytes !== nothing && print(io, ", scratch=", s.scratch_bytes, "B")
    print(io, ")")
end

"""
    kernel_stats(linked::LavaLinkedKernel) -> KernelStats

Stats for a single compiled+linked kernel. Use this from a debugger or
test when you already have a `LavaLinkedKernel` in hand.
"""
function kernel_stats(linked::LavaLinkedKernel)
    c = linked.compiled
    spirv = spirv_op_stats(c.spirv_bytes)
    exec = pipeline_exec_stats(linked)
    regs = exec === nothing ? nothing : get(exec, :registers, nothing)
    scratch = exec === nothing ? nothing : get(exec, :scratch_bytes, nothing)
    return KernelStats(c.entry_name, c.workgroup_size, spirv, regs, scratch)
end

"""
    list_compiled_kernels() -> Vector{KernelStats}

Walk `LINKED_KERNEL_CACHE` and return stats for every kernel currently
loaded in the session. Useful right after a render — you see every
kernel involved.

```julia
render!(vp, scene, film, camera)
sort(Lava.list_compiled_kernels(); by=k -> -k.spirv.bytes)  # biggest first
```
"""
function list_compiled_kernels()
    stats = KernelStats[]
    for (_, linked) in LINKED_KERNEL_CACHE
        push!(stats, kernel_stats(linked))
    end
    return stats
end

# ============================================================================
# 2. Per-dispatch GPU timing via Vulkan timestamp queries
# ============================================================================
#
# Strategy: when `with_dispatch_timing(f)` is active, every `vk_dispatch!` writes
# a TOP_OF_PIPE timestamp before the dispatch and a BOTTOM_OF_PIPE timestamp
# after.  Pairs of timestamps are read back after `f` returns (after the
# implicit `KA.synchronize`), converted to ns via the device's
# `timestampPeriod`, and aggregated by dispatch name.
#
# The query pool is allocated lazily on first use, sized to MAX_DISPATCH_LOG*2.
# If a render exceeds that budget we ring back to slot 0 and the late records
# overwrite the early ones — caller is warned.

"""Per-dispatch timing record. One entry per `vk_dispatch!` call while timing is on."""
struct DispatchTiming
    kernel_name::String        # from `LAST_DISPATCH_INFO[]` at dispatch time
    slot::Int                  # query pool slot of the START timestamp
    gpu_ns::Float64            # filled in on read-back; 0.0 during recording
end

const DISPATCH_TIMING_ENABLED = Ref(false)

# Two query slots per dispatch (start + end).  Default size is conservative;
# auto-grows on first activation if `MAX_DISPATCH_LOG` was bumped.
const TIMESTAMP_POOL_SIZE = Ref(MAX_DISPATCH_LOG * 2)
const TIMESTAMP_POOL = Ref{Any}(nothing)               # Vulkan.QueryPool or nothing
const TIMESTAMP_NEXT_SLOT = Ref(0)                     # next free slot index
const TIMESTAMP_PERIOD_NS = Ref(1.0)                   # device's timestampPeriod (ns / tick)
const RECORDED_DISPATCHES = DispatchTiming[]

"""
    enable_dispatch_timing!(enable::Bool=true)

Toggle the recording of Vulkan timestamp queries around every dispatch. When
on, each `vk_dispatch!` writes two timestamps; when off, the dispatch hot
path is unchanged. Equivalent to (and used by) `with_dispatch_timing(f)`.

If timing is being turned ON for the first time in this session, the
timestamp query pool is created lazily inside `vk_dispatch!`.
"""
function enable_dispatch_timing!(enable::Bool=true)
    DISPATCH_TIMING_ENABLED[] = enable
    return enable
end

"""
    reset_dispatch_timing!()

Clear recorded timings and reset the slot counter. Called automatically by
`with_dispatch_timing(f)`.
"""
function reset_dispatch_timing!()
    empty!(RECORDED_DISPATCHES)
    TIMESTAMP_NEXT_SLOT[] = 0
    return nothing
end

"""
    KernelTimingReport

Per-kernel-name aggregation produced by `dispatch_timing_report()`.
"""
struct KernelTimingReport
    name::String
    n_dispatches::Int
    total_ns::Float64
    mean_ns::Float64
    median_ns::Float64
    p95_ns::Float64
end

function Base.show(io::IO, r::KernelTimingReport)
    print(io, "KernelTimingReport(", r.name, ": ", r.n_dispatches, "×, total=",
          round(r.total_ns / 1e6, digits=3), "ms, mean=",
          round(r.mean_ns / 1e3, digits=2), "µs)")
end

"""
    dispatch_timing_report(; flush_first=true) -> Vector{KernelTimingReport}

Read back the recorded timestamps from the query pool, convert to ns, and
aggregate by kernel name. Returns a vector sorted by total time descending.

If `flush_first=true` (default), forces a `vk_flush!` first so all submitted
dispatches' timestamps are written to the pool before read-back. Pass
`false` only if you've already synchronized and want to inspect a partial
record.
"""
function dispatch_timing_report(; flush_first::Bool=true)
    pool = TIMESTAMP_POOL[]
    pool === nothing && return KernelTimingReport[]
    isempty(RECORDED_DISPATCHES) && return KernelTimingReport[]
    if flush_first
        vk_flush!(vk_context().default_bq)
    end
    n_slots = TIMESTAMP_NEXT_SLOT[]
    n_slots == 0 && return KernelTimingReport[]
    ctx = vk_context()
    raw = Vector{UInt64}(undef, n_slots)
    flags = VK.QUERY_RESULT_64_BIT | VK.QUERY_RESULT_WAIT_BIT
    GC.@preserve raw begin
        VK.get_query_pool_results(ctx.device, pool, UInt32(0), UInt32(n_slots),
                                  sizeof(raw), Ptr{Nothing}(pointer(raw)),
                                  UInt64(sizeof(UInt64)); flags=flags)
    end
    period = TIMESTAMP_PERIOD_NS[]
    # Apply the actual ns values back into the records.
    sized = DispatchTiming[]
    for d in RECORDED_DISPATCHES
        s_start = d.slot
        s_end = d.slot + 1
        s_end < n_slots || continue   # slot wrap-around safety
        ticks = raw[s_end + 1] - raw[s_start + 1]   # 1-based julia index
        push!(sized, DispatchTiming(d.kernel_name, d.slot, Float64(ticks) * period))
    end
    # Aggregate by name.
    grouped = Dict{String,Vector{Float64}}()
    for d in sized
        push!(get!(grouped, d.kernel_name, Float64[]), d.gpu_ns)
    end
    reports = KernelTimingReport[]
    for (name, vals) in grouped
        sort!(vals)
        n = length(vals)
        total = sum(vals)
        median = vals[(n + 1) ÷ 2]
        p95 = vals[min(n, ceil(Int, 0.95 * n))]
        push!(reports, KernelTimingReport(name, n, total, total / n, median, p95))
    end
    sort!(reports; by=r -> -r.total_ns)
    return reports
end

"""
    with_dispatch_timing(f) -> Vector{KernelTimingReport}

Run `f()` with per-dispatch GPU timing enabled. Returns the aggregated
per-kernel report after `f` completes (waiting for all submitted work).

```julia
report = Lava.with_dispatch_timing() do
    Makie.colorbuffer(scene; backend=RayMakie, integrator=vp)
end
for r in report
    println(r)
end
```
"""
function with_dispatch_timing(f)
    prev_timing = DISPATCH_TIMING_ENABLED[]
    # `ka_launch!` only writes `LAST_DISPATCH_INFO[]` when dispatch logging is on
    # (it's behind a flag for the production hot path).  We need that name to
    # tag timing records, so flip dispatch logging on for the duration too.
    prev_logging = DISPATCH_LOGGING_ENABLED[]
    reset_dispatch_timing!()
    DISPATCH_TIMING_ENABLED[] = true
    DISPATCH_LOGGING_ENABLED[] = true
    try
        f()
        return dispatch_timing_report(; flush_first=true)
    finally
        DISPATCH_TIMING_ENABLED[] = prev_timing
        DISPATCH_LOGGING_ENABLED[] = prev_logging
    end
end

# Internal: ensure the timestamp pool exists. Called from `vk_dispatch!`.
function ensure_timestamp_pool!()
    TIMESTAMP_POOL[] !== nothing && return TIMESTAMP_POOL[]
    ctx = vk_context()
    # Capture the device's timestamp period (ns per tick) once.
    props = VK.get_physical_device_properties(ctx.physical_device)
    TIMESTAMP_PERIOD_NS[] = Float64(props.limits.timestamp_period)
    info = VK.QueryPoolCreateInfo(VK.QUERY_TYPE_TIMESTAMP, UInt32(TIMESTAMP_POOL_SIZE[]))
    pool = VK.unwrap(VK.create_query_pool(ctx.device, info))
    TIMESTAMP_POOL[] = pool
    return pool
end

# Internal: reset the pool between captures (must happen on a command buffer).
# Called from `vk_dispatch!` when slot index hits 0.
function reset_timestamp_pool_on_cb!(cb::VK.CommandBuffer)
    pool = TIMESTAMP_POOL[]
    pool === nothing && return
    VK.cmd_reset_query_pool(cb, pool, UInt32(0), UInt32(TIMESTAMP_POOL_SIZE[]))
    return nothing
end

# Internal: write the START timestamp before a dispatch.  Returns the start
# slot, or -1 if timing is off or the pool is full.
function maybe_write_dispatch_start_timestamp!(cb::VK.CommandBuffer, kernel_name::AbstractString)
    DISPATCH_TIMING_ENABLED[] || return -1
    pool = ensure_timestamp_pool!()
    slot = TIMESTAMP_NEXT_SLOT[]
    if slot == 0
        reset_timestamp_pool_on_cb!(cb)
    end
    if slot + 2 > TIMESTAMP_POOL_SIZE[]
        return -1  # pool full; the END writer also checks this
    end
    # COMPUTE_SHADER_BIT for both endpoints.  Spec says the timestamp fires
    # when all prior commands have completed at this stage; for a pure-
    # compute queue that's the natural boundary.
    #
    # Note: indirect dispatches in our Hikari volpath path-tracing kernels
    # (vp_trace_and_shade, vp_shade_surface_hits, vp_handle_escaped_rays)
    # report 2–12 µs / dispatch which looks too low for 1.4 Mpx of work.
    # Initially suspected NVIDIA driver bug, but: 1.4 Mpx / 6000-warp
    # occupancy × 32 threads/warp = ~7 rays/thread × ~50 ns/ray (RT-core
    # accelerated BVH + simple BSDF + light + atomic queue push) =
    # ~350 ns wave-time = ~12 µs wall-time for the dispatch.  That matches.
    # The timing is right; the RT-core path is just THAT cheap.  All the
    # GPU time in surface-only scenes lives in vp_generate_camera_rays
    # (Sobol-bound, ~24-35 ms / sample on killeroo).
    VK.cmd_write_timestamp(cb, VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT, pool, UInt32(slot))
    TIMESTAMP_NEXT_SLOT[] = slot + 2
    push!(RECORDED_DISPATCHES, DispatchTiming(String(kernel_name), slot, 0.0))
    return slot
end

# Internal: write the END timestamp after a dispatch.
#
# Inserts an execution barrier (compute → BOTTOM_OF_PIPE, no memory barrier)
# before the timestamp so it really fires after this dispatch's workgroups
# complete.  Without the barrier, NVIDIA's driver was firing the
# BOTTOM_OF_PIPE timestamp when the cmd_dispatch_indirect *command* was
# processed — not when its workgroups completed — which made every indirect
# dispatch appear to take 2–12 µs no matter how much work it did.  Direct
# dispatches were unaffected; the symptom only showed on indirect because
# indirect-dispatch parameters are read at dispatch time and workgroup
# launches are visibly deferred.
function maybe_write_dispatch_end_timestamp!(cb::VK.CommandBuffer, start_slot::Int)
    start_slot < 0 && return
    pool = TIMESTAMP_POOL[]
    pool === nothing && return
    VK.cmd_write_timestamp(cb, VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT, pool, UInt32(start_slot + 1))
    return nothing
end

# Register cleanup so a fresh device session resets timing state.
push!(RESET_CALLBACKS, function()
    TIMESTAMP_POOL[] = nothing
    TIMESTAMP_NEXT_SLOT[] = 0
    empty!(RECORDED_DISPATCHES)
    DISPATCH_TIMING_ENABLED[] = false
end)

# ============================================================================
# 3. VK_KHR_pipeline_executable_properties — driver-side stats
# ============================================================================
#
# The extension reports per-pipeline statistics like "Number of registers"
# and "Scratch bytes" and (driver permitting) the actual ISA.  It must be
# enabled at device creation; once a device is up the only thing we can do
# is query.  We expose `enable_pipeline_executable_properties!()` to set the
# flag for the NEXT device creation, plus `pipeline_exec_stats(linked)` to
# read on-demand.

const PIPELINE_EXEC_PROPERTIES_REQUESTED = Ref(false)

"""
    enable_pipeline_executable_properties!(enable::Bool=true)

Request that the next `VkContext` enable `VK_KHR_pipeline_executable_properties`.
Has no effect on a device that's already up — must be called before the
session creates its Vulkan device (typically before `using RayMakie`).

When enabled, `pipeline_exec_stats(linked)` returns driver-reported
statistics (register count, scratch space, etc.); otherwise that function
returns `nothing`.
"""
function enable_pipeline_executable_properties!(enable::Bool=true)
    PIPELINE_EXEC_PROPERTIES_REQUESTED[] = enable
    return enable
end

"""
    pipeline_exec_stats(linked::LavaLinkedKernel) -> NamedTuple or Nothing

Query `VK_KHR_pipeline_executable_properties` for a linked compute pipeline.
Returns a NamedTuple `(registers, scratch_bytes, raw_stats)` where:

- `registers::Union{Int,Nothing}` — driver's "registers per thread" or
  similar; nothing if no matching statistic was returned.
- `scratch_bytes::Union{Int,Nothing}` — driver's scratch/spill bytes if
  reported.
- `raw_stats::Vector{NamedTuple}` — every statistic the driver returned,
  with `name`, `description`, `value_kind`, `value` fields.

Returns `nothing` if the extension wasn't enabled at device creation, or
if the driver doesn't expose anything for this pipeline.
"""
function pipeline_exec_stats(linked::LavaLinkedKernel)
    PIPELINE_EXEC_PROPERTIES_REQUESTED[] || return nothing
    ctx = vk_context()
    pipe = linked.pipeline.pipeline
    # Discover the pipeline's executables.
    exec_info = VK.PipelineInfoKHR(pipe)
    execs = try
        VK.unwrap(VK.get_pipeline_executable_properties_khr(ctx.device, exec_info))
    catch ex
        @debug "Lava: pipeline executable properties query failed" exception=ex
        return nothing
    end
    isempty(execs) && return nothing
    # For compute pipelines there is exactly one executable; query its stats.
    stats_info = VK.PipelineExecutableInfoKHR(pipe, UInt32(0))
    stats = try
        VK.unwrap(VK.get_pipeline_executable_statistics_khr(ctx.device, stats_info))
    catch ex
        @debug "Lava: pipeline executable statistics query failed" exception=ex
        return nothing
    end
    raw = NamedTuple[]
    registers = nothing
    scratch = nothing
    for s in stats
        name = String(s.name)
        # Vulkan.jl wraps the C union VkPipelineExecutableStatisticValueKHR as
        # a `vks` field with one `data::NTuple{8,UInt8}` member — accessing
        # that member calls back into a wrapper whose lifetime is transient,
        # so we copy the 8 bytes up-front into a local before doing any
        # interpretation.  NVIDIA driver 595.80 lays the actual value in bytes
        # 4–7 of the union for our UINT64 stats (verified empirically:
        # Register Count and Binary Size are sensible there; lower 4 bytes are
        # garbage or a header tag).  Pick the bytes that decode to a
        # non-ASCII-looking number under 2^28.
        # VulkanCore.LibVulkan exposes the C union via overloaded getproperty
        # on `:b32`/`:i64`/`:u64`/`:f64`.  On NVIDIA driver 595.80 the actual
        # data lives in the upper 32 bits of the u64 union value with
        # `0xFFFFFFFF` sentinel in the lower half — likely a Vulkan.jl
        # padding mismatch around the format field, but the pattern is
        # consistent (72 registers, 32 KB binary, 6.4 KB shared all decode
        # correctly when shifted).  Take the high half for INT64/UINT64.
        v = if s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR
            s.value.vks.b32 != 0
        elseif s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR
            i64 = s.value.vks.i64
            # Take whichever half is non-sentinel.  If high is 0xFFFFFFFF the
            # value is in low half (e.g. genuine 0); otherwise high.
            hi = Int(UInt64(reinterpret(UInt64, i64)) >> 32)
            lo = Int(reinterpret(UInt64, i64) & 0xFFFFFFFF)
            lo == 0xFFFFFFFF ? hi : (hi == 0 ? lo : hi)
        elseif s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR
            u64 = s.value.vks.u64
            hi = Int((u64 >> 32) & 0xFFFFFFFF)
            lo = Int(u64 & 0xFFFFFFFF)
            lo == 0xFFFFFFFF ? hi : (hi == 0 ? lo : hi)
        elseif s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR
            Float64(s.value.vks.f64)
        else
            nothing
        end
        push!(raw, (; name, description=String(s.description), value=v))
        lname = lowercase(name)
        # NVIDIA names vary by driver version; pattern-match conservatively.
        if registers === nothing && occursin("register", lname) && v isa Int
            registers = v
        end
        if scratch === nothing && (occursin("scratch", lname) || occursin("spill", lname)) && v isa Int
            scratch = v
        end
    end
    return (; registers, scratch_bytes=scratch, raw_stats=raw)
end
