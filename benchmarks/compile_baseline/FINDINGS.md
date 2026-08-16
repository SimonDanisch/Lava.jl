# Where Lava's kernel compile time goes

Measured on `sd/compiler-optimizations`, base `4b6359c`, with
`benchmarks/bench_compile_time.jl`. Every number here is reproducible with:

```julia
include("dev/Lava/benchmarks/bench_compile_time.jl")
BenchCompileTime.record_baseline()
```

Environment (see `environment.txt` — this is not decoration, see §0):

| | |
|---|---|
| julia | 1.12.6 |
| GPUCompiler | **2.1.1** |
| LLVM.jl | 9.12.0 (LLVM 18.1.7) |
| KernelAbstractions | 0.9.42 |
| device | NVIDIA RTX 4000 Ada Generation |

---

## 0. The GPUCompiler confound — checked first, and it is clear

`GPUCompiler.compile(:llvm)` is phase 1 of the pipeline being profiled, so its
version is a direct confound for "compilation got slower". Lava's `Project.toml`
notes 1.23.0 as validated and ~20% faster end-to-end than 1.13.3, and the
handoff machine had resolved *down* to 1.17.1.

**This machine resolves GPUCompiler 2.1.1**, above that line. None of the
regression is explained by a downgraded GPUCompiler here. No action needed.

## 0b. Two measurement traps that produce wrong numbers

Both were hit while building this harness, both silently:

1. **The first compile in a session is not a compile measurement.** It spends
   most of its wall clock JIT-compiling *Lava's own emitter*. Measured 0.466 s
   for a kernel that costs 0.019 s once warm — a 24× lie, and the phase table
   attributed it to `emit_spirv_from_llvm`, which is exactly where you would
   wrongly start optimising. `warmup!()` exists for this.

2. **A world bump between the two halves of an A/B loop recompiles everything.**
   See §2 — it costs 117 ms, which is larger than most effects being measured.

---

## 1. Axis A — per-kernel compile speed

`depth_sweep.txt`, TILE=16, DEPTH scaling IR from 27 kB to 145 kB:

| DEPTH | IR kB | SPIR-V | total s | gpucomp | passes | emit | lava-own |
|------:|------:|-------:|--------:|--------:|-------:|-----:|---------:|
| 4   | 27.6  | 3124  | 0.025 | 0.009 | 0.001 | 0.000 | 0.016 |
| 32  | 40.4  | 6020  | 0.034 | 0.018 | 0.002 | 0.001 | 0.016 |
| 64  | 55.2  | 9348  | 0.068 | 0.049 | 0.003 | 0.003 | 0.019 |
| 128 | 84.9  | 16004 | 0.137 | 0.102 | 0.003 | 0.017 | 0.034 |
| 256 | 144.9 | 29316 | 0.440 | 0.377 | 0.005 | 0.037 | 0.063 |

**The LLVM passes and the SPIR-V emitter are not the bottleneck.** Across a 5×
IR range, everything Lava controls goes 16 ms → 63 ms, while
`GPUCompiler.compile(:llvm)` goes 9 ms → 377 ms and is ~86% of the largest
compile.

Caveat, stated because it would otherwise overclaim: the `Val{DEPTH}` recursion
that scales IR is itself quadratic for Julia's inference, so that `gpucomp`
column is partly an artifact of the knob. It is *not* evidence that GPUCompiler
is broken. It **is** evidence that Lava's own two stages scale fine and are not
where a "compile times regressed" investigation should start.

The staged GEMM, unmodified and reached through the real KA launch path, agrees:
149 kB IR / 49 kB SPIR-V, cold compile 0.123–0.211 s, of which passes+emit are
~24 ms.

### 1a. What Lava's own cost actually was: fixed per-kernel overhead

For a *small* kernel the Lava-owned cost was ~15 ms and almost none of it was
compilation:

| stage | before |
|---|---:|
| `run_spirv_opt` | 6.91 ms |
| `validate_spirv` | 6.60 ms |
| `run_llvm_passes!` | 1.41 ms |
| `emit_spirv_from_llvm` | 0.66 ms |
| `write .ll` + `string(mod)` + `write .spv` | 1.07 ms |

`spirv-opt` and `spirv-val` are subprocesses whose *actual* work on a 3 kB
module is ~1.3 ms each. The rest was Lava waiting badly.

**Fix 1 — `lava_run` polled with a flat `sleep(0.005)`.** Isolated measurement
of one `spirv-val` spawn on the same file:

| poll | median |
|---|---:|
| `sleep(0.005)` (was) | 6.43 ms |
| `sleep(0.001)` | 2.40 ms |
| `yield()` only | 1.36 ms |
| escalating backoff (now) | **1.36 ms** |

Two spawns per compile, so ~10 ms per kernel of pure rounding. Replaced with a
brief spin then exponential backoff, so a genuinely stuck child still cannot peg
a core for the full 180 s timeout (verified: a 0.4 s child returns in 460 ms
with the right exit code).

**Fix 2 — `string(mod)` and two disk writes ran unconditionally.**
`dev/tmp_kernels` had grown to **5675 files / 275 MB**. Now gated behind
`LAVA_DUMP_KERNELS=1` (implied by `LAVA_DEBUG_PASSES=1` or
`LAVA_SPIRV_DUMP_DIR`). `validate_spirv`'s `llvm_ir` parameter turned out to be
dead entirely.

*This one was not free, and the guardrail is what said so.* The reasoning that
gating `ir` was safe — "both caches already store `""` there" — was true for
cached kernels and **wrong for freshly compiled ones**, which is where
`kernel_source_name` (profiling.jl) had been recovering the kernel's name by
regexing the IR. `test_frozen_kernels_visible.jl` failed on
`occursin("fcpd_scale", k.source)`.

Fixed properly rather than by ungating: `LavaGPUKernel` gained a `source_name`
field holding the mangled GPUCompiler entry symbol — a few dozen bytes, exactly
what the regex was digging out of megabytes of IR, and session-portable, so both
caches now keep it (they still drop the IR). `kernel_source_name` prefers it and
gained a plain-Itanium `_Z<len><name>` case, since a non-KA device function has
neither `CompilerMetadata` nor a `julia_` symbol to match. Pinned by
`test/test_compile_overhead.jl`.

**Result (small kernel): Lava-owned 15.0 ms → 5.8–7.2 ms, bytes written per
compile 37570 → 6232, and codegen byte-for-byte unchanged** (§3).

---

## 2. Axis B — how often that cost is paid at all. This is the real problem.

Same GEMM, same source, same argument types:

| | |
|---|---:|
| warm launch, stable world | **0.49 ms** |
| launch after one world bump | **117 ms** (median 125) |
| penalty | **239×** |
| pipeline actually re-ran? | **yes — all 45 phases** |

A world bump does not merely miss a lookup. It re-runs
`GPUCompiler.compile(:llvm)`, `run_llvm_passes!`, the SPIR-V emitter,
`spirv-opt`, `spirv-val` and the disk writes — **a full recompile of a kernel
that has not changed**. Defining any method anywhere triggers it, for every
kernel, on its next launch. Revise edits, loading a package, defining a helper
in the REPL all qualify.

For scale: this single effect costs more per launch than the *entire* cold
compile of the same GEMM, and axis A's whole fixed overhead is 15 ms against it.

### Root cause

`get_compiled_kernel_and_pipeline` (launch.jl:541) calls
`GPUCompiler.cached_compilation`, whose key is (GPUCompiler 2.1.1,
`deprecated.jl`):

```julia
world = tls_world_age()
key = (objectid(src), world, cfg)
```

The **raw world counter is in the key**, so any bump is a miss even though the
`CodeInstance` is still perfectly valid.

`frozen_cache.jl` does not absorb this and is not meant to: `FROZEN_VERSION[]`
is `""` by default, which disables that cache entirely — it is a
`@compile_workload` mechanism, a different thing.

### The fix, and why it is the supported one

`cached_compilation` is **the legacy API** in GPUCompiler 2.x. Its own docstring
says so: *"This is the legacy caching API used before GPUCompiler 2.0. New code
should use `cached_results`."*

`cached_results(V, job)` looks results up through `job_code_instance(job)` →
`cache_view(job)` → `CacheView(cache_owner(job), job.world)`. That resolves to a
`CodeInstance` whose `[min_world, max_world]` *range* covers `job.world`, so an
unrelated method definition still hits. Method redefinition of the kernel itself
still invalidates the CI, so Revise correctness is preserved — which is the
property the current world-in-key was there to provide, obtained without the
false invalidations.

Sketch (mirrors the `MetalResults` example in GPUCompiler's own docstring;
per-device `linked` because a `LavaLinkedKernel` owns a `VkPipeline`, and
`cached_results` attaches to the CI globally rather than per device):

```julia
mutable struct LavaResults
    compiled::Union{Nothing, LavaGPUKernel}          # session-portable SPIR-V
    linked::Vector{Pair{VkContext, LavaLinkedKernel}} # session-local handles
    LavaResults() = new(nothing, Pair{VkContext, LavaLinkedKernel}[])
end
```

Open questions for that work, both real:
* `clear_kernel_cache!` currently empties a per-device dict; results attached to
  CodeInstances need an equivalent, or the benchmark's cold path and any user
  expectation of it silently stop working.
* GPUCompiler wipes session-dependent results at image-write time via
  `mark_session_dependent!`; check whether Lava's SPIR-V qualifies before
  letting it into a package image.

---

## 3. The guardrail, and a finding it produced by accident

Every change above was checked two ways: the `LAVA_FAST=1` subset, and a SPIR-V
comparison against a recording.

The SPIR-V comparison was initially an exact text diff, and it **reported all
three guard cases changed by an unmodified compiler**. The 16-line diff was
entirely `%7`/`%8`/`%9` renumbering plus one moved `OpTypeInt` line: Julia and
GPUCompiler placed the two `@noinline` helpers into the LLVM module in the other
order, `collect_reachable_callees` walked them in that order, and every `<id>`
downstream shifted.

Within a session, repeat compiles are bit-identical. **Across sessions they are
not.** That contradicts an assumption `lava_disk_cache_store` states it depends
on: *"same (specTypes, workgroup_size) yields the same SPIR-V bytes across
sessions, which lets the driver's persistent VkPipelineCache match by
bit-identical SPIR-V hash."* That assumption does not currently hold, and it is
worth a separate look — a pipeline cache that never matches is invisible.

The guardrail therefore compares an `<id>`- and order-invariant fingerprint
(opcode + literal operands, identity erased, sorted). It catches added, removed
or changed instructions, changed constants, decorations, type widths and
capabilities. It does **not** catch a pure rewiring that preserves every opcode
and literal — covered instead by the numerical check against
`probe_reference` and by the runtime numbers. A guardrail is a net, not a proof;
what it does not catch is stated rather than implied.

Against the pre-change recordings, all three guard cases are **codegen
identical**.
