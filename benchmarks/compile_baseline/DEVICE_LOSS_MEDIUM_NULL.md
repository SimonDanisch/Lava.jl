# Open: intermittent device loss, and it is a collection landing mid-recording

Status: **reproducible in ~80 s, attributed to a class, not yet fixed.** Not
caused by the shading-path unification — it fired hours before that work
existed, and the scene it fires on contains none of the materials that change
touched.

## The fault

```
LavaError during vkWaitSemaphores:
  device was lost while waiting for timeline 999 (counter stuck at 997, 2 batch(es) in flight)
```

Always `medium_null_interface_homog`, software BVH. Never the medium scenes
around it, which run immediately before it in the suite and pass every time.

## Reproducer

```
HUNT_N=4 HUNT_SPP=64 julia --project=. /sim/tmp/bench/skip_probe.jl
```

~80 s: render 1 pays shader compilation, renders 2-3 are ~2 s each, and the
device goes somewhere in the first few. `/sim/tmp/bench/hunt_devlost.jl` is the
same thing at 256 spp.

## Rate

| workload | failures |
|---|---|
| full pbrt suite, SW, 256 spp | 2 / 4 runs |
| repeated renders of the scene alone | ~1 / 10 |

## What it is not

* **Not the shading maths.** Sampling is deterministic — ZSobol, fixed seed,
  fixed spp — so every render of this scene submits identical work. A fixed
  workload that fails one run in five is a race, not a data-dependent hang.
  (The framebuffer sums are bit-identical across renders, which confirms the
  determinism rather than indicating anything skipped: one ray per pixel per
  sample, accumulated in dispatch order.)
* **Not an unbounded loop.** Delta tracking is bounded twice over —
  `max_segments = 256` over majorant segments, `max_samples = 1024` inside one —
  and `σ_maj < 1e-10` short-circuits. The alpha-test loop caps at 16. The
  null-material crossing does not increment path depth, but the host bounce loop
  caps rounds regardless.
* **Not reported by synchronization validation.** A full run under
  `DebugConfig(validation = true, sync_val = true, pool_disabled = true)` emitted
  zero hazards and still lost the device. Whatever this is, the layer does not
  see it as a missing barrier between two accesses it can pair up.

## What it is

The trigger is *when* a collection happens, not that one happens:

| references | collection | outcome |
|---|---|---|
| dropped | whenever the GC feels like it | **device lost on render 3** |
| dropped | forced `GC.gc(true)` between renders | 6 / 6 survived |
| held alive | never | 6 / 6 survived |

Forcing collection at a quiet point is safe. Letting it land wherever it likes
is not. Which is precisely the window `vk_free!` already documents:

> **Not certainly the last of it.** After this landed the hang was seen once
> more … roughly 90 clean trials across every reproduction that used to fail in
> ten or fewer. So the dominant path is closed and the residual rate is low, but
> either a second window exists or something rarer shares this one. If it
> recurs, the next thing to check is whether a buffer can be reached by an open
> batch through something `pins` does not count either.

This is that recurrence, on a workload that reaches it in 80 s rather than
needing 90 trials.

## The scan fires, in volume

`ctx.diag.freed_bda_scan` answers the comment's question directly: before
destroying a buffer it walks live arg slabs for its address, which is a
reference nothing pinned. Run over the reproducer
(`/sim/tmp/bench/bda_scan.jl`):

| render | freed BDAs still present in a live arg slab |
|---|---|
| 1 | 4372 |
| 2 | 8731 |
| 3 | device lost |

So the condition the comment predicted is not rare — it is thousands of slots
per render, and it grows.

**What this does NOT yet establish.** `scan_arg_slabs_for_bda!` walks every live
slab and does not ask whether that slab is still IN FLIGHT. A retired slab
holding a stale address is harmless; only one an unretired batch will submit is
a use-after-free. Lava recycles slabs, so an unknown share of these 4372 are
stale-but-retired. Qualifying the scan by in-flight status is what turns this
count into a diagnosis, and is the next step.

The scan also zeroes each hit it finds (`unsafe_store!(p, UInt64(0), k+1)`),
which is deliberate — BDA_POISON, so a shader dereferencing it faults on null
rather than on recycled memory. Worth knowing when reading the counts: the same
slot cannot be counted twice across renders.

## The stale entries are real; poisoning them is NOT what prevents the hang

With the scan's own crash fixed (below), it runs to completion — and the run
does not lose the device:

| render | freed BDAs in live slabs | new this render |
|---|---|---|
| 1 | 4369 | — |
| 2 | 8728 | +4359 |
| 3 | 13087 | +4359 |
| 4 | 17446 | +4359 |
| 5 | 21805 | +4359 |
| 6 | 26164 | +4359 |

Two things in one table. The rate is EXACTLY constant — 4359 fresh stale slots
per render — so this is a fixed set of buffers per render, not drift. And six
renders survived, where the same workload without the scan loses the device on
the third.

The scan zeroes every hit (`unsafe_store!(p, UInt64(0), k+1)` — BDA_POISON), so
the obvious reading is that it neutralises the stale references and that is why
the run survives. **That reading is wrong**, and the test that settles it is one
flag: `freed_bda_scan_poisons = false` makes the scan a pure observer.

| poisoning | renders | outcome |
|---|---|---|
| on  | 6 | survived, 26164 hits |
| off | 6 | survived, 26294 hits |

Identical. Leaving every stale BDA in place survives just as well, so what
prevented the fault was not the poisoning — it was the scan's cost. Each render
takes 42 s under the scan against 0.46 s without it, 90x, because it walks every
slab on every free. That shifts whatever window the collector and the recording
overlap in, and the fault needs that window.

So the stale entries are a real and precisely-rated finding, and they are NOT
yet shown to cause this. Which fits everything else here: fast renders die, slow
ones do not; uncontrolled collection dies, controlled collection does not. The
fault is timing, and any instrument heavy enough to observe it changes the
timing enough to hide it. That is the actual difficulty of this bug, and it is
why the next step is not another scan.

## A bug in the instrument

The scan is not finalizer-safe. `memory.jl:829` reads `slab.buf[]`, and when
that slab's own `DataRef` has already been released the getindex throws inside
`vk_free!`:

```
error in running finalizer: Core.ArgumentError("Attempt to use a freed reference.")
  getindex at GPUArrays/src/host/abstractarray.jl:73
  scan_arg_slabs_for_bda! at runtime/memory.jl:829
  vk_free! at runtime/memory.jl:723
```

Diagnostic-only — the flag is off by default — but it means the scan cannot be
left on for a long run, and the slab list it walks needs the same liveness
qualifier as the hits it reports.

Worth pairing with `ctx.diag.free_debug`, whose log records for each free
whether a batch was recording at the time — the two together should say whether
the buffer was freed during a recording and then named by it.

## A fast probe, and how to run a variant against it without ruining it

The compile dominates: ~150 s for the first render, ~0.5 s for each one after.
So one process amortises it and then rolls the dice N times —
`/sim/tmp/bench/rate_fast.jl` with `HUNT_N=15`, driven by
`/sim/tmp/bench/rate_trials.sh <label> <trials>`. Measured on that probe: one
trial survived 15 renders, the next lost the device, so a trial of 15 is roughly
a coin flip and a handful of trials per arm is enough to see a real difference.

**The mistake to avoid, having made it.** A first attempt to measure
defer-always against baseline produced nothing usable, because the baseline
driver was still running when its source was patched and the variant driver was
started alongside it. Two arms then shared a GPU and one of them changed
compilation unit mid-run. Discarded.

Rules that follow:

* one arm at a time, and confirm the previous driver has EXITED (not just that
  its log stopped growing — these runs are slow, and a quiet log is not a dead
  process);
* never edit `src/` while any trial is in flight, since each trial is a fresh
  process that picks up whatever is on disk when it starts;
* run the trials in the FOREGROUND. Background shells here are killed after
  roughly eight minutes, which silently truncates an arm to however many trials
  fit — and a truncated arm looks like a clean one.

The variant itself is kept at `/sim/tmp/bench/apply_defer_always.py`: it makes
`vk_free!` never destroy inline, on the theory that the four defer branches all
read a point-in-time fact and then fall through to a destroy nothing serialises
against a recording starting immediately after.

## Ruled out: deferring every free

The hypothesis was that `vk_free!`'s four defer branches each read a
point-in-time fact — pinned, in flight, batch recording, wrong thread — and then
fall through to a `destroy_buffer!` that nothing serialises against a recording
STARTING immediately afterwards. Removing the fall-through entirely closes that
gap: hand every free to the owning thread's deferred list and let
`drain_deferred_frees!` do it at a flush or submit boundary, where no recording
is open by construction.

Measured on the fast probe, one arm at a time, 15 renders per trial:

| arm | lost | survived |
|---|---|---|
| baseline | 5 | 1 |
| defer-always | 4 | 2 |

No difference. So the inline destroy is not the window, and destruction TIMING
is not what this is — deferring everything to a quiet boundary fails at the same
rate as destroying immediately.

That is worth as much as a fix would have been, because it is the hypothesis the
existing comment points at and it would have cost a day. What it leaves: the
corruption is not caused by WHEN the buffer is destroyed, so either it is not
destruction at all (the pool handing a still-referenced block back out through
`return_to_pool!` would look identical and is untouched by deferring), or the
reference that matters is captured before any of this and survives the defer.

Patch preserved at `/sim/tmp/bench/apply_defer_always.py`; the tree is reverted.

## Also ruled out: the pool, and the concurrent dispatch groups

Same probe, same protocol, one arm at a time, 15 renders per trial:

| arm | lost | survived |
|---|---|---|
| baseline | 5 | 1 |
| defer every free | 4 | 2 |
| `pool_disabled = true` | 4 | 0 |
| no concurrent groups | 2 | 0 |

**The pool.** `destroy_buffer!` on a pooled chunk calls `return_to_pool!` rather
than destroying, so a still-referenced block handed back to the next
`pool_alloc` would look exactly like this. One VkBuffer per allocation removes
that path entirely, and the fault is unchanged.

**The groups.** `concurrent_dispatch_group` elides the barriers between the
dispatches inside it and `concurrent_indirect_group` fuses their prepares, so
the GPU may run them overlapped — a plausible source of nondeterminism in a
workload whose sampling is deterministic, and where two of the three races found
in this codebase already lived. Making both wrappers pass-throughs, so every
dispatch keeps its own barrier and every indirect records its own prepare
inline, does not help either. (The toggle was verified live in the running
process, not assumed.)

## Where that leaves it, and the one thing the eliminations point at

Six hypotheses are now dead: missing barrier, unbounded loop, stale BDAs,
destroy timing, pool reuse, dispatch overlap. What remains has to explain how a
workload that submits IDENTICAL work every run — the framebuffer sums are
bit-identical, so this is not in question — fails one run in five.

If the work is fixed and the outcome is not, the varying input is not the work:
it is the ADDRESS LAYOUT. Allocation order and virtual addresses differ run to
run, so an out-of-bounds access that usually lands in another live buffer, and
is therefore invisible, occasionally lands in an unmapped page and takes the
device.

That reading also explains the pool result, which looked like noise: disabling
the pool makes it 4/4 rather than 5/6, and one VkBuffer per allocation means
MORE unmapped gaps between allocations for a stray access to find. Suballocation
inside a big block hides exactly this.

So the next instrument is GPU-assisted validation with bounds checking
(`DebugConfig(gpu_av = true, gpu_av_shaders = [...])`), narrowed to the medium
kernels — with the caveat already in `vk_free!`'s message that GPU-AV can itself
crash on a workload that kills the device, so it wants narrowing first.
