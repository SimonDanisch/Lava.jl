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

## Poisoning the stale entries prevents the hang

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
it is not observing the stale references, it is neutralising them. That makes
this causal rather than correlational: remove the stale addresses and the hang
does not happen.

**The confound, stated plainly.** The scan also makes each render 90x slower
(42 s against 0.46 s), because it walks every slab on every free. A slower
render shifts whatever window the collector and the recording overlap in, and
that alone could hide the fault. So this is strong evidence, not proof. What
would settle it: log the hits WITHOUT zeroing them and see whether the device
still dies at the same rate. If it does, the poisoning is what mattered; if it
does not, the slowdown is.

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
