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
| held alive | never | 3 / 3 survived |

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

## Next

`ctx.diag.freed_bda_scan` is the instrument for exactly the question the comment
poses: before destroying a buffer it scans live arg slabs for its address, which
is a reference nothing pinned. `/sim/tmp/bench/bda_scan.jl` turns it on and runs
the reproducer; a hit names the buffer and the slab.

Worth pairing with `ctx.diag.free_debug`, whose log records for each free
whether a batch was recording at the time — the two together should say whether
the buffer was freed during a recording and then named by it.
