# Debugging

Lava exposes a small set of introspection and recovery utilities for development and post-mortem.

## Inspect the device

```julia
ctx = Lava.vk_context()
@show ctx.device_name
@show ctx.queue_family
@show ctx.ray_query_available
@show ctx.rt_pipeline_properties
```

## Memory usage

```julia
gpu_memory_usage()
# → (live_bytes = 1.2e9, live_buffers = 14, default_bq = (...) , ...)
```

Returns a snapshot of currently-live device buffers, mapped memory, slab pools, and per-batch refs. Useful to spot leaks across iterations.

## Full runtime state dump

```julia
dump_state()
```

Prints batch queue contents, in-flight command buffers, deferred frees, slab inventories, and the timeline semaphore counters. The same information backs the `mwe_*.jl` regression tests in `test/`.

## Dispatch logging

```julia
set_dispatch_logging!(true)
# ... run your code ...
log = get_dispatch_log()   # Vector of (kernel_name, dispatch_time_µs, args_summary)
```

This is how the per-dispatch latency numbers in [Performance](performance.md) were measured.

## Recovering from device-lost

A driver bug, OOM, or your own kernel writing past a buffer can leave the Vulkan device in a lost state. Recover without restarting Julia:

```julia
Lava.vk_reset_device!()
```

This tears down the current `VkDevice` (and everything dependent on it) and creates a fresh one. Pre-existing `LavaArray`s become invalid — recreate them after the reset.

## Where to look for breadcrumbs

* **Validation layers** — Lava enables `VK_LAYER_KHRONOS_validation` by default in debug builds; messages are printed and also captured in `LavaError`. Disable with `LAVA_NO_VALIDATION=1` for production.
* **Last compiled IR and SPIR-V** — every kernel compile writes the post-pass LLVM IR to `/tmp/lava_last.ll` and the resulting SPIR-V to `/tmp/lava_last.spv`. The disassembly is at `/tmp/lava_last.dis`. After a `spirv-val` failure these are the artefacts to inspect.
* **Per-kernel dumps** — `dev/Lava/tmp_kernels/kernel_<n>_<entryname>.{ll,spv}` is written for every successful compile, so you can diff between two kernels or two Julia runs.
* **GitHub Actions logs** — the CI runs the full test suite under lavapipe and is the easiest place to see what a clean run looks like.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `OpAccessChain into a structure must be an OpConstant` | A struct member index was emitted via a runtime path | Open an issue with `/tmp/lava_last.ll` and the kernel |
| `vkCmdTraceRaysIndirectKHR` aborts with `VK_ERROR_DEVICE_LOST` after many compute dispatches on RADV | Known driver bug, see [Known Issues](known_issues.md) | Workaround listed there |
| `glfwInit failed` on package load | Headless box without DISPLAY | Run under `xvfb-run` or set `LAVA_HEADLESS=1` (compute only) |
| Test suite fails with `display` errors | Same as above | `xvfb-run -s '-screen 0 1024x768x24' julia --project -e 'using Pkg; Pkg.test()'` |
| `MethodError` from broadcasting between `LavaArray` and `Array` | Implicit host-device mixing | Use `Adapt.adapt(LavaBackend(), array)` explicitly |
