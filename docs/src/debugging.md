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

## Printing from kernels

Lava can print directly from a running kernel, which is often the fastest way to
diagnose a wrong value or a driver miscompile (you see the offending index/offset
instead of inferring it from the output). It is implemented with the Vulkan
validation layer's `NonSemantic.DebugPrintf`.

Two entry points are available:

* **`KernelAbstractions.@print`** — the portable API (same one CUDA.jl and
  AMDGPU.jl implement). Interleave string literals with values, like `print`;
  format specifiers are chosen automatically from the argument types.
* **`@lava_printf`** — Lava-specific, with an explicit C-style format string when
  you want full control over specifiers, width and padding.

```julia
using Lava, KernelAbstractions

@kernel function k!(out)
    i = @index(Global)
    @print("tid=", i, "  val=", out[i], "\n")          # portable
    @lava_printf "tid=%u  val=%f\n" UInt32(i) out[i]    # explicit format
    @inbounds out[i] = Float32(i)
end

Lava.enable_debug_printf!()        # resets the device with the layer feature on
backend = Lava.LavaBackend()
out = Lava.LavaArray(zeros(Float32, 8))
k!(backend)(out; ndrange = 8)
Lava.vk_flush!(Lava.vk_context().default_bq)
KernelAbstractions.synchronize(backend)

Lava.get_printf_output()           # Vector{String} of captured lines
Lava.disable_debug_printf!()       # back to the fast path
```

Captured lines are also `@info`-logged as they arrive. Read them with
`Lava.get_printf_output()` and reset with `Lava.clear_printf_output!()`.

Format-specifier **width** must match the argument width or the layer warns:

| Argument type | specifier |
|---|---|
| `Int32` / smaller, `Bool` | `%d` |
| `UInt32` / smaller | `%u` (also `%x`) |
| `Int64`, `Int` | `%ld` |
| `UInt64` | `%lu` |
| `Float16`, `Float32` | `%f` (also `%e`, `%g`) |
| `Float64` | `%lf` (the `l` is **required** for 64-bit) |

`@print` picks these for you; with `@lava_printf` you write them.

!!! note
    Output only appears while debug printf is enabled. Without it the
    `DebugPrintf` instruction is inert — the kernel still runs and computes
    correctly and simply produces no output — so prints are safe to leave in
    code. Debug printf and GPU-Assisted Validation both instrument shaders and
    cannot be active at the same time, so `enable_debug_printf!()` turns GPU-AV
    off. Like all validation features it slows execution substantially; use it
    for triage. Do not run a debug-printf workload concurrently with another
    heavy GPU process — that has been observed to crash the NVIDIA driver.

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
