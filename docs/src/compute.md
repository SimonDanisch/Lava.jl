# Compute

Lava implements the standard Julia GPU contract: `LavaBackend <: KernelAbstractions.GPU` and `LavaArray{T,N} <: GPUArrays.AbstractGPUArray`. Code written against those interfaces — your own kernels, GPUArrays.jl operations, AcceleratedKernels.jl — runs on Vulkan without changes.

## LavaBackend and LavaArray

```julia
using Lava, KernelAbstractions

backend = LavaBackend()

a = LavaArray(rand(Float32, 1024))
b = LavaArray(rand(Float32, 1024))
c = LavaArray{Float32}(undef, 1024)

@kernel function vadd!(C, A, B)
    i = @index(Global)
    C[i] = A[i] + B[i]
end

vadd!(backend)(c, a, b; ndrange=length(c))
@assert Array(c) ≈ Array(a) .+ Array(b)
```

Most KA features work out of the box:

| KA feature | Status |
|---|---|
| `@index(Global)` / `@index(Local)` / `@index(Group)` | ✓ |
| `@groupsize` / `@uniform` / `@private` | ✓ |
| `@localmem` with primitive and struct element types | ✓ |
| `@synchronize` (workgroup barrier) | ✓ |
| `Const` reads | ✓ |
| Dynamic ndrange and workgroup size | ✓ |
| Heterogeneous launch (mixed-type args) | ✓ |

## Broadcasting and reductions

`LavaArray` participates in regular Julia broadcasting and `mapreduce`. Both go through the same SPIR-V compilation path, so a one-liner like

```julia
v = LavaArray(rand(Float32, 10^7))
sum(sin, v) / length(v)
```

compiles two kernels (the elementwise `sin` and a tree reduction) and dispatches them on the GPU.

## Atomics

Lava supports atomic operations through Atomix.jl on the standard 32-bit integer and float types:

```julia
using Atomix

@kernel function atomic_hist!(counts, samples)
    i = @index(Global)
    @inbounds bucket = clamp(samples[i] >> 4, 0, length(counts) - 1) + 1
    Atomix.@atomic counts[bucket] += 1
end
```

Supported ops: `+ - & | ⊻ xchg min max` on `Int32`/`UInt32`, plus add/min/max on `Float32` (implemented via a CAS loop where the hardware doesn't have native float atomics). `cmpxchg!` is also supported.

## Buffer Device Address (BDA) arguments

All kernel arguments are packed into a single device-memory buffer and passed via a single 8-byte push constant pointing to the BDA. Consequences:

* No 256-byte push-constant limit — argument count and total size are practically unbounded.
* No descriptor sets per buffer — `LavaArray` arguments become BDA pointers automatically.
* `isbits` structs are passed by value, copied into the arg buffer.
* `LavaArray`/`LavaDeviceArray` arguments are passed by raw GPU address.

For the curious: `src/compiler/entry_wrapper.jl` wraps each kernel entry so its first argument is the BDA, and the wrapper unpacks the original arguments before calling the user function.

## Memory model

Three memory regions are visible to compute kernels:

* **`LavaArray` (device-local)** — the default. Lives on the GPU; copied to/from host via `Array(...)`/`copyto!`.
* **Unified / BAR memory** — auto-selected for small buffers (≤ 64 bytes by default). Host has a directly mapped pointer, so `Array(buf)` is a single fence instead of a fence + staging copy.
* **`@localmem`** — workgroup-scoped scratchpad. Supports primitive types, primitive arrays, and `isbits` struct arrays.

## Compilation cache

Compiled SPIR-V is cached at two layers:

* **In-memory** by Julia `MethodInstance` (zero-cost after first compile)
* **On-disk** under `~/.julia/compiled/Lava/spirv/...`, keyed by IR hash + Vulkan device fingerprint

To force a recompile after editing a kernel:

```julia
Lava.clear_kernel_cache!()
```

To wipe the on-disk cache too (rarely needed):

```julia
Lava.clear_spirv_disk_cache!()
```

## Performance notes

* Per-dispatch overhead is ~1.8 µs (vs ~8.8 µs for AMDGPU). At small problem sizes this is what dominates.
* Multiple dispatches are batched into a single command buffer. They flush automatically before a host read (`Array(buf)`); call `vk_flush!(backend.bq)` to flush explicitly.
* For tight CPU↔GPU loops, prefer Unified/BAR memory by allocating small buffers explicitly through KA's `unified=true` allocator.

See [Performance](performance.md) for benchmarks and tuning tips.
