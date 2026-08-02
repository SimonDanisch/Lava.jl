"""
Two devices in one process: the acceptance probe for `GUARDRAILS.md` §8.

    julia --project=<env> dev/Lava/test/twodevice_probe.jl

**It passes**, and is in `runtests.jl`. It was held out while it segfaulted,
because a segfault takes the whole suite with it.

## Why this is possible now, and was not before

`init_vulkan!(; select)` lets a caller choose the physical device and returns a
context **without** installing it as the global. The Vulkan loader enumerates the
real GPU and lavapipe from one instance, so every machine here has a two-device
pair with no second card:

    gpu = vk_context()
    cpu = init_vulkan!(select = devs -> only(filter(islavapipe, devs)))

## What it is for

Four caches hold device-owned handles at module scope. Keying them by
`VkContext.id` (never reused) is necessary and — as this probe demonstrated on
its first run — **not sufficient**. Module-scope device state is a larger
category than the cache list, and the only way to enumerate it honestly is to
run two devices and see what breaks.

## What it found, in the order it found them

Each was fixed and the probe re-run, which is why the list is ordered: nothing
below was visible until everything above it was fixed.

1. **`CMD_PIPELINE_BARRIER_FPTR`** — FIXED, now `VkContext.cmd_pipeline_barrier_fptr`.
   A device function pointer is per device: `vkGetDeviceProcAddr` returns one
   valid only for the device asked. As a global, creating the second context
   overwrote it, and the FIRST device's command buffers were then recorded
   through the SECOND device's driver. The crash was inside `libvulkan_lvp.so`
   while dispatching on the NVIDIA context, which is as confusing as it sounds.

   This is the most dangerous class of the lot and `GUARDRAILS.md` §8 does not
   name it: §8 lists four caches holding *handles*, and says nothing about the
   function table. A stale handle is undefined behaviour; a foreign function
   pointer is an immediate jump into another driver.

2. **`PREPARE_INDIRECT_*`** — FIXED. Four `Ref`s holding one compiled pipeline
   and its layout, now one `PrepareIndirect` per device.

3. **`DEVICE_SUBGROUP_SIZE` and `SUBGROUP_SIZE_CONTROL`** — FIXED, now per-device
   dicts. These differ from the handle caches in a way worth keeping in mind: a
   stale pipeline is undefined behaviour and usually crashes, while a stale
   device *property* just returns the wrong number. 32 here, 64 on RDNA 3.5 —
   every tiling decision keyed on it would be made for the other device, and
   nothing would crash.

4. **THE ALLOCATOR** — the one that actually mattered. FIXED.

   `POOL_BLOCKS` and `POOL_FREE_LISTS` are module-level, and `PoolBlock` carries
   no device:

       mutable struct PoolBlock
           buffer, memory, base_address, capacity, bump, live_count
       end

   Measured: allocate on the GPU (one 64 MiB block is created), then allocate on
   lavapipe — `length(POOL_BLOCKS)` is **still 1**. The lavapipe array was carved
   out of the NVIDIA device's block. `buf.ctx` correctly says `cpu`; the memory
   underneath belongs to the other device.

   That is both observed symptoms at once. `fill!` on the second context then
   reads back **0.0** — it wrote into memory that device does not own — and in a
   different call order it segfaults instead.

   **A bigger class than anything in `GUARDRAILS.md` §8, which lists four caches
   holding pipeline handles.** The allocator hands out *memory*, so the failure
   is silent data corruption rather than a bad handle, and no amount of cache
   keying reaches it.

   Now `DevicePool` per device, with each `PoolBlock` carrying a back-reference
   to its pool so `return_to_pool!` — which runs from a **finalizer**, where a
   lookup must not allocate and must not be able to miss — is a field hop.

5. **`LavaBackend()` built inside the library, six places.** An unpinned backend
   resolves its queue through `vk_context()`, so `fill!`, `mul!`,
   `coopmat_gemm!`, broadcast `_copyto!` and the identity-matrix constructor all
   dispatched on whichever context was global rather than on the array's own.
   The array read back as zeros and Lava's `sync_access!` guard caught it much
   later as *"buffer was last written on a BatchQueue from a DIFFERENT
   VkContext"* — a good error a long way from its cause. All six now derive the
   backend from the data with `KA.get_backend`.

   That guard existing and firing is worth noting: the library already knew this
   was possible and said so precisely.

6. **`GEMM_SPLIT_SCRATCH`** — FIXED, and now exercised above. A single `Ref`
   holding device memory: the memory pool's defect in miniature, and it would
   only have fired on a split-K GEMM, which is why the first version of this
   probe missed it. The probe now runs a GEMM and a reduction on each device
   rather than a bare dispatch, because a global that only some kernels touch is
   invisible to a probe that only runs one kernel.

7. **`_REDUCE_SCRATCH`** — FIXED. It was already an `IdDict` keyed by context,
   and still allocated its buffer on `vk_context()`. Keyed right, allocated
   wrong — which reads as correct on any machine with one device, and is the
   subtlest shape in this whole list.

Still unaudited, because nothing on this path reaches them: `TIMESTAMP_POOL` (a
`Vulkan.QueryPool`), `BLIT_PIPELINE` and `GFX_SHADER_CACHE` (graphics), and
`WORKGROUP_LIMIT` (a policy limit, not a queried one). Extend the probe before
trusting a second device for graphics or dispatch profiling.

The list above was produced by RUNNING two devices. Reading produced a list of
four caches, and **none of the four was what actually broke it.**
"""

using Lava, KernelAbstractions, LinearAlgebra
const KA = KernelAbstractions

@kernel function twodev!(d, s)
    i = @index(Global)
    @inbounds d[i] = d[i] * 2.0f0 + s
end

function probe()
    gpu = Lava.vk_context()
    cpu = Lava.init_vulkan!(select = devs -> only(filter(Lava.islavapipe, devs)))

    println("gpu id=$(gpu.id)  $(gpu.device_name)")
    println("cpu id=$(cpu.id)  $(cpu.device_name)")
    gpu.id == cpu.id && error("two contexts share an id — every per-device key is void")
    gpu.cmd_pipeline_barrier_fptr == cpu.cmd_pipeline_barrier_fptr &&
        error("both devices report the same vkCmdPipelineBarrier — the global is back")

    before = length(Lava.PIPELINE_CACHE)
    for (name, ctx) in (("gpu", gpu), ("cpu", cpu))
        b = LavaBackend(ctx)

        # ── a dispatch, and the fill! that feeds it
        a = KA.allocate(b, Float32, 64)
        fill!(a, 1.0f0)
        twodev!(b, 64)(a, 3.0f0; ndrange = 64)
        KA.synchronize(b)
        got = Array(a)
        ok = all(got .== 5.0f0)

        # ── a reduction: `_REDUCE_SCRATCH` is keyed by context but used to
        #    ALLOCATE on the global one, so the second device's entry held the
        #    first device's buffer. Keyed right, allocated wrong.
        r = Lava.vk_reduce_sum(a)
        okr = r ≈ 64 * 5.0f0

        # ── a GEMM big enough to split K: `GEMM_SPLIT_SCRATCH` was one `Ref`
        #    holding device memory, i.e. the memory pool's defect in miniature.
        m = 64
        A = KA.allocate(b, Float16, m, m); fill!(A, Float16(1))
        B = KA.allocate(b, Float16, m, m); fill!(B, Float16(2))
        C = KA.allocate(b, Float32, m, m); fill!(C, 0.0f0)
        LinearAlgebra.mul!(C, A, B)
        KA.synchronize(b)
        okg = all(Array(C) .== Float32(2 * m))

        println("  $name: dispatch ", ok  ? "ok" : "WRONG ($(got[1]))",
                "   reduce ", okr ? "ok" : "WRONG ($r)",
                "   gemm ",   okg ? "ok" : "WRONG ($(Array(C)[1]))")
        (ok && okr && okg) || error("$name produced a wrong result")
        A = B = C = a = nothing
    end

    # The load-bearing assertion. One kernel on two devices must compile TWICE:
    # a single new entry means the second device was handed the first's pipeline,
    # which is the §8 defect and can still produce a right answer by luck.
    grew = length(Lava.PIPELINE_CACHE) - before
    println("\nPIPELINE_CACHE grew by $grew (1 means the devices SHARED a pipeline)")
    println("LINKED_KERNEL_CACHE device keys: ", sort(collect(keys(Lava.LINKED_KERNEL_CACHE))))
    println("LAUNCH_PLAN_CACHE   device keys: ", sort(collect(keys(Lava.LAUNCH_PLAN_CACHE))))
    grew >= 2 || error("the two devices shared a pipeline")
    println("\nPASS")
end

if abspath(PROGRAM_FILE) == @__FILE__
    probe()
end
