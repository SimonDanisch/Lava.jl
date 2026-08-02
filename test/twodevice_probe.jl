"""
Two devices in one process: the acceptance probe for `GUARDRAILS.md` §8.

    julia --project=<env> dev/Lava/test/twodevice_probe.jl

**Deliberately NOT in `runtests.jl`.** It currently segfaults, and a segfault
takes the whole suite with it — including everything that would have run after.
It goes back in the suite when it passes.

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

## What it has found so far

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

2. **The next one is still open.** With (1) fixed the GPU dispatch is correct and
   the crash moves to the lavapipe dispatch. Candidates, all module-scope and all
   device-owned, in descending order of suspicion:

       PREPARE_INDIRECT_PIPELINE_REF   a LavaComputePipeline, bound at dispatch
       PREPARE_INDIRECT_OFFSETS_REF    its layout, same lifetime
       TIMESTAMP_POOL                  a Vulkan.QueryPool
       BLIT_PIPELINE                   a graphics pipeline
       GEMM_SPLIT_SCRATCH              device memory
       DEVICE_SUBGROUP_SIZE            a device property cached globally —
                                       wrong answer rather than a crash, and so
                                       the one most likely to survive unnoticed
       SUBGROUP_SIZE_CONTROL           same
       WORKGROUP_LIMIT                 same

   That list is the phase-2 worklist, and it was produced by running rather than
   by reading — which matters, because reading produced a list of four.
"""

using Lava, KernelAbstractions
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
        a = KA.allocate(b, Float32, 64)
        fill!(a, 1.0f0)
        twodev!(b, 64)(a, 3.0f0; ndrange = 64)
        KA.synchronize(b)
        got = Array(a)
        println("  $name: ", all(got .== 5.0f0) ? "correct" : "WRONG ($(got[1]))")
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

abspath(PROGRAM_FILE) == @__FILE__ && probe()
