# Debugging

## Inspecting a compile

[`lava_compile`](@ref) returns every intermediate form:

```julia
r = lava_compile(f, tt)
print(r.pre_pass_ir)     # GPUCompiler's LLVM IR
print(r.post_pass_ir)    # after Lava's passes
print(r.spirv_disasm)    # the module
r.source_map             # SPIR-V id => (file, line)
```

## Compile errors

GPU-incompatible code throws a `LavaCompilationError` that names the problem
(dynamic dispatch, global variable access, heap allocation, an unsupported
intrinsic) and each call chain from the kernel to it, with source locations. The
deepest user-code frame of a chain is usually where the type instability or the
allocation starts; the runtime call itself sits a few inlined frames further in.

## Dumping modules

| variable | effect |
|---|---|
| `LAVA_SPIRV_DUMP_DIR=<dir>` | write every compiled module (`.spv`) and its post-pass IR (`.ll`), named by kernel and content hash |
| `LAVA_DUMP_KERNELS=1` | write the `.ll`/`.spv` pair of every compile to `LAVA_DUMP_KERNELS_DIR` |
| `LAVA_DUMP_PASSES=<dir>` | write the module after every LLVM pass, to find the pass that broke it |
| `LAVA_VERIFY_PASSES=1` | run LLVM's verifier after every pass |
| `LAVA_DEBUG_PASSES=1` | write intermediate IR to `LAVA_DEBUG_DIR` (default `tempdir()`) |
| `LAVA_SKIP_PASSES=<a,b,…>` | skip the named steps of the control-flow structurization |
| `LAVA_PHASE_TIMING=1` | print the time each compilation phase takes |

A dumped module can be checked and disassembled with SPIRV-Tools:

```julia
using Lava
const ST = Lava.SPIRV_Tools_jll
run(`$(ST.spirv_val()) --target-env vulkan1.3 kernel.spv`)
run(`$(ST.spirv_dis()) kernel.spv`)
```

## Printing from a kernel

`@lava_printf` and KernelAbstractions' `@print` emit `NonSemantic.DebugPrintf`
(see [Device code](device.md#Printing)). The Vulkan validation layer executes it, so
output needs a device created with printf enabled:

```julia
using Mantle, Lava
import KernelInterface as KI

function show_index!(out)
    i = KI.get_global_id().x
    @lava_printf "i = %u\n" UInt32(i)
    @inbounds out[i] = i
    return nothing
end

dev = Mantle.Device(Mantle.VulkanAPI(); debug = Mantle.DebugConfig(printf = true))
out = Mantle.devicearray(dev, zeros(Int, 4))
KI.@launch dev ndrange=4 show_index!(out)
```

Each printed line arrives as an `@info` log message.

## Driver miscompiles

When a module passes `spirv-val` and still computes wrong values on one driver, the
pattern to look for is in the module, not in the driver: dump it with
`LAVA_SPIRV_DUMP_DIR`, reduce the kernel until the difference is a few
instructions, and change the emitter to produce a pattern every driver handles.
`test/glsl/` holds hand-written GLSL compiled with glslang that serves as a
reference for the same operations.
