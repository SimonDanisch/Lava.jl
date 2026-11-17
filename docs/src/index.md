# Lava.jl

Lava compiles Julia functions to SPIR-V for Vulkan: compute kernels and every graphics
stage, through GPUCompiler and its own LLVM IR to SPIR-V emitter.

Lava is the compiler only. Devices, memory, arrays, launches, graphics pipelines and
acceleration structures belong to [Mantle.jl](https://github.com/SimonDanisch/Mantle.jl),
whose Vulkan backend calls Lava to compile. Most code never calls Lava directly: it
writes kernels against KernelInterface or KernelAbstractions and launches them on a
Mantle device.

## Installation

```julia
import Pkg
Pkg.add("Lava")       # the compiler
Pkg.add("Mantle")     # to run what it compiles
```

Lava requires Julia 1.12 or newer. Compiling needs no GPU and no driver, only a Vulkan loader,
which `Vulkan_Loader_jll` provides where the system has none. Running compiled code needs a Vulkan 1.2+ driver
with `bufferDeviceAddress` and `variablePointers`: NVIDIA, AMD, Intel, or lavapipe
(`Lavapipe_jll`) as a CPU device.

## A kernel, compiled and run

```julia
using Mantle
import KernelInterface as KI

function vadd!(c, a, b)
    i = KI.get_global_id().x
    i <= length(c) && (@inbounds c[i] = a[i] + b[i])
    return nothing
end

dev = Mantle.Device()
a = Mantle.devicearray(dev, rand(Float32, 1024))
b = Mantle.devicearray(dev, rand(Float32, 1024))
c = Mantle.devicearray(dev, zeros(Float32, 1024))
KI.@launch dev ndrange=1024 vadd!(c, a, b)
Array(c) ≈ Array(a) .+ Array(b)
```

The same function compiled without a device:

```julia
using Lava

r = lava_compile(vadd!, Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1},
                              LavaDeviceArray{Float32,1}})
print(r.spirv_disasm)
```

## Contents

* [Compiling](compiling.md): `lava_compile`, stages, kernel arguments, compile errors
* [Device code](device.md): intrinsics, math semantics, atomics, workgroup memory,
  cooperative matrices, ray queries
* [The device vocabulary](vocabulary.md): the device API every backend implements
* [Graphics stages](graphics.md): vertex to mesh stages and their configuration
* [Architecture](architecture.md): passes, emitter, validation and caching
* [Debugging](debugging.md): dumps, source maps, printing from a kernel
* [API reference](api.md) and [vocabulary reference](vocabulary_api.md)
