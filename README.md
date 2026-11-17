# Lava.jl

[![CI](https://github.com/SimonDanisch/Lava.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/SimonDanisch/Lava.jl/actions/workflows/ci.yml)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://SimonDanisch.github.io/Lava.jl/dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A Julia to SPIR-V compiler for Vulkan. Lava compiles plain Julia functions into
SPIR-V modules for compute kernels and for every graphics stage, with its own
LLVM IR to SPIR-V emitter.

Lava is the compiler only. Devices, memory, arrays, launches, graphics pipelines and
acceleration structures are [Mantle.jl](https://github.com/SimonDanisch/Mantle.jl)'s,
whose Vulkan backend calls Lava to compile.

## Installation

```julia
import Pkg
Pkg.add("Lava")       # the compiler
Pkg.add("Mantle")     # to run what it compiles
```

Requires Julia 1.12 or newer. Compiling needs no GPU and no driver. Running needs a Vulkan 1.2+
driver with `bufferDeviceAddress` and `variablePointers`: NVIDIA, AMD (RADV, AMDVLK,
Windows), Intel, or lavapipe (`Lavapipe_jll`) as a CPU device.

## What it compiles

* **Compute kernels**: plain functions written against
  [KernelInterface](https://github.com/JuliaGPU/KernelAbstractions.jl/tree/main/lib/KernelInterface)
  (`KI.get_global_id`, `KI.localmemory`, `KI.barrier`, subgroup shuffles and
  reductions), and KernelAbstractions `@kernel` functions.
* **Graphics stages**: vertex, fragment, geometry, tessellation control and evaluation,
  task and mesh shaders, from Julia functions.
* **Ray queries** (`VK_KHR_ray_query`) against a top-level acceleration structure.
* **Device intrinsics**: atomics through Atomix with explicit memory orderings
  (sequential consistency lowers to the strongest ordering Vulkan has), subgroup
  operations, `@lava_printf` through `NonSemantic.DebugPrintf`, integer dot products,
  and Base math through `GLSL.std.450`. `min`, `max`, `^`, `round`, `clamp` and
  `hypot` keep Julia's NaN, signed-zero and domain rules; the relaxed GPU forms are
  opt-in through `Base.FastMath` (`min_fast`, `max_fast`, `pow_fast`).
* **Cooperative matrices**: `VK_KHR_cooperative_matrix`, and the
  `VK_NV_cooperative_matrix2` operations (per-element callbacks, reductions,
  conversions, tensor addressing, workgroup scope).

## The device vocabulary

`src/vocabulary/` declares the device-side functions and types that kernels and
shaders are written against and that KernelInterface does not have yet: cooperative
matrices (`CoopMatrix`, `AcceleratedMatrix`, `MatrixShape`, `coopmat_*`), tensor
layouts, the graphics builtins (`vertex_index`, `frag_coord`, `discard`, …), the mesh
pipeline (`MeshConfig`, `MeshEmitter`, …) and `DeviceCaps`. The functions have no
device methods here. Each backend supplies them as overlays in its own method table:
Lava for Vulkan, Mantle's Metal backend for Metal. It loads without a Vulkan loader,
so a Metal-only machine can use it too. The names are public, not exported. Mantle
re-exports the ones a user writes.

## Running a kernel

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

## Compiling without a device

```julia
using Lava

r = lava_compile(vadd!, Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1},
                              LavaDeviceArray{Float32,1}}; workgroup_size = (64, 1, 1))
r.spirv_bytes      # the module
print(r.spirv_disasm)
```

`lava_compile(f, tt; stage)` takes `stage = :compute` (default), `:vertex`,
`:fragment`, `:geometry`, `:tess_control`, `:tess_eval`, `:task` or `:mesh`. The
`CompilationResult` keeps the LLVM IR before and after Lava's passes, the SPIR-V and
its disassembly, the push-constant layout, and a map from SPIR-V ids to Julia source
lines.

## How it compiles

```
Julia function
  │  GPUCompiler (LavaCompilerTarget)
LLVM IR (LLVM.jl 10)
  │  Lava's passes: GEP lifting, alloca retyping, CFG structurization,
  │  Vulkan preparation, intrinsic lowering
  │  Lava's SPIR-V emitter (src/compiler/spirv/)
SPIR-V
  │  spirv-opt -O, spirv-val (SPIRV_Tools_jll)
SPIR-V module
```

The emitter is written in Julia against LLVM.jl; LLVM's SPIR-V backend is not
involved. That backend crashes on geometry shaders and lacks
`SPV_KHR_physical_storage_buffer` and `SPV_KHR_variable_pointers`, which the modules
Lava emits are built on: every kernel argument is read through one buffer device
address passed as a push constant.

Compiled stages are cached per method instance, and on disk across sessions in
CompilerCaching's object cache. `LAVA_SPIRV_DUMP_DIR` writes every module and its
post-pass IR to a directory.

## Source layout

```
src/
├── Lava.jl            module, exports
├── vocabulary/        the device vocabulary, shared by every backend
├── compiler/          GPUCompiler target, caches, compilation pipeline
│   ├── passes/        LLVM IR passes ahead of the emitter
│   └── spirv/         the SPIR-V emitter: types, instructions, graphics,
│                      cooperative matrices, ray queries
├── device/            device intrinsics and the Vulkan overlays of the vocabulary,
│                      KernelInterface, KernelAbstractions, Base math and Atomix
├── graphics/types.jl  tessellation stage configuration
└── runtime/           compile-time error types and intrinsic helpers
```

## Tests

```julia
import Pkg; Pkg.test("Lava")
```

The suite needs no GPU. It checks emitted SPIR-V patterns, compiles every stage and
validates the result with `spirv-val`, and checks the lowering of Base math,
atomics, cooperative matrices and graphics builtins. Execution tests run in Mantle's
suite, on every backend a machine has.

## Documentation

* [Dev docs](https://SimonDanisch.github.io/Lava.jl/dev/)
* [CONTRIBUTING.md](CONTRIBUTING.md): how emitter and driver bugs are fixed and tested

## License

MIT. See [LICENSE](LICENSE).
