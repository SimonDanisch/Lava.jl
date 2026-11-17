# Architecture

## Pipeline

```
Julia function
  │  GPUCompiler (LavaCompilerTarget, Lava's method table of device overrides)
LLVM IR (LLVM.jl 10)
  │  Lava's LLVM passes
  │  Lava's SPIR-V emitter
SPIR-V
  │  spirv-opt -O, spirv-val
SPIR-V module
```

GPUCompiler produces LLVM IR for the `spirv64` triple; it is never asked for SPIR-V.
Device code resolves through Lava's method table, where `@lava_device_override`
replaces Base, KernelInterface, KernelAbstractions, Atomix and vocabulary functions
with their Vulkan lowering.

## LLVM passes

The passes bring LLVM's IR into a shape SPIR-V can express. In order:

* **Inlining and cleanup.** Everything is inlined into the entry point except
  functions too large for some drivers' per-function limits, which are outlined.
  `unreachable`, `freeze`, `noreturn` and `llvm.assume` are removed.
* **Integer lowering.** 64-bit divisions by run-time values become 32-bit or
  multiply-high sequences; integers of odd widths, which SPIR-V lacks, are widened;
  `memset` and `memcpy` of run-time length become loops.
* **Read-only loads.** Loads through arguments the runtime marked read-only are
  tagged, so the emitter decorates them non-writable.
* **Typed addressing.** Byte-offset GEPs become typed GEPs, chained GEPs are
  flattened, and `alloca`s that Julia types as byte arrays get the element type
  every access uses. SPIR-V's `OpAccessChain` needs the types LLVM erased.
* **Structured control flow.** `LowerSwitch`, `FixIrreducible` and `StructurizeCFG`,
  with Lava's preparation for merge blocks shared across nesting levels, so every
  loop and selection has the single merge block SPIR-V requires.
* **Vulkan preparation.** Workgroup arrays are flattened, type-punned allocas and
  phis are split by type, and composite loads and stores through physical storage
  buffer pointers are decomposed into scalar accesses.

`LAVA_SKIP_PASSES` and `LAVA_VERIFY_PASSES` control the pipeline for debugging
(see [Debugging](debugging.md)).

## The emitter

`src/compiler/spirv/` translates the post-pass module into SPIR-V directly, through
LLVM.jl's API. LLVM's own SPIR-V backend is not used: it fails on geometry shaders
and lacks `SPV_KHR_physical_storage_buffer` and `SPV_KHR_variable_pointers`, which
every module Lava emits is built on.

* `module.jl`: the module builder, ids, capabilities, extensions, serialization.
* `types.jl`: LLVM to SPIR-V types, including pointee types recovered from use,
  since LLVM's pointers are opaque.
* `emit.jl`: instructions, structured control flow, decorations, atomics, the Vulkan
  memory model.
* `graphics.jl`: stage entry points, interface variables, execution modes.
* `coopmat.jl`, `rayquery.jl`: cooperative matrices and ray queries.

Every module uses the `PhysicalStorageBuffer64` addressing model and the Vulkan
memory model. Kernel arguments arrive through one buffer device address (see
[Compiling](compiling.md#Arguments)).

One module has to run correctly on every driver. When a driver miscompiles a valid
pattern, the emitter produces a different valid pattern that every driver handles,
instead of branching on the vendor.

## Validation and optimization

Each module goes through `spirv-opt -O` (SPIRV-Tools) and is validated with
`spirv-val`. A failure is a `LavaCompilationError` with the SPIR-V id mapped back to
the Julia source line.

## Caching

Compiled kernels and stages are kept with the `CodeInstance` of the kernel they were
compiled from, as GPUCompiler does for CUDA.jl. Redefining the kernel or anything
inlined into it invalidates the entry; unrelated definitions do not. Kernels compiled
while a package precompiles go into its package image, so a package's
`@compile_workload` leaves them compiled for every later session.

Kernels compiled at run time are also kept on disk in CompilerCaching's object cache,
keyed by the canonical LLVM module, the compile settings, and the versions of Lava,
Julia, LLVM and SPIRV-Tools.

The compilation pipeline itself runs in the world age captured when Lava loads, so
packages loaded afterwards cannot invalidate its precompiled code.
