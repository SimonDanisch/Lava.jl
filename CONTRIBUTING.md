# Contributing to Lava

Lava emits one SPIR-V module that has to run correctly on every Vulkan driver:
NVIDIA, RADV, AMDVLK, Intel, lavapipe. The rules below keep it that way as driver
workarounds accumulate.

## No vendor branching

* No vendor-conditional code in `src/`: no `if vendor == "NVIDIA"`, no per-driver
  `@static`. When a driver miscompiles a pattern, change how the emitter produces it,
  so that every driver handles the new pattern. Driver names belong in comments that
  explain why a pattern looks the way it does, never in code paths.
* No vendor-conditional tests and no vendor names in test file names. A test pins a
  source pattern, and its assertion is the same on every platform.
* A test that needs a device feature is gated on the capability, never on the
  vendor.

## Fixing a driver miscompile

1. Reduce the failing kernel until the wrong result comes from a few instructions.
   Dump the module with `LAVA_SPIRV_DUMP_DIR` and compare it with a working variant.
2. Check that the module is valid: `spirv-val`, and the Vulkan validation layer. An
   invalid module is an emitter bug.
3. If it is valid and one driver still computes something else, emit a different
   valid pattern that every driver handles. Prefer rewrites that cost nothing on the
   drivers that were already right.
4. Add a test named for the pattern, with a comment stating the symptom and what the
   fix emits.
5. Run Lava's suite and Mantle's execution tests on every device you have.

## Tests

```julia
import Pkg; Pkg.test("Lava")
```

Lava's suite needs no GPU: it compiles kernels and shaders, checks the emitted
SPIR-V and validates it with `spirv-val`. Execution tests are in Mantle's suite,
which runs every portable test file on each backend a machine has.

Every emitter or pass fix comes with a test in `test/spirv/` (emission patterns) or
`test/` (LLVM passes), and every new source file is listed in
`test/test_compiler_runtime_split.jl`.

## Compile errors that only appear on one machine

A kernel that compiles elsewhere and fails with `dynamic dispatch` or
`heap allocation` usually has a type instability in the code it calls; the error's
call chain ends at the user-code line where it starts. Retry once from a fresh
session first: stale method definitions in a long-running session can produce
inference failures that do not reproduce.
