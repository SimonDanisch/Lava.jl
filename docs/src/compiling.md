# Compiling

## `lava_compile`

```julia
lava_compile(f, tt; stage = :compute, workgroup_size = (64, 1, 1),
             config = nothing, validate = true) -> CompilationResult
```

Compiles `f` for the argument types `tt` (a `Tuple` type) into one SPIR-V module.

* `stage` is `:compute`, `:vertex`, `:fragment`, `:geometry`, `:tess_control`,
  `:tess_eval`, `:task` or `:mesh`.
* `workgroup_size` is the compute stage's `LocalSize`. It is part of the module.
* `config` carries a stage's configuration: a [`GeometryConfig`](@ref) for a geometry
  stage, a [`TessConfig`](@ref) for tessellation, a `MeshConfig` for a mesh stage.
* `validate` runs `spirv-val` on the result.

The [`CompilationResult`](@ref) keeps every step:

| field | content |
|---|---|
| `pre_pass_ir` | LLVM IR as GPUCompiler produced it |
| `post_pass_ir` | LLVM IR after Lava's passes, what the emitter reads |
| `spirv_bytes` | the module |
| `spirv_disasm` | its `spirv-dis` disassembly |
| `entry_name`, `stage`, `workgroup_size` | the entry point |
| `push_info` | the push-constant layout of the argument buffer |
| `source_map` | SPIR-V result id to Julia file and line |

A runtime does not call `lava_compile`. Mantle compiles through Lava's cached entry
points, which keep compiled kernels with their method instances (see
[Architecture](architecture.md)).

## Kernel functions

A compute kernel is a plain Julia function that returns `nothing`. It reads its
position from KernelInterface:

```julia
import KernelInterface as KI

function scale!(y, x, a)
    i = KI.get_global_id().x
    i <= length(y) && (@inbounds y[i] = a * x[i])
    return nothing
end
```

KernelAbstractions `@kernel` functions compile the same way; `@index`, `@localmem`,
`@synchronize` and `@private` lower to the same intrinsics.

Everything the kernel calls has to be inferable and must not allocate: no dynamic
dispatch, no access to non-constant globals, no heap objects. A violation is a
`LavaCompilationError` that names the reason and the chain of calls from the kernel
to it:

```
LavaCompilationError during kernel compilation:
  Cannot compile ms_mesh(...)

  Call chain(s) with GPU-incompatible code:
  1. MeshWrapper @ gfx_intrinsics.jl:1007 → ms_mesh @ test_many_stage_args.jl:44
     Problem: global variable access
```

## Arguments

Arguments are `isbits` values. A device array is a [`LavaDeviceArray`](@ref), a
pointer and dimensions; a runtime converts host arrays into one with Adapt before
the launch. Structs holding device arrays are passed by value the same way.

Every argument is read through one buffer device address. The entry point takes a
single 64-bit push constant pointing at an argument buffer that holds the arguments
in order at their natural alignment, so neither the 128-byte push-constant minimum
nor descriptor sets limit how many arguments a kernel has. A compute kernel that can
throw takes a second 64-bit word, through which it reports the exception.

Each argument's memory is a `PhysicalStorageBuffer` pointer in SPIR-V. When the
runtime knows that a kernel only reads an argument and that no written argument
aliases it, loads through it are marked non-writable, which lets drivers use scalar
or constant caches.
