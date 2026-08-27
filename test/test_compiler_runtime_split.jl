"""
Lava names no Vulkan type. Asserted from the source, on every file.

This file used to describe a LINE inside Lava — a compiler half that must not
reach into a runtime half that was still in the same package. On 2026-08-27 the
runtime moved to `Mantle/src/vulkan/` and the line became the package boundary,
so the assertion got simpler and stronger: not "no file on that side", but **no
file**.

What had to be untangled before the move was possible, each of which would put a
`VkContext` back here if it came undone:

  * `DeviceCaps` was written twice, field for field, in Lava and in Mantle, and
    bridged by a POSITIONAL copy — both definitions carried a comment warning
    that a field inserted anywhere but the end would misalign it silently. It is
    `KernelInterface`'s now, and `caps` is one function with a method on each
    side.
  * The emitter read `VK_CONTEXT_REF[]` for two booleans (SER, ray-query). It
    reads `TargetFeatures` now, which the runtime pushes from `bind_context!`.
  * `spirv_content_hash` lived with the pipeline code; both sides reached for it.
  * The frozen cache mixed a SPIR-V cache with a `VkPipelineCache` blob store.
    The ray-tracing entries take no context and the compute ones take one — that
    asymmetry was the split.

Parsed, not grepped, so a `VkContext` named in a docstring does not count. Several
are, deliberately, to say where something went; `Vulkan` itself still appears 87
times in prose and once as `MemModel.Vulkan`, a SPIR-V enum this package defines.

The list of files is exhaustive and checked: a new file under `src/` that is not
in it fails, because a compiler gaining a source file is worth one line of
acknowledgement.
"""

using Test, Lava

const LAVA_SOURCES = [
    "Lava.jl",
    "graphics/types.jl",
    "runtime/errors.jl",
    "runtime/intrinsics.jl",
    "compiler/phase_timer.jl", "compiler/target.jl", "compiler/entry_wrapper.jl",
    "compiler/compilation.jl", "compiler/target_features.jl", "compiler/frozen_spirv.jl",
    "compiler/frozen_world.jl",
    "compiler/spirv/module.jl", "compiler/spirv/content_hash.jl", "compiler/spirv/types.jl",
    "compiler/spirv/emit.jl", "compiler/spirv/raytracing.jl", "compiler/spirv/rayquery.jl",
    "compiler/spirv/coopmat.jl", "compiler/spirv/graphics.jl",
    "compiler/passes/lift_geps.jl", "compiler/passes/retype_allocas.jl",
    "compiler/passes/cfg_utils.jl", "compiler/passes/structurize_cfg.jl",
    "compiler/passes/prepare_vulkan.jl", "compiler/passes/lower_intrinsics.jl",
    "device/math.jl", "device/quirks.jl", "device/rt_intrinsics.jl",
    "device/ray_query_intrinsics.jl", "device/gfx_intrinsics.jl", "device/printf.jl",
    "device/atomics.jl", "device/subgroup.jl", "device/acceleratedmatrix.jl",
    "device/coopmat_intrinsics.jl", "device/tensor_intrinsics.jl",
    "device/kernelinterface.jl", "device/devicearray.jl",
]

# `Vulkan` is the package; the other three are the runtime's entry points into it.
const VULKAN_NAMES = Set([:Vulkan, :VkContext, :vk_context, :VK_CONTEXT_REF])

"""Every identifier a file references, from its syntax — docstrings excluded for
free, since a docstring is a string literal and not a symbol."""
function referenced_names(path::AbstractString)
    out = Set{Symbol}()
    function walk(ex)
        ex isa Symbol && return push!(out, ex)
        ex isa Expr || return
        ex.head === :quote && return
        if ex.head === :.                      # `a.b`: `b` is a field name
            walk(ex.args[1]); return
        elseif ex.head === :macrocall
            m = ex.args[1]
            m isa Symbol && push!(out, m)
            m isa Expr && m.head === :. && walk(m.args[1])
            foreach(walk, ex.args[2:end]); return
        elseif ex.head === :kw                 # `f(; key = v)`: `key` is not a name
            length(ex.args) > 1 && walk(ex.args[2]); return
        elseif ex.head === :const || ex.head === :global
            # The left-hand side is a DEFINITION, not a reference. Without this,
            # `module MemModel; const Vulkan = UInt32(3); end` — a SPIR-V
            # memory-model enum this package owns — reads as naming the package.
            a = ex.args[1]
            if a isa Expr && a.head === :(=)
                walk(a.args[2]); return
            end
        elseif ex.head === :import || ex.head === :using
            # An `import Vulkan` WOULD count, so these are inspected rather than
            # skipped: the module path is exactly what must not appear.
            foreach(walk, ex.args); return
        end
        foreach(walk, ex.args)
        return
    end
    walk(Meta.parseall(read(path, String)))
    return out
end

@testset "Lava names no Vulkan" begin
    src = joinpath(pkgdir(Lava), "src")
    found = String[]
    for (root, _, files) in walkdir(src), f in files
        endswith(f, ".jl") && push!(found, relpath(joinpath(root, f), src))
    end

    # Exhaustive in both directions: nothing unlisted, nothing stale.
    @test sort(found) == sort(LAVA_SOURCES)

    @testset "$f" for f in LAVA_SOURCES
        hits = intersect(referenced_names(joinpath(src, f)), VULKAN_NAMES)
        # Named, not counted: a failure says which file reached and for what.
        @test (f, sort(collect(hits); by = string)) == (f, Symbol[])
    end

    # And the dependency itself is gone, which is the fact the file names assert
    # one at a time. `Vulkan` not being loadable from inside Lava is what makes
    # the rest of it true rather than merely tidy.
    @testset "Vulkan is not a dependency" begin
        toml = read(joinpath(pkgdir(Lava), "Project.toml"), String)
        deps = split(split(toml, "[deps]")[2], "[")[1]
        for pkg in ("Vulkan", "GLFW", "GPUArraysCore", "LinearAlgebra")
            @test !occursin(Regex("^$pkg = ", "m"), deps)
        end
        # The compiler's own dependencies are still there.
        for pkg in ("LLVM", "GPUCompiler", "SPIRV_Tools_jll", "KernelInterface")
            @test occursin(Regex("^$pkg = ", "m"), deps)
        end
    end
end
