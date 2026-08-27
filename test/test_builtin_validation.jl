"""
Every SPIR-V builtin Lava can emit produces a module `spirv-val` accepts.

An `llvmcall` intrinsic emits nothing until a kernel calls it, so an entry in
`SPIRV_BUILTIN_MAP` with no caller has never been through the validator. That is
not hypothetical: `lava_workgroup_size` was declared, called by nothing in `src/`
or `test/`, and emits a module Vulkan rejects — 26955 passing tests said nothing,
because not one of them built a module containing it.

Two builtins were nearly added on the same blind spot: `SubgroupMaxSize`, which
spirv-val rejects outright under the Kernel (OpenCL) capability, and
`NumSubgroups`/`SubgroupId`, which are fine. The difference was invisible until a
kernel called them.

So: one trivial kernel per builtin, compiled and validated. The cost is seconds
and it closes the gap for every future addition to the table.
"""

using Test, Lava

# `compile_and_disasm` comes from `SPIRVTestUtils`, which `runtests.jl` includes
# once and imports names out of. Included here too when it has not been, so the
# file runs on its own — which is how it gets run while a builtin is being added,
# and a test that only works inside the full suite is one that does not get run
# then.
#
# The guard is on the MODULE, not on the function: the helper defines a module
# rather than bare functions, so `isdefined(:compile_and_disasm)` is false even
# right after the include, and the include would run a second time.
@isdefined(SPIRVTestUtils) || include(joinpath(@__DIR__, "spirv_test_utils.jl"))
using .SPIRVTestUtils: compile_and_disasm

@testset "SPIR-V builtins validate" begin
    # Each entry: a kernel calling exactly one builtin, so a rejection names it.
    # 3D builtins take a 1-based dimension; the 1D ones take nothing.
    BUILTIN_KERNELS = [
        ("GlobalInvocationId",        (o) -> (@inbounds o[1] = Lava.lava_global_invocation_id(1); nothing)),
        ("LocalInvocationId",         (o) -> (@inbounds o[1] = Lava.lava_local_invocation_id(1);  nothing)),
        ("WorkgroupId",               (o) -> (@inbounds o[1] = Lava.lava_workgroup_id(1);         nothing)),
        ("NumWorkgroups",             (o) -> (@inbounds o[1] = Lava.lava_num_workgroups(1);       nothing)),
        ("LocalInvocationIndex",      (o) -> (@inbounds o[1] = Lava.lava_local_invocation_index(); nothing)),
        ("SubgroupSize",              (o) -> (@inbounds o[1] = Lava.lava_subgroup_size();          nothing)),
        ("SubgroupLocalInvocationId", (o) -> (@inbounds o[1] = Lava.lava_subgroup_local_id();      nothing)),
        ("NumSubgroups",              (o) -> (@inbounds o[1] = Lava.lava_num_subgroups();          nothing)),
        ("SubgroupId",                (o) -> (@inbounds o[1] = Lava.lava_subgroup_id();            nothing)),
    ]

    tt = Tuple{Lava.LavaDeviceArray{UInt32,1}}

    @testset "$name" for (name, kern) in BUILTIN_KERNELS
        # `validate=true` runs spirv-val; a rejection throws.
        @test (compile_and_disasm(kern, tt; validate = true); true)
    end

    # The known-bad one, pinned rather than left silent.
    #
    # Vulkan requires `WorkgroupSize` to decorate a CONSTANT — the size is fixed
    # when the kernel is compiled, so SPIR-V wants an `OpConstantComposite`, not
    # a load from an Input variable:
    #
    #   BuiltIn decoration on target '%__spirv_BuiltInWorkgroupSize'
    #   must be a constant for WorkgroupSize
    #
    # `KernelInterface.get_local_size`/`get_global_size` wait on this. When the
    # emitter learns to emit the constant, this flips to a pass and moves up into
    # the list above.
    @testset "WorkgroupSize (known broken)" begin
        wg_kernel = (o) -> (@inbounds o[1] = Lava.lava_workgroup_size(1); nothing)
        @test_broken (compile_and_disasm(wg_kernel, tt; validate = true); true)
    end

    # Every COMPUTE builtin is covered above. `SPIRV_BUILTIN_MAP` is merged at
    # load time with the graphics (`graphics.jl:31`) and ray-tracing
    # (`raytracing.jl:37`) tables, and those are only legal in their own
    # execution models — a compute kernel reading `FragCoord` or `LaunchIdKHR`
    # would be rejected for the stage, not for the builtin. They want the same
    # treatment in a test that compiles at `stage = :fragment` / `:raygen`,
    # which `compile_and_disasm` supports; this file is the compute half.
    #
    # Asserted rather than trusted: a new compute builtin with no kernel above
    # is precisely the hole this file exists to close.
    @testset "every compute builtin is covered" begin
        compute = setdiff(Set(keys(Lava.SPIRV_BUILTIN_MAP)),
                          Set(keys(Lava.SPIRV_GFX_BUILTIN_MAP)),
                          Set(keys(Lava.SPIRV_RT_BUILTIN_MAP)))
        tabled  = Set(replace(k, "__spirv_BuiltIn" => "") for k in compute)
        covered = Set(first.(BUILTIN_KERNELS)) ∪ Set(["WorkgroupSize"])
        @test setdiff(tabled, covered) == Set{String}()
    end
end
