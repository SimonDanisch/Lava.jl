# Unit test for the vendored `replace_unreachable!` pass.
#
# Why a direct LLVM-IR unit test instead of an end-to-end kernel test:
# GPUCompiler (1.13.x) already lowers all throws/`unreachable` upstream, so by
# the time Lava's custom passes see the module there are *zero* `unreachable`
# terminators left — even for a kernel with an explicit `error(...)` or a
# `@noinline` helper that `throw`s (verified empirically). The pass is therefore
# a no-op on the normal compile path and can only be exercised by feeding it IR
# that still contains `unreachable`. We hand-build such a module here.
#
# What we lock down:
#   1. Every `unreachable` is removed (the emitter cannot represent it) and the
#      result is valid LLVM IR.
#   2. Lowering an `unreachable` in a *non-entry helper* is loud: it `@warn`s,
#      and extra-loud (POINTER) when the helper returns a pointer — because that
#      undef return is liable to be dereferenced by the caller (the exact
#      miscompile that motivated the hardening).
#   3. Lowering an `unreachable` in the *entry/kernel* itself is silent (there
#      it is a legitimate early thread-exit, the best a GPU can do).

using Test
using Lava
using LLVM
using Logging

# Count `unreachable` terminators across every defined function in a module.
function count_unreachable(mod::LLVM.Module)
    total = 0
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            inst isa LLVM.UnreachableInst && (total += 1)
        end
    end
    return total
end

# An entry kernel + two value-returning helpers, each with a throw path that
# ends in `unreachable`. The pointer-returning helper is the dangerous case.
const _IR_KERNEL_AND_HELPERS = """
define void @kernel(i1 %c) {
entry:
  br i1 %c, label %bad, label %ok
bad:
  unreachable
ok:
  ret void
}

define i32 @helper_i32(i1 %c) {
entry:
  br i1 %c, label %bad, label %ok
bad:
  unreachable
ok:
  ret i32 42
}

define ptr @helper_ptr(i1 %c) {
entry:
  br i1 %c, label %bad, label %ok
bad:
  unreachable
ok:
  ret ptr null
}
"""

const _IR_KERNEL_ONLY = """
define void @kernel(i1 %c) {
entry:
  br i1 %c, label %bad, label %ok
bad:
  unreachable
ok:
  ret void
}
"""

@testset "replace_unreachable!" begin
    @testset "lowers every unreachable to valid IR" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_KERNEL_AND_HELPERS)
            entry = LLVM.functions(mod)["kernel"]
            @test count_unreachable(mod) == 3
            # Suppress the (expected) warnings here; correctness is the focus.
            Logging.with_logger(Logging.NullLogger()) do
                Lava.replace_unreachable!(mod, entry)
            end
            @test count_unreachable(mod) == 0
            # The result must be structurally valid LLVM IR.
            @test (LLVM.verify(mod); true)
        end
    end

    @testset "warns for non-entry helpers, loud for pointer return" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_KERNEL_AND_HELPERS)
            entry = LLVM.functions(mod)["kernel"]
            # A warning naming each helper must fire; the pointer helper's
            # warning must additionally flag the POINTER hazard. match_mode=:any
            # ignores ordering and any non-matching logs.
            @test_logs(
                (:warn, r"helper_i32"),
                (:warn, r"helper_ptr"),
                (:warn, r"POINTER"),
                match_mode = :any,
                Lava.replace_unreachable!(mod, entry),
            )
        end
    end

    @testset "entry/kernel lowering is silent" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_KERNEL_ONLY)
            entry = LLVM.functions(mod)["kernel"]
            # No warn/error-level logs when the only lowered unreachable is in
            # the entry kernel itself.
            @test_logs min_level = Logging.Warn Lava.replace_unreachable!(mod, entry)
            @test count_unreachable(mod) == 0
        end
    end

    @testset "no entry given: backward-compatible, still silent" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_KERNEL_AND_HELPERS)
            # The legacy 1-arg form (entry === nothing) must not warn and must
            # still remove every unreachable.
            @test_logs min_level = Logging.Warn Lava.replace_unreachable!(mod)
            @test count_unreachable(mod) == 0
        end
    end
end
