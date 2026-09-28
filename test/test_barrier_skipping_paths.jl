# Unit test for `fix_barrier_skipping_paths!`, on hand-built IR.
#
# The pass reroutes an early return — a lowered `throw` — that would let an
# invocation skip a barrier the others reach. It took two ordinary edges for
# one, both in DNNKernels' `attn_flash_rows!`, and both into a return block
# holding the kernel's TAIL rather than only a `ret`:
#
#   * the empty arm of `if c; store; end` before the final barrier. Its sibling
#     reaches a barrier — the same one — so the empty arm was sent through the
#     store and every invocation stored.
#   * a key loop's zero-trip edge, sent into the loop. The exit's phis still
#     named it as a predecessor, which is invalid IR, and spirv-val said so.
#
# Both need the edge to be a block of its own, which is what GVN leaves when it
# hoists a value onto it; whether LLVM makes one is up to LLVM, so a
# kernel-level test cannot pin either.
#
# Not a GPU test: the question is only which block the edge targets.

using Test
using Lava
using LLVM

const _IR_TAIL_BARRIER = """
declare void @llvm.spv.group.memory.barrier.with.group.sync()

define void @kernel(i1 %c, ptr addrspace(3) %p, i32 %x) {
entry:
  br i1 %c, label %store, label %skip
skip:
  %b = or i32 %x, 1
  br label %tail
store:
  %a = or i32 %x, 1
  store i32 %a, ptr addrspace(3) %p
  br label %tail
tail:
  %v = phi i32 [ %a, %store ], [ %b, %skip ]
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  ret void
}
"""

const _IR_ZERO_TRIP = """
declare void @llvm.spv.group.memory.barrier.with.group.sync()

define void @kernel(i1 %c, i32 %n, i32 %x) {
entry:
  br i1 %c, label %pre, label %skip
skip:
  %b = or i32 %x, 1
  br label %tail
pre:
  br label %loop
loop:
  %i = phi i32 [ 0, %pre ], [ %i1, %loop ]
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  %i1 = add i32 %i, 1
  %d = icmp slt i32 %i1, %n
  br i1 %d, label %loop, label %tail
tail:
  %v = phi i32 [ %b, %skip ], [ %i1, %loop ]
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  ret void
}
"""

# The case the pass exists for: `bail` returns before a barrier that `work`
# reaches, so it must be rerouted through `work`. The phi in `done` is there so
# the rerouted edge also has to leave it, and the `lifetime.end` because the
# entry wrapper's exit has them: a return block with only those still returns.
const _IR_EARLY_RETURN = """
declare void @llvm.spv.group.memory.barrier.with.group.sync()
declare void @llvm.lifetime.end.p0(i64, ptr)

define void @kernel(i1 %c, ptr addrspace(3) %p) {
entry:
  %a = alloca i32
  br i1 %c, label %work, label %bail
work:
  store i32 1, ptr addrspace(3) %p
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  br label %done
bail:
  br label %done
done:
  %r = phi i32 [ 0, %bail ], [ 1, %work ]
  call void @llvm.lifetime.end.p0(i64 4, ptr %a)
  ret void
}
"""

successorsof(f, name) =
    [LLVM.name(s) for s in LLVM.successors(LLVM.terminator(only(b for b in LLVM.blocks(f) if LLVM.name(b) == name)))]

@testset "fix_barrier_skipping_paths!" begin
    @testset "an empty arm into the barrier-holding tail is left alone" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_TAIL_BARRIER)
            f = LLVM.functions(mod)["kernel"]
            Lava.fix_barrier_skipping_paths!(f)
            # The empty arm is first, as it was where this was found: visited
            # first, it was the one rerouted.
            @test successorsof(f, "skip") == ["tail"]
            @test successorsof(f, "store") == ["tail"]
            @test (LLVM.verify(mod); true)
        end
    end

    @testset "a loop's zero-trip edge is left alone" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_ZERO_TRIP)
            f = LLVM.functions(mod)["kernel"]
            Lava.fix_barrier_skipping_paths!(f)
            @test successorsof(f, "skip") == ["tail"]
            @test (LLVM.verify(mod); true)
        end
    end

    @testset "an early return before a barrier is rerouted" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_EARLY_RETURN)
            f = LLVM.functions(mod)["kernel"]
            @test Lava.fix_barrier_skipping_paths!(f)
            @test successorsof(f, "bail") == ["work"]
            @test (LLVM.verify(mod); true)
        end
    end
end
