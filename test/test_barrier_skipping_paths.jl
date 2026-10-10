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
# And it once saw no barrier at all. With inlining off, each `@synchronize`
# survives as its own wrapper function (`call @llvm.spv...barrier; ret`) rather
# than an inlined intrinsic, so the pass found zero barrier blocks and did
# nothing: an `error()` path that returned early skipped a barrier the other
# invocations reached, a deadlock on lavapipe and lost writes on hardware.
# `function_contains_barrier` looks through such wrapper calls now. That unit
# test came from Mantle's `test/vulkan/test_barrier_skip.jl`; the kernel half,
# a dead invocation that still reaches the second barrier, is Mantle's
# (`test/test_kernel_error.jl`, on every backend).
#
# The rerouted invocation also STOPS: it reaches the barriers and nothing else.
# Every store after the reroute target runs only while a flag the rerouted block
# sets is clear (`stopdeadinvocations!`). Before that it ran the rest of the
# kernel and stored its results, which is not what a throw means anywhere else.
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
    [s.name for s in only(b for b in f.blocks if b.name == name).terminator.successors]

@testset "fix_barrier_skipping_paths!" begin
    @testset "an empty arm into the barrier-holding tail is left alone" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_TAIL_BARRIER)
            f = mod.functions["kernel"]
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
            f = mod.functions["kernel"]
            Lava.fix_barrier_skipping_paths!(f)
            @test successorsof(f, "skip") == ["tail"]
            @test (LLVM.verify(mod); true)
        end
    end

    @testset "an early return before a barrier is rerouted" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_EARLY_RETURN)
            f = mod.functions["kernel"]
            @test Lava.fix_barrier_skipping_paths!(f)
            @test successorsof(f, "bail") == ["work"]
            @test (LLVM.verify(mod); true)
        end
    end

    @testset "the rerouted invocation stores nothing" begin
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, _IR_EARLY_RETURN)
            f = mod.functions["kernel"]
            Lava.fix_barrier_skipping_paths!(f)
            @test (LLVM.verify(mod); true)
            # The store has a block of its own, entered only on a branch.
            store = only(i for b in f.blocks for i in b.instructions
                         if i isa LLVM.StoreInst && i.operands[1] isa LLVM.ConstantInt &&
                            convert(Int, i.operands[1]) == 1 &&
                            !(LLVM.value_type(i.operands[1]) == LLVM.Int1Type()))
            guarded = LLVM.parent(store)
            preds = collect(LLVM.predecessors(guarded))
            @test length(preds) == 1
            @test length(only(preds).terminator.successors) == 2
            # The barrier is not behind the branch: every invocation reaches it.
            barrierblock = only(b for b in f.blocks for i in b.instructions
                                if i isa LLVM.CallInst &&
                                   i.called_operand.name == "llvm.spv.group.memory.barrier.with.group.sync")
            @test barrierblock !== guarded
            # `bail` sets the flag before it joins `work`.
            bail = only(b for b in f.blocks if b.name == "bail")
            @test any(i -> i isa LLVM.StoreInst &&
                           LLVM.value_type(i.operands[1]) == LLVM.Int1Type(), bail.instructions)
        end
    end

    @testset "function_contains_barrier sees wrapped barriers" begin
        ir = """
        declare void @llvm.spv.group.memory.barrier.with.group.sync()

        define internal void @sync_wrapper() {
          call void @llvm.spv.group.memory.barrier.with.group.sync()
          ret void
        }

        define internal void @plain_helper() {
          ret void
        }

        define internal void @calls_wrapper() {
          call void @sync_wrapper()
          ret void
        }
        """
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, ir)
            memo = Dict{LLVM.Function,Bool}()
            barrier = "llvm.spv.group.memory.barrier.with.group.sync"
            fns = mod.functions
            # Direct barrier wrapper, a transitive caller, and a plain helper.
            @test Lava.function_contains_barrier(fns["sync_wrapper"], barrier, memo)
            @test Lava.function_contains_barrier(fns["calls_wrapper"], barrier, memo)
            @test !Lava.function_contains_barrier(fns["plain_helper"], barrier, memo)
        end
    end
end
