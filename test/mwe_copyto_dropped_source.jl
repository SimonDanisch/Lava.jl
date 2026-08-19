# A device-to-device `copyto!` whose SOURCE is dropped before the batch submits.
#
#   copyto!(dst, Adapt.adapt(backend, host_array))
#
# The temporary source has no reference left after the call, the GC collects it,
# its `DataRef` releases, the pooled block goes back — and the recorded
# `vkCmdCopyBuffer` still names it. `submit!` catches it:
#
#   AssertionError: sync_access!: buffer is not ALIVE (state=1) — use-after-free
#
# NOT FIXED. This file is the reproducer and the diagnosis; run it to see the
# assertion.
#
# ── Why the existing pin does not cover it ────────────────────────────────────
#
# `cmd_copy_buffer!` does pin both ends (`pin!(batch, src)`, `pin!(batch, dst)`),
# and its docstring says a `VkManagedBuffer` is "pinned + sync-tracked" while
# only raw `Vulkan.Buffer` handles need the caller to keep them alive. That is
# the intent, and the pin is not enough to deliver it: `batch.pinned` holds the
# `VkManagedBuffer` OBJECT, which keeps it reachable for the GC, while the thing
# that decides whether the memory is still ours is the `DataRef`'s refcount. The
# array's finalizer releases that ref, the releaser returns the block to the
# pool, and the still-pinned `VkManagedBuffer` now describes memory the pool has
# handed onward.
#
# ── Why the obvious fix is not a one-liner ────────────────────────────────────
#
# `pin!(batch, copy(src.buf))` retains the ref (that is what `Base.copy` on a
# `DataRef` does) — but nothing releases it. `batch.pinned` is dropped wholesale
# at teardown, and a `DataRef` copy has no finalizer of its own; only
# `unsafe_free!` releases. So a retain at record time needs a matching release at
# batch completion, which means a list the batch owns and drains rather than a
# pin. That is a change to the lifetime layer, and this repository has had three
# races there already, so it wants its own pass with RAM and VRAM tracked either
# side — see the project note on tracking both.
#
# ── Scope ─────────────────────────────────────────────────────────────────────
#
# `copyto!` from a HOST array is fine and is the common spelling:
# `copyto!(device_array, host_matrix)` stages through `copy_buffer!` and keeps
# what it needs alive. Only a device-to-device copy from a value nobody holds is
# affected, which is why nothing in the suite hits it. Found on 2026-08-19 while
# writing a scratch comparison, where it looked at first like a bug in the code
# under test.

using Lava, KernelAbstractions, Adapt
import KernelAbstractions as KA

function copy_from_dropped_source(n = 4096)
    backend = Lava.LavaBackend()
    dst = KA.allocate(backend, Float32, n)
    KA.fill!(dst, 0f0)
    copyto!(dst, Adapt.adapt(backend, fill(3f0, n)))   # source unreferenced from here
    GC.gc(true); GC.gc(true)
    KA.synchronize(backend)                            # asserts inside submit!
    return count(==(3f0), Array(dst))
end

# The same copy with the source held, which is what correct code does today.
function copy_from_held_source(n = 4096)
    backend = Lava.LavaBackend()
    dst = KA.allocate(backend, Float32, n)
    KA.fill!(dst, 0f0)
    src = Adapt.adapt(backend, fill(3f0, n))
    copyto!(dst, src)
    GC.gc(true); GC.gc(true)
    KA.synchronize(backend)
    GC.@preserve src nothing
    return count(==(3f0), Array(dst))
end
