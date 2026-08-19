# Variant under test: `vk_free!` never destroys inline.
#
# Every branch above it defers when it can see a reason to — in flight, batch
# recording, wrong thread. What none of them covers is the gap between the
# check and the destroy: they READ whether a batch is recording, and nothing
# stops one starting immediately after. A buffer destroyed in that gap can still
# be named by a slab the new recording picks up.
#
# So: hand every free to the owning thread's deferred list and let
# `drain_deferred_frees!` do it at a flush/submit boundary, where by
# construction no recording is open. The cost is the one the existing comment
# already prices — "one entry on a list that drain_deferred_frees! empties at
# the next flush or submit".
import sys

import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "src", "runtime", "memory.jl")
src = open(path).read()

OLD = """    destroy_buffer!(buf)
end

\"\"\"Actually destroy a buffer's Vulkan resources."""

NEW = """    # EXPERIMENT (defer-always): never destroy inline from `vk_free!`.
    # Every defer branch above reads a point-in-time fact — in flight, batch
    # recording, wrong thread — and then falls through to a destroy that nothing
    # serialises against a recording starting in between. Deferring
    # unconditionally removes the gap rather than narrowing it; the drain runs
    # at a flush/submit boundary where no recording is open by construction.
    let c = buf.ctx
        if c isa VkContext
            bq = c.default_bq
            lock(bq.deferred_frees_lock) do
                push!(bq.deferred_frees, buf)
            end
            return
        end
    end

    destroy_buffer!(buf)
end

\"\"\"Actually destroy a buffer's Vulkan resources."""

if OLD not in src:
    sys.exit("anchor not found — vk_free! tail changed")
open(path, "w").write(src.replace(OLD, NEW, 1))
print("defer-always applied")
