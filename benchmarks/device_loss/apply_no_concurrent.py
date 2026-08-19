# Variant: no concurrent dispatch groups.
#
# `concurrent_dispatch_group` sets a global that elides the barriers between the
# dispatches inside it, and `concurrent_indirect_group` additionally defers each
# indirect dispatch so one fused multi-prepare covers them all. Both exist for
# throughput, and both mean the GPU may run those dispatches OVERLAPPED — which
# is how a workload with deterministic sampling can still fail intermittently,
# because the overlap order is not deterministic.
#
# It is also where two of the three races already found in this codebase lived:
# a mid-recording slab reset, and a group-elided prepare->indirect barrier.
#
# So: make both wrappers pass-throughs. Every dispatch gets its own barrier, and
# every indirect dispatch records its own prepare inline — the behaviour from
# before the 2026-06-10 grouping work, which was correct then.
import sys

import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "src", "runtime", "command.jl")
src = open(path).read()

OLD_DISPATCH = """function concurrent_dispatch_group(f::F) where F
    prev_active  = CONCURRENT_GROUP_ACTIVE[]
    prev_started = CONCURRENT_GROUP_STARTED[]
    CONCURRENT_GROUP_ACTIVE[]  = true
    CONCURRENT_GROUP_STARTED[] = false"""

NEW_DISPATCH = """function concurrent_dispatch_group(f::F) where F
    # EXPERIMENT (no-concurrent): run the body with grouping OFF, so every
    # dispatch keeps its own barrier and nothing overlaps.
    if EXPERIMENT_NO_CONCURRENT[]
        return f()
    end
    prev_active  = CONCURRENT_GROUP_ACTIVE[]
    prev_started = CONCURRENT_GROUP_STARTED[]
    CONCURRENT_GROUP_ACTIVE[]  = true
    CONCURRENT_GROUP_STARTED[] = false"""

OLD_INDIRECT = """function concurrent_indirect_group(f::F, bq::BatchQueue = vk_context().default_bq) where F
    prev = bq.deferred_indirect"""

NEW_INDIRECT = """function concurrent_indirect_group(f::F, bq::BatchQueue = vk_context().default_bq) where F
    # EXPERIMENT (no-concurrent): no deferral, so each indirect dispatch records
    # its own prepare and barrier inline rather than sharing a fused one.
    if EXPERIMENT_NO_CONCURRENT[]
        return f()
    end
    prev = bq.deferred_indirect"""

if OLD_DISPATCH not in src or OLD_INDIRECT not in src:
    sys.exit("anchors not found — the group wrappers changed")

src = src.replace(OLD_DISPATCH, NEW_DISPATCH, 1)
src = src.replace(OLD_INDIRECT, NEW_INDIRECT, 1)

# The toggle itself, next to the globals it shadows.
ANCHOR = "function concurrent_dispatch_group(f::F) where F"
src = src.replace(ANCHOR,
                  'const EXPERIMENT_NO_CONCURRENT = Ref(true)\n\n' + ANCHOR, 1)
open(path, "w").write(src)
print("no-concurrent applied")
