# The identity of a SPIR-V module: every byte of it.
#
# Moved out of `runtime/pipeline.jl`. It is a hash of a byte vector and touches
# nothing else, and BOTH sides of the compiler/runtime line need it — the dump
# path names a file after it, the pipeline cache keys on it. Whichever side it
# lived on, the other reached across for it, so it lives with the thing it
# identifies.

"""
    spirv_content_hash(bytes) -> UInt64

A hash over **every** byte of a SPIR-V module.

`Base.hash` on a large `Vector` deliberately *samples* elements instead of
reading all of them. For a dictionary of distinct arrays that is a sensible
trade; as a pipeline-cache key it is a silent-wrong-results bug, because two
instantiations of one kernel can differ in a handful of bytes and nothing else.

Measured: the 256- and 512-wide instantiations of the same kernel produce modules
that differ at **exactly one byte** — index 230, the `LocalSize` x operand — and
`hash` returns the identical value for both. The 512-wide launch therefore looked
up the 256-wide pipeline, dispatched a 256-thread shader over a grid computed for
512, and wrote exactly half its output. At 1024 it wrote a quarter. Whichever
size compiled first won, which is why it looked order- and body-dependent, and
why adding any unrelated store "fixed" it (different bytes, no collision).

That is the entire content of what was recorded for a long time as "above 256
this device silently runs fewer invocations than the shader declares". The device
runs every lane.

FNV-1a, because it reads every byte and needs no dependency. A 9 KB module costs
microseconds, and this runs once per pipeline creation, never per launch.
"""
function spirv_content_hash(bytes::AbstractVector{UInt8})
    h = 0xcbf29ce484222325
    @inbounds for b in bytes
        h = (h ⊻ b) * 0x00000100000001b3
    end
    return h
end
