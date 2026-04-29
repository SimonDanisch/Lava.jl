# GPU compute kernel composing GJK + EPA over pairs of instance transforms.
#
# narrow_phase_kernel: one thread per (i, j) pair index.  Reads transforms[i]
# and transforms[j], runs gjk(); on overlap runs epa() and writes the
# EPAResult into results[k] (where k is the linear pair index).  On
# separation, writes the NO_CONTACT sentinel (depth == 0f0).
#
# This is the P4.4 building block: P4.5 will add a ContactRecord struct +
# atomic compaction; P4.6 will produce the `pairs` buffer from a broad-phase
# rayQuery sweep; P4.7 stitches the three together end-to-end.  For now the
# `pairs` buffer is supplied externally (CPU test harnesses or future
# pair-discovery kernels).
#
# Convention: pair indices are 1-based Julia indices (transforms is a Julia
# AbstractVector, indexed from 1).  Both bodies share the same `shape`
# argument; mixed-shape narrow-phase needs an enum-dispatch design that's
# out of scope for P4.4.

"""
    ContactRecord

Per-contact record produced by the narrow-phase pipeline and consumed by the
XPBD solver (P5).  Layout mirrors the design spec section 3.4:

- `i`, `j`: 1-based grain indices for the contact pair.
- `n_hat`: contact normal pointing from B (j) toward A (i), unit length.
- `p`: contact point on A's surface in world space.
- `depth`: penetration depth (>= 0) along `n_hat`.

Tightly packed (no padding) at 36 bytes; `isbitstype` so it can live in a
GPU-resident `LavaArray`.
"""
struct ContactRecord
    i::UInt32
    j::UInt32
    n_hat::Vec3f
    p::Vec3f
    depth::Float32
end

"""
    NO_CONTACT

Sentinel `EPAResult` written into the result slot for pairs that do not
overlap.  Distinguished from a valid contact by `r.depth == 0f0`; downstream
consumers (P4.5 contact compaction, P4.7 integration test) test on
`r.depth == 0f0` to skip non-contacts.

`converged` is set to `true` because there's nothing to converge -- a future
reader who wants "did EPA fail to converge?" must check `r.depth > 0f0` AND
`r.converged == false` together.
"""
const NO_CONTACT = EPAResult(Vec3f(0f0, 0f0, 0f0), 0f0, Vec3f(0f0, 0f0, 0f0), 0, true)

"""
    narrow_phase_kernel(transforms, pairs, shape, results)

`KernelAbstractions.@kernel`: one thread per pair index `k`.  Reads
`pairs[k] = (i, j)` (1-based indices into `transforms`), looks up the row-
major 3x4 transforms `T_A = transforms[i]` and `T_B = transforms[j]`, runs
`gjk(shape, shape, T_A, T_B)`.  If GJK reports overlap, runs `epa(...)` and
writes the resulting `EPAResult` into `results[k]`; otherwise writes the
`NO_CONTACT` sentinel (`depth == 0f0`).

Preconditions (not validated -- this is a kernel; violations produce wrong
output, not errors):
- `length(results) == length(pairs)`.
- Every `(i, j) ∈ pairs` satisfies `1 ≤ i, j ≤ length(transforms)`.

Output sentinel: `r.depth == 0f0` ⇔ no overlap for that pair.

Both bodies share `shape`.  Mixed-shape narrow-phase (cube vs sphere etc.)
is a future enhancement; for now the kernel is specialized at compile time
on the concrete `shape` type, which lets `support()` inline cleanly into
the GPU compile.

Element types:
- `transforms`:  `AbstractVector{NTuple{12, Float32}}` (matches
  `LavaInstanceRecord.transform` layout).
- `pairs`:       `AbstractVector{NTuple{2, Int32}}` (4-byte ints; halves
  the buffer footprint vs `Int64`).
- `shape`:       `ConvexShape` subtype, e.g. `UnitCube()`.
- `results`:     `AbstractVector{EPAResult}` of length `== length(pairs)`.
"""
@kernel function narrow_phase_kernel(
        @Const(transforms),
        @Const(pairs),
        shape,
        results)
    k = @index(Global)
    @inbounds pair = pairs[k]
    i = pair[1]
    j = pair[2]
    @inbounds T_A = transforms[i]
    @inbounds T_B = transforms[j]
    g = gjk(shape, shape, T_A, T_B)
    if g.overlap
        @inbounds results[k] = epa(shape, shape, T_A, T_B, g.simplex)
    else
        @inbounds results[k] = NO_CONTACT
    end
end
