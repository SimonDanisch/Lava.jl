###
# What a kernel has to know about the device it will run on.
#
# One type for the compiler and the graph layer above it, which both name it:
# written twice, field for field, the two can only be bridged by a positional
# copy that misaligns silently.
###

"""
    DeviceCaps

What a kernel has to know about the GPU it will run on, in terms every GPU has.

Nothing here is any one API's. The vocabulary is — "subgroup" is SPIR-V's word,
Metal says simdgroup and CUDA says warp; "workgroup" is SPIR-V's, Metal says
threadgroup — but every field is a fact about the hardware that each of them
reports under some name. Kept in that vocabulary rather than renamed, because a
kernel library already speaks it and a rename buys nothing a comment cannot.

A kernel that picks its tiling from these numbers is portable exactly to the
extent that these numbers are, and that is the whole point of `tile`: Metal's
`simdgroup_matrix` is 8x8 where RDNA 3.5 is 16x16 — a different *number*, not a
different kernel.

`coopmat` is a floor, not a promise about every operation: the cooperative-matrix
operations that exist everywhere are load, store and multiply-add. Anything
narrower — per-element application, in-tile reduction — is a vendor extension and
is not portable even across one API, so a kernel that wants it asks separately
and carries the answer.

    coopmat           cooperative-matrix multiply-add is usable at all
    tile              its tile extent: 16 on RDNA 3.5, 8 for Metal's simdgroup
    subgroup          lanes per subgroup — 32 on NVIDIA, 32 or 64 on RDNA3
    coopmatsubgroup   …and the width a cooperative-matrix kernel actually gets
    sharedbudget      bytes of workgroup-shared memory
    workgrouplimit    threads per workgroup
    cores             SMs / CUs; 0 when the device will not say
    warps             max resident subgroups per core; 0 = ditto
    wggran            workgroup-scope matrix shapes: (invocations, M, N, K) rows
    shapes            every subgroup-scope [`MatrixShape`](@ref) reported

`wggran` is a *table* rather than a number because the answer depends on the
launch: the legal `(M, N, K)` multiples for a matrix spanning the whole workgroup
differ per workgroup size, and coarsen as it grows. Empty means the device has no
workgroup-scope matrices, so it doubles as the capability test — a kernel cannot
learn "may I?" without learning "at what shapes?", and those two must not drift
apart. Read it with [`wggranularity`](@ref).

`shapes` is the subgroup-scope counterpart, and `tile` is one entry of it: the
square fp16 -> fp32 instruction, kept as a field because that is what nearly
every caller wants. Ask [`bestshape`](@ref) for anything else rather than
assuming this device's table looks like the one the kernel was written on.
"""
struct DeviceCaps
    coopmat::Bool
    tile::Int
    subgroup::Int
    coopmatsubgroup::Int
    sharedbudget::Int
    workgrouplimit::Int
    cores::Int
    warps::Int
    wggran::Vector{NTuple{4, Int}}
    shapes::Vector{MatrixShape}
end

# Eight positional arguments still construct one, meaning "no workgroup-scope
# matrices" — every caller that predates `wggran` says exactly that. Those
# callers also mean "a device with a square `tile` fp16 -> fp32 instruction",
# which is the shape table they get: leaving it empty would let `tile` and
# `shapes` describe different devices on a synthetic caps, and that disagreement
# only ever surfaces as a plan declining for a reason the test did not ask about.
DeviceCaps(
    coopmat, tile, subgroup, coopmatsubgroup, sharedbudget,
    workgrouplimit, cores, warps, wggran = NTuple{4, Int}[]
) =
    DeviceCaps(
    coopmat, tile, subgroup, coopmatsubgroup, sharedbudget,
    workgrouplimit, cores, warps, wggran,
    coopmat ? [MatrixShape(Float16, Float32, tile, tile, tile, SubgroupScope())] :
        MatrixShape[]
)

"""
    DeviceCaps(c::DeviceCaps; kw...) -> DeviceCaps

`c` with named fields replaced, for asking what a kernel would decide on a device
that is not this one — a wave64 card, or this card with cooperative matrices
switched off — without that device being present. It is what makes a tiling
decision testable on a machine that cannot run it.

It changes exactly the fields it is given. In particular `coopmat = false` does
NOT empty `shapes`: naming one field and moving three is how a copy stops meaning
what it says. The gate lives in the accessors below, so a caps with
`coopmat = false` may carry a full table and still answer as the device it claims
to be.
"""
DeviceCaps(
    c::DeviceCaps;
    coopmat = c.coopmat, tile = c.tile, subgroup = c.subgroup,
    coopmatsubgroup = c.coopmatsubgroup, sharedbudget = c.sharedbudget,
    workgrouplimit = c.workgrouplimit, cores = c.cores, warps = c.warps,
    wggran = c.wggran, shapes = c.shapes
) =
    DeviceCaps(
    coopmat, tile, subgroup, coopmatsubgroup, sharedbudget,
    workgrouplimit, cores, warps, wggran, shapes
)

"""
    caps(backend::KI.Backend)::DeviceCaps

What this device can do. Nothing above it needs to know which backend answered.

!!! note
    Backend implementations **must** implement this function. Everything else in
    this file derives from it, so a backend that answers this one question does
    not implement the rest.
"""
function caps end

"""
    wggranularity(c::DeviceCaps, nt::Integer) -> Union{NTuple{3, Int}, Nothing}

The `(M, N, K)` multiples a workgroup-scope fp16 x fp16 -> fp32 matrix must be at
a workgroup of `nt` invocations. `nothing` means this device runs no
workgroup-scope matrix at that workgroup size.

Here rather than in each kernel library: the table is the device's, so the lookup
into it is too. Two callers had written the same loop with different argument
orders, which is the shape that drifts.

The multiples COARSEN as the workgroup grows — on an RTX 4000 Ada, 16/16/16 at 32
and 64 invocations, 32/16/16 at 128, 32/32/16 at 256. So a head dimension of 72
pads to 80 at 128 invocations and to 96 at 256, and the padding is 33% of both
products in the second case. That is a real difference and not a rounding
detail: every tiling measured on that card was faster at 128 for exactly this
reason.
"""
function wggranularity(c::DeviceCaps, nt::Integer)
    for (n, m, nn, k) in c.wggran
        n == nt && return (m, nn, k)
    end
    return nothing
end

"""
    matrix_shapes(backend)::AbstractVector{MatrixShape}
    matrix_shapes(c::DeviceCaps)

Every cooperative-matrix shape the device declares legal, or an empty table if it
has none.

The `coopmat` gate is applied here rather than in the copy constructor, so a
`DeviceCaps` built with `coopmat = false` reports no shapes even while it still
carries the table the driver returned.
"""
matrix_shapes(c::DeviceCaps) = c.coopmat ? c.shapes : MatrixShape[]
matrix_shapes(backend::KI.Backend) = matrix_shapes(caps(backend))

# The rest of `matrix.jl`'s queries, for the two things that hold a shape table.
# They are here rather than there because a method signature is evaluated where
# it is written, and `DeviceCaps` does not exist until this file.
supports(c::DeviceCaps, s::MatrixShape) = supports(matrix_shapes(c), s)
supports(backend::KI.Backend, s::MatrixShape) = supports(matrix_shapes(backend), s)

bestshape(c::DeviceCaps, ab, acc; scope::MatrixScope = SubgroupScope()) =
    bestshape(matrix_shapes(c), ab, acc; scope)
bestshape(backend::KI.Backend, ab, acc; scope::MatrixScope = SubgroupScope()) =
    bestshape(matrix_shapes(backend), ab, acc; scope)
