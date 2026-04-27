# ============================================================================
# Lava.HWTLAS — concrete hardware-accelerated TLAS living in Lava
# ============================================================================
#
# This is the Lava-native implementation of Raycore.AbstractAccel.  It fuses
# the previously split Raycore.HWTLAS (backend-agnostic stubs) and
# RaycoreLavaExt (Lava-specific bindings) into one concrete, fully-typed type.
#
# Phase C of the Raycore HWTLAS release cleanup.  Raycore.HWTLAS continues to
# exist until Phase E; both coexist temporarily.

import Raycore
import Adapt
using GeometryBasics
using StaticArrays: SVector, SMatrix
using Base: @propagate_inbounds

const Mat4f = SMatrix{4, 4, Float32, 16}

# `unsafe_free!(::Nothing)` lives in `raytracing/acceleration.jl` alongside
# the other `unsafe_free!` methods so the method table is easy to audit.

# ============================================================================
# InstanceBatch struct
# ============================================================================

"""
    InstanceBatch

A batch of N TLAS instances all referencing the same BLAS, with instance
records read from a GPU-resident `LavaArray{LavaInstanceRecord, 1}` at
sync! / refit time. Returned `handle` lets callers track the batch in
`HWTLAS.handle_to_range` for delete!/refit.
"""
struct InstanceBatch
    blas::LavaBLAS
    instance_buf::LavaArray{LavaInstanceRecord, 1}
    n::Int
    instance_mask::UInt8
    handle::Raycore.TLASHandle
end

# ============================================================================
# HWTLAS struct
# ============================================================================

"""
    HWTLAS{Tri} <: Raycore.AbstractAccel

Lava-native hardware-accelerated TLAS.  Concretely typed on `Tri` (the
per-primitive triangle type, typically `Raycore.Triangle{UInt32}`).

Build geometry with `push!(hwtlas, mesh, transform)`, then call
`Raycore.sync!(hwtlas)` to upload and build the Vulkan AS.  The adapted form
lives in `hwtlas.static_tlas` as a `HWAdaptedAccel{HWTLAS{Tri}}`.

# Adapted-form invariant

`sync!(hwtlas)` is the sole owner of `hwtlas.static_tlas`.  It rebuilds as
efficiently as possible — in place via `resize!`/`copyto!` where the backing
buffer can be reused, reallocated only when a buffer grew — and stores the
result in `hwtlas.static_tlas`.  `sync!` MAY reassign `hwtlas.static_tlas`
when a buffer was reallocated.

Every consumer that hands the accel to a raytracing dispatch MUST go through
`hwtlas.static_tlas` or `Adapt.adapt(backend, hwtlas)` (which reads /
refreshes `hwtlas.static_tlas`) per dispatch.  Both are cheap; `sync!` did
the heavy lifting.  **NEVER cache the `HWAdaptedAccel` returned by `adapt`
across mutations** — consumers that cache silently see stale geometry.

# Non-blocking sync!

`sync!(hwtlas)` does NOT call `KA.synchronize(backend)`.  Old backings are
dropped via `Lava.unsafe_free!`, which defers destruction through
`hwtlas.bq`'s timeline (`bq.deferred_as_frees` / `bq.deferred_frees`) when
prior dispatches are still in flight.  Phase-B pinning of RT closure leaves
(tri_gpu, off_gpu, hw_tlas, hw_accel) is what makes the timeline tracking
correct on the BDA path.

For a CPU-blocking drain use `Raycore.wait_for_gpu!(hwtlas)`, which calls
`vk_flush!(hwtlas.bq)` (waits on the HWTLAS's own queue specifically, not
the backend-wide queue the Raycore default `wait_for_gpu!` uses).
"""
mutable struct HWTLAS{Tri} <: Raycore.AbstractAccel
    backend::LavaBackend

    # BatchQueue for all AS builds and RT dispatches on this HWTLAS.
    bq::BatchQueue

    # Geometry (accumulated on push!)
    blas_list::Vector{LavaBLAS}
    blas_triangles::Vector{Vector{Tri}}
    blas_offsets::Vector{UInt32}

    # Instances
    instance_blas_indices::Vector{Int}
    instance_transforms::Vector{NTuple{12,Float32}}
    instance_custom_indices::Vector{UInt32}
    instance_masks::Vector{UInt8}

    # Instance batches -- N instances of one BLAS each, transforms in a GPU buffer.
    # An HWTLAS uses EITHER per-mesh push! (CPU instance_transforms) OR
    # push_instances! (GPU instance_buf), not both. Mixed mode errors loudly
    # in rebuild_hw_tlas!. To switch modes, build a fresh HWTLAS.
    instance_batches::Vector{InstanceBatch}

    # Handle management
    handle_to_range::Dict{Raycore.TLASHandle, UnitRange{Int}}
    deleted_handles::Set{Raycore.TLASHandle}
    next_handle_id::UInt32

    # Bounding box (CPU-side, updated on push!)
    root_aabb::Raycore.Bounds3

    # Built on sync! — Lava-specific
    hw_tlas::Union{Nothing, LavaTLAS}
    hw_accel::Union{Nothing, HardwareAccel{Vector{Tri}}}
    tri_gpu::Union{Nothing, LavaArray{Tri, 1}}
    off_gpu::Union{Nothing, LavaArray{UInt32, 1}}

    # GPU-adapted form, owned by sync!.  Consumers read this via
    # `hwtlas.static_tlas` or `Adapt.adapt(backend, hwtlas)` per dispatch.
    # `nothing` until first sync!.
    static_tlas::Any   # Union{Nothing, HWAdaptedAccel{HWTLAS{Tri}}}

    dirty::Bool
end

"""
    HWTLAS{Tri}(backend::LavaBackend; bq=backend.bq) -> HWTLAS{Tri}

Construct an empty HWTLAS parametrised on triangle type `Tri`.
"""
function HWTLAS{Tri}(backend::LavaBackend; bq::BatchQueue=backend.bq) where {Tri}
    HWTLAS{Tri}(
        backend, bq,
        LavaBLAS[], Vector{Tri}[], UInt32[],
        Int[], NTuple{12,Float32}[], UInt32[], UInt8[],
        InstanceBatch[],          # new
        Dict{Raycore.TLASHandle, UnitRange{Int}}(),
        Set{Raycore.TLASHandle}(),
        UInt32(1),
        Raycore.Bounds3(),
        nothing, nothing, nothing, nothing,
        nothing,
        true,
    )
end

"""
    HWTLAS(backend::LavaBackend; bq=backend.bq) -> HWTLAS{Triangle{UInt32}}

Default constructor — narrows to `Triangle{UInt32}`.
"""
HWTLAS(backend::LavaBackend; bq::BatchQueue=backend.bq) =
    HWTLAS{Raycore.Triangle{UInt32}}(backend; bq)

# ============================================================================
# HWAdaptedAccel
# ============================================================================

"""
    HWAdaptedAccel{H<:HWTLAS} <: Raycore.AbstractAdaptedAccel

GPU-adapted form of `HWTLAS`.  Dispatches ray tracing to hardware.
"""
struct HWAdaptedAccel{H<:HWTLAS} <: Raycore.AbstractAdaptedAccel
    hwtlas::H
end

# ============================================================================
# PrecomputedHitsAccel (struct defined here so adapt_structure can reference it)
# ============================================================================

"""
    PrecomputedHitsAccel{R,T,O,Tri} <: Raycore.AbstractAdaptedAccel

Wraps a batch-traced result buffer plus the GPU triangle + offset arrays for
CPU-side `closest_hit` dispatch after `batch_trace_indirect`.

`empty` is a sentinel triangle returned on ray miss (instead of indexing
into `triangles[1]` which is UB when the geometry is empty).
"""
struct PrecomputedHitsAccel{R, T, O, Tri} <: Raycore.AbstractAdaptedAccel
    results::R
    triangles::T
    offsets::O
    empty::Tri
end

# ============================================================================
# Adapt.adapt_structure
# ============================================================================

"""
    Adapt.adapt_structure(to, hwtlas::HWTLAS) -> HWAdaptedAccel

Ensure `sync!` has run then return `hwtlas.static_tlas`.  Cheap on a clean
HWTLAS — do NOT cache the return across mutations.
"""
function Adapt.adapt_structure(to, hwtlas::HWTLAS{Tri}) where Tri
    Raycore.sync!(hwtlas)
    # `static_tlas` is typed `::Any` in the struct body (forward-reference to
    # HWAdaptedAccel{HWTLAS{Tri}}); narrow via return-assertion so callers see
    # a concrete type and the dispatch hot path stays type-stable.
    return hwtlas.static_tlas::HWAdaptedAccel{HWTLAS{Tri}}
end

"""
    Adapt.adapt_structure(to, p::PrecomputedHitsAccel) -> PrecomputedHitsAccel

Adapt all array fields to the target context; copy `empty` verbatim.
"""
function Adapt.adapt_structure(to, p::PrecomputedHitsAccel)
    PrecomputedHitsAccel(
        Adapt.adapt(to, p.results),
        Adapt.adapt(to, p.triangles),
        Adapt.adapt(to, p.offsets),
        p.empty,
    )
end

# ============================================================================
# Accessors
# ============================================================================

Raycore.world_bound(hwtlas::HWTLAS)    = hwtlas.root_aabb
Raycore.n_geometries(hwtlas::HWTLAS)   = length(hwtlas.blas_list)
Raycore.n_instances(hwtlas::HWTLAS)    = length(hwtlas.instance_blas_indices)
Raycore.refit_tlas!(hwtlas::HWTLAS)    = nothing

# RayMakie compat: hwtlas.instances -> lightweight length-only view
struct HWTLASInstances
    n::Int
end
Base.isempty(x::HWTLASInstances) = x.n == 0
Base.length(x::HWTLASInstances) = x.n

function Base.getproperty(hwtlas::HWTLAS, s::Symbol)
    s === :instances ? HWTLASInstances(length(getfield(hwtlas, :instance_blas_indices))) :
                       getfield(hwtlas, s)
end

# ============================================================================
# wait_for_gpu!
# ============================================================================

"""
    Raycore.wait_for_gpu!(hwtlas::HWTLAS) -> hwtlas

Block until all pending GPU work on `hwtlas.bq` has completed.
Uses Lava's `vk_flush!` rather than `KA.synchronize(backend)`.
"""
function Raycore.wait_for_gpu!(hwtlas::HWTLAS)
    vk_flush!(hwtlas.bq)
    return hwtlas
end

# ============================================================================
# C2 — Mutation API
# ============================================================================

function hwtlas_add_geometry!(hwtlas::HWTLAS{Tri}, mesh::GeometryBasics.Mesh) where {Tri}
    nmesh = GeometryBasics.expand_faceviews(mesh)
    fs = decompose(TriangleFace{UInt32}, nmesh)
    verts = decompose(Point3f, nmesh)
    norms = Raycore.Normal3f.(Raycore.decompose_normals(nmesh))
    uvs_raw = GeometryBasics.decompose_uv(nmesh)
    uvs = isnothing(uvs_raw) ? Point2f[] : Point2f.(uvs_raw)
    indices = collect(reinterpret(UInt32, fs))

    has_meta = hasproperty(nmesh, :face_meta)
    n_faces = length(fs)

    cpu_triangles = let tris = Tri[]
        for i in 1:n_faces
            Raycore.is_degenerate_face(verts, indices, i) && continue
            meta = has_meta ? nmesh.face_meta[indices[3*(i-1)+1]] : UInt32(i)
            push!(tris, Raycore.build_triangle(verts, norms, uvs, indices, i, meta))
        end
        tris
    end
    isempty(cpu_triangles) && error("Geometry has no valid triangles")

    # Extract vertex positions for BLAS build
    n_tris = length(cpu_triangles)
    blas_vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        vs = cpu_triangles[i].vertices
        for j in 1:3
            v = vs[j]
            blas_vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end
    blas_indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        blas_indices[i+1] = UInt32(i)
    end

    # Inline of build_hw_blas (RaycoreLavaExt lines 46-50)
    hw_blas = as_build() do ctx
        build_blas(ctx, blas_vertices, blas_indices)
    end

    push!(hwtlas.blas_list, hw_blas)
    push!(hwtlas.blas_triangles, cpu_triangles)
    blas_idx = length(hwtlas.blas_list)

    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    for tri in cpu_triangles
        for v in tri.vertices
            hwtlas.root_aabb = Raycore.union(hwtlas.root_aabb, Raycore.Bounds3(Point3f(v)))
        end
    end

    hwtlas.dirty = true
    return blas_idx
end

"""
Internal: add N instances of `blas_idx` to the HWTLAS.

`instance_ids` (if given) supplies the per-instance interface override that
the HW closest-hit shader reads via `gl_InstanceCustomIndexEXT`.  When
`nothing`, every instance gets `0` (inherit from triangle metadata).

`instance_masks` (if given) supplies per-instance Vulkan cullMask values.
When `nothing`, every instance gets `0xff` (visible to all ray queries).
"""
function hwtlas_add_instances!(hwtlas::HWTLAS, blas_idx::Int, transforms;
                                instance_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                                instance_masks::Union{Nothing, AbstractVector{UInt8}}=nothing)
    if instance_ids !== nothing && length(instance_ids) != length(transforms)
        throw(ArgumentError("instance_ids length $(length(instance_ids)) != transforms length $(length(transforms))"))
    end
    if instance_masks !== nothing && length(instance_masks) != length(transforms)
        throw(ArgumentError("instance_masks length $(length(instance_masks)) != transforms length $(length(transforms))"))
    end
    start_idx = length(hwtlas.instance_blas_indices) + 1
    for (i, transform) in enumerate(transforms)
        iid  = instance_ids    === nothing ? UInt32(0)  : UInt32(instance_ids[i])
        mask = instance_masks  === nothing ? UInt8(0xff) : UInt8(instance_masks[i])
        push!(hwtlas.instance_blas_indices, blas_idx)
        push!(hwtlas.instance_transforms, mat4_to_vk_transform(transform))
        push!(hwtlas.instance_custom_indices, iid)
        push!(hwtlas.instance_masks, mask)
    end
    end_idx = length(hwtlas.instance_blas_indices)

    handle = Raycore.TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    hwtlas.handle_to_range[handle] = start_idx:end_idx
    return handle
end

function Base.push!(hwtlas::HWTLAS, mesh::GeometryBasics.Mesh, transform::Mat4f=Mat4f(I);
                    instance_id::UInt32=UInt32(0), instance_mask::UInt8=UInt8(0xff))
    blas_idx = hwtlas_add_geometry!(hwtlas, mesh)
    return hwtlas_add_instances!(hwtlas, blas_idx, (transform,);
                                  instance_ids=UInt32[instance_id],
                                  instance_masks=UInt8[instance_mask])
end

function Base.push!(hwtlas::HWTLAS, mesh::GeometryBasics.Mesh, transforms::AbstractVector{Mat4f};
                    instance_ids::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                    instance_mask::UInt8=UInt8(0xff))
    blas_idx = hwtlas_add_geometry!(hwtlas, mesh)
    # Single mask applies to all transforms (same geometry type); per-transform
    # masks are a future feature if needed.
    masks = fill(instance_mask, length(transforms))
    return hwtlas_add_instances!(hwtlas, blas_idx, transforms; instance_ids, instance_masks=masks)
end

"""
    push!(hwtlas::HWTLAS, blas::LavaBLAS, transform::Mat4f=Mat4f(I);
          instance_id::UInt32=UInt32(0)) -> TLASHandle

Register a pre-built `LavaBLAS` (e.g. from `build_blas_aabb`) as a new
geometry + instance.  No triangle data is associated; accordingly the
`hw_accel` / ray-tracing pipeline path is not usable on this HWTLAS after
this call.  Use the compute-rayQuery path (`lava_launch!` with `tlas=hwtlas`)
instead.
"""
function Base.push!(hwtlas::HWTLAS{Tri}, blas::LavaBLAS, transform::Mat4f=Mat4f(I);
                    instance_id::UInt32=UInt32(0), instance_mask::UInt8=UInt8(0xff)) where {Tri}
    # Register the pre-built BLAS — no triangles.
    push!(hwtlas.blas_list, blas)
    push!(hwtlas.blas_triangles, Tri[])
    blas_idx = length(hwtlas.blas_list)
    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    hwtlas.dirty = true
    return hwtlas_add_instances!(hwtlas, blas_idx, (transform,);
                                  instance_ids=UInt32[instance_id],
                                  instance_masks=UInt8[instance_mask])
end

"""
    Raycore.push_instances!(tlas::HWTLAS, blas::LavaBLAS,
                            instance_buf::LavaArray{LavaInstanceRecord, 1};
                            n::Integer, instance_mask::UInt8) -> Raycore.TLASHandle

Register an N-instance batch in the TLAS. All N instances reference the
same `blas`; per-instance transforms / custom_indices live in `instance_buf`
and are written by a GPU compute kernel (see `write_grain_instances_kernel`).

Returns one `TLASHandle` for the whole batch. Subsequent `sync!` builds
the underlying `LavaTLAS` with `allow_update=true` so per-frame refits
via `Raycore.refit_tlas!(::HWTLAS)` work.
"""
function Raycore.push_instances!(tlas::HWTLAS, blas::LavaBLAS,
                                  instance_buf::LavaArray{LavaInstanceRecord, 1};
                                  n::Integer, instance_mask::UInt8 = UInt8(0xff))
    n_int = Int(n)
    n_int <= length(instance_buf) || error(
        "push_instances!: n=$n_int exceeds instance_buf length $(length(instance_buf))")
    handle = Raycore.TLASHandle(tlas.next_handle_id)
    tlas.next_handle_id += UInt32(1)
    push!(tlas.instance_batches, InstanceBatch(blas, instance_buf, n_int, instance_mask, handle))
    tlas.dirty = true
    return handle
end

"""
    update_transform!(hwtlas::HWTLAS, handle::TLASHandle, transform::Mat4f)

Update every instance belonging to `handle` to the same transform.
Marks the HWTLAS dirty so the next `sync!` repacks the instance buffer.
"""
function Raycore.update_transform!(hwtlas::HWTLAS, handle::Raycore.TLASHandle, transform::Mat4f)
    r = get(hwtlas.handle_to_range, handle, nothing)
    r === nothing && return false
    vk_tr = mat4_to_vk_transform(transform)
    for i in r
        hwtlas.instance_transforms[i] = vk_tr
    end
    hwtlas.dirty = true
    return true
end

"""
    update_transform_at!(hwtlas::HWTLAS, handle::TLASHandle, i::Integer, transform::Mat4f)

Update the i-th instance (1-based) within `handle`'s instance range.
"""
function Raycore.update_transform_at!(hwtlas::HWTLAS, handle::Raycore.TLASHandle, i::Integer, transform::Mat4f)
    r = get(hwtlas.handle_to_range, handle, nothing)
    r === nothing && return false
    1 <= i <= length(r) || throw(BoundsError(1:length(r), i))
    hwtlas.instance_transforms[first(r) + i - 1] = mat4_to_vk_transform(transform)
    hwtlas.dirty = true
    return true
end

function Base.delete!(hwtlas::HWTLAS, handle::Raycore.TLASHandle)::Bool
    haskey(hwtlas.handle_to_range, handle) || return false
    handle in hwtlas.deleted_handles && return false
    push!(hwtlas.deleted_handles, handle)
    hwtlas.dirty = true
    return true
end

# ============================================================================
# C3 — rebuild helpers + sync!
# ============================================================================

# Try to reuse `prev` as the GPU sink for `data` via capacity-aware resize+copyto.
# If `prev` has a matching element type, the in-place path keeps the same LavaArray
# identity (and the same backing VkManagedBuffer whenever `data` still fits the
# buffer's existing capacity). Otherwise allocate fresh.
function _reuse_or_alloc(prev, data::AbstractArray{T}) where T
    if prev isa LavaArray{T}
        resize!(prev, length(data))
        copyto!(prev, data)
        return prev
    end
    return LavaArray(data)
end

function rebuild_hw_tlas!(hwtlas::HWTLAS{Tri}) where {Tri}
    if !isempty(hwtlas.instance_batches)
        isempty(hwtlas.instance_blas_indices) || error(
            "HWTLAS: cannot mix instance batches with per-mesh push! instances. " *
            "Use either push_instances! (batch mode) or push! (per-mesh mode), not both.")
        return rebuild_hw_tlas_from_batch!(hwtlas)
    end
    return rebuild_hw_tlas_from_per_instance!(hwtlas)
end

function rebuild_hw_tlas_from_per_instance!(hwtlas::HWTLAS{Tri}) where {Tri}
    n_inst = length(hwtlas.instance_blas_indices)
    blas_refs = [hwtlas.blas_list[hwtlas.instance_blas_indices[i]] for i in 1:n_inst]

    # Per-instance triangle-array offset, indexed by gl_InstanceID (0-based).
    per_instance_tri_offsets = UInt32[hwtlas.blas_offsets[bi] for bi in hwtlas.instance_blas_indices]

    hw_tlas = as_build() do ctx
        build_tlas(ctx, blas_refs;
                   transforms=hwtlas.instance_transforms,
                   custom_indices=hwtlas.instance_custom_indices,
                   masks=hwtlas.instance_masks)
    end

    all_tris = reduce(vcat, hwtlas.blas_triangles)::Vector{Tri}
    per_inst_offsets = collect(per_instance_tri_offsets)

    # Reuse the previous HardwareAccel when we have one -- keeps the same
    # RayTracingPipeline + SBT alive across sync! rebuild cycles.
    hw_accel = if hwtlas.hw_accel isa HardwareAccel
        accel_prev = hwtlas.hw_accel
        accel_prev.tlas = hw_tlas
        accel_prev.triangle_data = all_tris
        accel_prev.blas_offsets = hwtlas.blas_offsets
        accel_prev.per_instance_tri_offsets = per_inst_offsets
        accel_prev
    else
        HardwareAccel(hw_tlas, all_tris, hwtlas.blas_offsets, per_inst_offsets; bq=hwtlas.bq)
    end

    # Grow-and-overwrite on the previous LavaArray when the eltype matches.
    tri_gpu = _reuse_or_alloc(hwtlas.tri_gpu, all_tris)

    # off_gpu is keyed by gl_InstanceID (one entry per instance).
    off_cpu = collect(per_instance_tri_offsets)
    off_gpu = _reuse_or_alloc(hwtlas.off_gpu, off_cpu)

    return (hw_tlas, hw_accel, tri_gpu, off_gpu)
end

function rebuild_hw_tlas_from_batch!(hwtlas::HWTLAS{Tri}) where {Tri}
    length(hwtlas.instance_batches) == 1 || error(
        "HWTLAS: multiple instance_batches not yet supported (got $(length(hwtlas.instance_batches))). " *
        "Push exactly one batch per HWTLAS for now.")
    batch = hwtlas.instance_batches[1]

    hw_tlas = as_build() do ctx
        build_tlas(ctx, batch.instance_buf, batch.n; allow_update=true)
    end

    # Single-batch HWTLAS doesn't need triangle/offset buffers (Hikari renders
    # via the rendering instances directly; physics queries don't need them
    # either -- primitive_index / instance_id come from the rayQuery).
    # Supply empty buffers; consumers that read tri_gpu / off_gpu
    # against a batch HWTLAS will see zero-length arrays.
    all_tris = Tri[]
    per_inst_offsets = UInt32[]

    hw_accel = if hwtlas.hw_accel isa HardwareAccel
        accel_prev = hwtlas.hw_accel
        accel_prev.tlas = hw_tlas
        accel_prev.triangle_data = all_tris
        accel_prev.blas_offsets = UInt32[]
        accel_prev.per_instance_tri_offsets = per_inst_offsets
        accel_prev
    else
        HardwareAccel(hw_tlas, all_tris, UInt32[], per_inst_offsets; bq=hwtlas.bq)
    end

    tri_gpu = _reuse_or_alloc(hwtlas.tri_gpu, all_tris)
    off_gpu = _reuse_or_alloc(hwtlas.off_gpu, per_inst_offsets)

    return (hw_tlas, hw_accel, tri_gpu, off_gpu)
end

"""
    Raycore.sync!(hwtlas::HWTLAS)

Rebuild the hardware TLAS (and any needed BLASes) if dirty.
Does NOT call KA.synchronize — uses Lava's timeline-based deferred free path.
No-op on the topology if already up-to-date (fast path, allocation-free).

After this returns:
- The hardware AS reflects all prior push!/delete! calls.
- `hwtlas.static_tlas` is the fresh adapted form.
- Old LavaTLAS/LavaArray objects that were replaced have been passed to
  `unsafe_free!` for timeline-gated destruction.
"""
function Raycore.sync!(hwtlas::HWTLAS)
    # True no-op when nothing changed and static_tlas is already built.
    if !hwtlas.dirty && hwtlas.static_tlas !== nothing
        return hwtlas
    end

    # First sync! on a freshly-built HWTLAS with no mutations.
    if !hwtlas.dirty
        hwtlas.static_tlas = HWAdaptedAccel(hwtlas)
        return hwtlas
    end

    # BLASes evicted by this rebuild — released after the rebuild.
    dropped_blases = LavaBLAS[]

    # Compact: deleted handles leave orphan entries in instance arrays.
    if !isempty(hwtlas.deleted_handles)
        deleted_inst_idx = BitSet()
        for h in hwtlas.deleted_handles
            r = get(hwtlas.handle_to_range, h, nothing)
            r === nothing && continue
            for i in r
                push!(deleted_inst_idx, i)
            end
            delete!(hwtlas.handle_to_range, h)
        end

        keep_inst = [i for i in eachindex(hwtlas.instance_blas_indices) if !(i in deleted_inst_idx)]
        hwtlas.instance_blas_indices  = [hwtlas.instance_blas_indices[i]  for i in keep_inst]
        hwtlas.instance_transforms    = [hwtlas.instance_transforms[i]    for i in keep_inst]
        hwtlas.instance_custom_indices = [hwtlas.instance_custom_indices[i] for i in keep_inst]
        hwtlas.instance_masks         = [hwtlas.instance_masks[i]         for i in keep_inst]

        # After removing instances, some BLAS may be unreferenced.
        still_used = Set(hwtlas.instance_blas_indices)
        if length(still_used) < length(hwtlas.blas_list)
            old_to_new = Dict{Int,Int}()
            new_blas    = LavaBLAS[]
            new_tris    = Vector{eltype(hwtlas.blas_triangles)}()
            new_offsets = UInt32[]
            running_offset = UInt32(0)
            for (old_idx, blas) in enumerate(hwtlas.blas_list)
                if old_idx in still_used
                    push!(new_blas, blas)
                    push!(new_tris, hwtlas.blas_triangles[old_idx])
                    push!(new_offsets, running_offset)
                    running_offset += UInt32(length(hwtlas.blas_triangles[old_idx]))
                    old_to_new[old_idx] = length(new_blas)
                else
                    push!(dropped_blases, blas)
                end
            end
            hwtlas.blas_list      = new_blas
            hwtlas.blas_triangles = new_tris
            hwtlas.blas_offsets   = new_offsets
            hwtlas.instance_blas_indices = [old_to_new[i] for i in hwtlas.instance_blas_indices]
        end

        # Rebuild handle_to_range against the compacted indices.
        sorted_deleted = sort(collect(deleted_inst_idx))
        shift_of(i) = begin
            s = 0
            for d in sorted_deleted
                d < i && (s += 1)
            end
            s
        end
        new_range = Dict{Raycore.TLASHandle, UnitRange{Int}}()
        for (h, r) in hwtlas.handle_to_range
            lo = first(r) - shift_of(first(r))
            hi = last(r)  - shift_of(last(r))
            new_range[h] = lo:hi
        end
        hwtlas.handle_to_range = new_range
        empty!(hwtlas.deleted_handles)
    end

    n_inst = length(hwtlas.instance_blas_indices)
    if n_inst == 0 && isempty(hwtlas.instance_batches)
        old_hw_tlas = hwtlas.hw_tlas
        old_tri_gpu = hwtlas.tri_gpu
        old_off_gpu = hwtlas.off_gpu
        hwtlas.hw_tlas   = nothing
        hwtlas.hw_accel  = nothing
        hwtlas.tri_gpu   = nothing
        hwtlas.off_gpu   = nothing
        hwtlas.dirty     = false
        hwtlas.static_tlas = HWAdaptedAccel(hwtlas)
        # No KA.synchronize — Lava uses timeline-gated deferred free.
        unsafe_free!(old_hw_tlas)
        unsafe_free!(old_tri_gpu)
        unsafe_free!(old_off_gpu)
        for blas in dropped_blases
            unsafe_free!(blas)
        end
        return hwtlas
    end

    hw_tlas, hw_accel, tri_gpu, off_gpu = rebuild_hw_tlas!(hwtlas)

    old_hw_tlas = hwtlas.hw_tlas
    # tri_gpu/off_gpu get reused-in-place when sizes permit — do NOT release
    # unless the backend returned a different object (couldn't reuse).
    old_tri_gpu = hwtlas.tri_gpu === tri_gpu ? nothing : hwtlas.tri_gpu
    old_off_gpu = hwtlas.off_gpu === off_gpu ? nothing : hwtlas.off_gpu

    hwtlas.hw_tlas   = hw_tlas
    hwtlas.hw_accel  = hw_accel
    hwtlas.tri_gpu   = tri_gpu
    hwtlas.off_gpu   = off_gpu
    hwtlas.dirty     = false
    hwtlas.static_tlas = HWAdaptedAccel(hwtlas)
    # No KA.synchronize — Lava uses timeline-gated deferred free.
    unsafe_free!(old_hw_tlas)
    unsafe_free!(old_tri_gpu)
    unsafe_free!(old_off_gpu)
    for blas in dropped_blases
        unsafe_free!(blas)
    end
    return hwtlas
end

# ============================================================================
# C4 — Trace dispatch
# ============================================================================

function trace_closest_hits!(results, rays, accel::HWAdaptedAccel, n)
    trace_closest_hits!(results, rays, accel.hwtlas.hw_accel, n)
end

function trace_closest_hits_indirect!(results, rays, accel::HWAdaptedAccel, n_buf)
    trace_closest_hits_indirect!(results, rays, accel.hwtlas.hw_accel, n_buf)
end

function batch_trace_indirect(results, rays, accel::HWAdaptedAccel, n_buf)
    trace_closest_hits_indirect!(results, rays, accel.hwtlas.hw_accel, n_buf)
    Tri = eltype(eltype(accel.hwtlas.blas_triangles))
    empty = Raycore.empty_triangle(Tri)
    return PrecomputedHitsAccel(results, accel.hwtlas.tri_gpu, accel.hwtlas.off_gpu, empty)
end

function set_custom_anyhit!(accel::HWAdaptedAccel, anyhit_fn, raygen_fn)
    hw = accel.hwtlas.hw_accel
    hw === nothing && error("HWTLAS not synced")
    set_anyhit_pipeline!(hw, anyhit_fn, raygen_fn)
end

# RT shader intrinsics forwarded from HWAdaptedAccel -> Lava intrinsics
rt_primitive_id(::HWAdaptedAccel)          = lava_rt_primitive_id()
rt_instance_custom_index(::HWAdaptedAccel) = lava_rt_instance_custom_index()
rt_instance_id(::HWAdaptedAccel)           = lava_rt_instance_id()
rt_launch_id_x(::HWAdaptedAccel)           = lava_rt_launch_id_x()
rt_global_invocation_id_x(::HWAdaptedAccel) = lava_global_invocation_id_x()
rt_ignore_intersection(::HWAdaptedAccel)   = lava_rt_ignore_intersection()
rt_terminate_ray(::HWAdaptedAccel)         = lava_rt_terminate_ray()
rt_payload_store!(::HWAdaptedAccel, val, slot) = lava_rt_payload_store_f32_at(val, slot)
rt_payload_load(::HWAdaptedAccel, slot)    = lava_rt_payload_load_f32_at(slot)

function rt_trace_ray!(::HWAdaptedAccel, flags, mask, sbt_offset, sbt_stride, miss_idx,
                       ox, oy, oz, tmin, dx, dy, dz, tmax)
    lava_rt_trace_ray(flags, mask, sbt_offset, sbt_stride, miss_idx,
                       ox, oy, oz, tmin, dx, dy, dz, tmax)
end

@propagate_inbounds function Raycore.closest_hit(accel::PrecomputedHitsAccel, ray)
    tid = lava_global_invocation_id_x() + UInt32(1)
    result = accel.results[tid]

    if result.hit == UInt32(0)
        return (false, accel.empty, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end

    # `accel.offsets` is keyed by gl_InstanceID (0-based).  `result.instance_id`
    # is that builtin; `result.instance_custom_index` carries the interface
    # override and is forwarded as the 5th return value.
    tri_idx = Int(accel.offsets[result.instance_id + UInt32(1)]) +
              Int(result.primitive_id) + 1
    tri = accel.triangles[tri_idx]

    w = 1f0 - result.bary_u - result.bary_v
    bary = SVector{3,Float32}(w, result.bary_u, result.bary_v)

    return (true, tri, result.t, bary, result.instance_custom_index)
end
