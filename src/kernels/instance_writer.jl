# GPU compute kernels for writing TLAS instance records from grain state.
#
# write_grain_instances_kernel: one thread per grain, writes 2 instance records
#   (physics @ index 2i-1 with mask=0x02, rendering @ index 2i with mask=0x04).
#   Both records reference different BLASes via the supplied device addresses.
#
# All math is GPU-side; positions/quats live in LavaArrays and never touch CPU.

"""
    quat_to_rot3x3(q::Vec4f) -> NTuple{9, Float32}

Convert a unit quaternion (x, y, z, w) to a row-major 3x3 rotation matrix
returned as a 9-tuple (m00, m01, m02, m10, m11, m12, m20, m21, m22).
"""
@inline function quat_to_rot3x3(q::Vec4f)
    x, y, z, w = q[1], q[2], q[3], q[4]
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    return (
        1f0 - 2f0 * (yy + zz),  2f0 * (xy - wz),       2f0 * (xz + wy),
        2f0 * (xy + wz),         1f0 - 2f0 * (xx + zz), 2f0 * (yz - wx),
        2f0 * (xz - wy),         2f0 * (yz + wx),       1f0 - 2f0 * (xx + yy),
    )
end

"""
    build_4x3(rot9::NTuple{9, Float32}, scale::Float32, p::Point3f) -> NTuple{12, Float32}

Combine a 3x3 rotation, uniform scale, and translation into a row-major 3x4
transform suitable for VkAccelerationStructureInstanceKHR.
"""
@inline function build_4x3(rot9::NTuple{9, Float32}, scale::Float32, p::Point3f)
    (rot9[1] * scale, rot9[2] * scale, rot9[3] * scale, p[1],
     rot9[4] * scale, rot9[5] * scale, rot9[6] * scale, p[2],
     rot9[7] * scale, rot9[8] * scale, rot9[9] * scale, p[3])
end

"""
    write_grain_instances_kernel(positions, quats, radius,
                                  aabb_blas_addr, tri_blas_addr,
                                  instances)

One thread per grain. Reads `positions[i]` and `quats[i]`, writes two
`LavaInstanceRecord`s into `instances`:

  - `instances[2i - 1]` references the AABB BLAS, mask = `0x02` (physics).
  - `instances[2i]`     references the triangle BLAS, mask = `0x04` (rendering).

Both records share the same transform: rotation from `quats[i]`, uniform
scale `radius`, translation `positions[i]`. Custom index = `i - 1`.
"""
@kernel function write_grain_instances_kernel(
        @Const(positions),
        @Const(quats),
        radius::Float32,
        aabb_blas_addr::UInt64,
        tri_blas_addr::UInt64,
        instances)
    i = @index(Global)
    @inbounds p = positions[i]
    @inbounds q = quats[i]
    rot9 = quat_to_rot3x3(q)
    T = build_4x3(rot9, radius, p)
    cidx = UInt32(i - 1)
    # Pack custom_index_and_mask: low 24 bits = cidx, high 8 bits = mask.
    cim_phys = (cidx & 0x00FFFFFF) | (UInt32(0x02) << 24)
    cim_rend = (cidx & 0x00FFFFFF) | (UInt32(0x04) << 24)
    sof = UInt32(0)  # sbt_offset = 0, flags = 0
    @inbounds instances[2i - 1] = LavaInstanceRecord(T, cim_phys, sof, aabb_blas_addr)
    @inbounds instances[2i]     = LavaInstanceRecord(T, cim_rend, sof, tri_blas_addr)
end

"""
    write_meshscatter_instances_kernel(positions, rotations, scale, blas_addr,
                                        instance_mask, instances)

One thread per instance. Reads `positions[i]` and `rotations[i]` (as a
unit quaternion `Vec4f(x, y, z, w)`), writes ONE `LavaInstanceRecord` into
`instances[i]` referencing `blas_addr` with the supplied `instance_mask` and
uniform scale.

Used by RayMakie's meshscatter recipe for GPU-resident positions/rotations.
For the dual-record (physics + render) demo case, use
`write_grain_instances_kernel` instead.
"""
@kernel function write_meshscatter_instances_kernel(
        @Const(positions),
        @Const(rotations),
        scale::Float32,
        blas_addr::UInt64,
        instance_mask::UInt8,
        instances)
    i = @index(Global)
    @inbounds p = positions[i]
    @inbounds q = rotations[i]
    rot9 = quat_to_rot3x3(q)
    T = build_4x3(rot9, scale, p)
    cidx = UInt32(i - 1)
    cim = (cidx & 0x00FFFFFF) | (UInt32(instance_mask) << 24)
    sof = UInt32(0)
    @inbounds instances[i] = LavaInstanceRecord(T, cim, sof, blas_addr)
end
