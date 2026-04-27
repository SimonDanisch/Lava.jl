# LavaInstanceRecord — a 64-byte plain-bits mirror of VkAccelerationStructureInstanceKHR.
#
# Field layout (matches Vulkan exactly):
#   bytes  0-47   transform: 12 × Float32 (row-major 3×4)
#   bytes 48-50   instanceCustomIndex (24 bits)
#   byte  51      mask (8 bits)
#   bytes 52-54   instanceShaderBindingTableRecordOffset (24 bits)
#   byte  55      flags (8 bits)
#   bytes 56-63   accelerationStructureReference: UInt64 (BLAS device address)
#
# Plain-bits means LavaArray{LavaInstanceRecord, 1} is a clean GPU buffer.
# The two packed UInt32 fields combine the (24 + 8)-bit pairs Vulkan defines.

struct LavaInstanceRecord
    transform::NTuple{12, Float32}
    custom_index_and_mask::UInt32      # 24 bits index | 8 bits mask
    sbt_offset_and_flags::UInt32        # 24 bits sbt   | 8 bits flags
    blas_address::UInt64
end

"""
    LavaInstanceRecord(transform, blas_address;
                       custom_index=UInt32(0), mask=UInt8(0xff),
                       sbt_offset=UInt32(0), flags=UInt8(0)) -> LavaInstanceRecord

Construct a TLAS instance record. `transform` is a row-major 3×4 matrix
packed as `NTuple{12, Float32}`. `mask` is the cull mask the driver tests
against `cullMask` at trace time.
"""
function LavaInstanceRecord(transform::NTuple{12, Float32}, blas_address::UInt64;
                             custom_index::UInt32 = UInt32(0),
                             mask::UInt8 = UInt8(0xff),
                             sbt_offset::UInt32 = UInt32(0),
                             flags::UInt8 = UInt8(0))
    cim = (custom_index & 0x00FFFFFF) | (UInt32(mask) << 24)
    sof = (sbt_offset & 0x00FFFFFF) | (UInt32(flags) << 24)
    return LavaInstanceRecord(transform, cim, sof, blas_address)
end

"""
    identity_transform() -> NTuple{12, Float32}

Row-major 3×4 identity transform. Useful default for instance records that
want pure translation specified later.
"""
identity_transform() = (1f0, 0f0, 0f0, 0f0,
                         0f0, 1f0, 0f0, 0f0,
                         0f0, 0f0, 1f0, 0f0)

# Sanity checks at module load time.
@assert sizeof(LavaInstanceRecord) == 64 "LavaInstanceRecord must be exactly 64 bytes (Vulkan layout); got $(sizeof(LavaInstanceRecord))"
@assert isbitstype(LavaInstanceRecord)   "LavaInstanceRecord must be plain bits for LavaArray storage"
