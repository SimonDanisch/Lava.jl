# SPIR-V Module Builder
# Foundation of the custom LLVM IR → SPIR-V emitter.
# Handles: ID allocation, section buffers, instruction encoding, binary serialization.
#
# SPIR-V binary format: all UInt32 words.
# Instruction: [word_count << 16 | opcode, result_type?, result_id?, operands...]
# Module: Header + ordered sections (Capabilities, Extensions, ..., Functions)

# ---- SPIR-V Constants ----

# Magic number
const SPIRV_MAGIC = 0x07230203

# SPIR-V version 1.4 (required for Vulkan 1.2+)
const SPIRV_VERSION_1_4 = UInt32(0x00010400)

# Generator magic (Lava.jl = 0, no registered ID yet)
const SPIRV_GENERATOR = UInt32(0)

# ---- Opcodes (subset needed for compute, will grow) ----
# Full list: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
module Op
    const OpNop                     = UInt16(0)
    const OpSource                  = UInt16(3)
    const OpName                    = UInt16(5)
    const OpMemberName              = UInt16(6)
    const OpExtInstImport           = UInt16(11)
    const OpExtInst                 = UInt16(12)
    const OpMemoryModel             = UInt16(14)
    const OpEntryPoint              = UInt16(15)
    const OpExecutionMode           = UInt16(16)
    const OpCapability              = UInt16(17)
    const OpTypeVoid                = UInt16(19)
    const OpTypeBool                = UInt16(20)
    const OpTypeInt                 = UInt16(21)
    const OpTypeFloat               = UInt16(22)
    const OpTypeVector              = UInt16(23)
    const OpTypeMatrix              = UInt16(24)
    const OpTypeArray               = UInt16(28)
    const OpTypeRuntimeArray        = UInt16(29)
    const OpTypeStruct              = UInt16(30)
    const OpTypePointer             = UInt16(32)
    const OpTypeFunction            = UInt16(33)
    const OpConstant                = UInt16(43)
    const OpConstantComposite       = UInt16(44)
    const OpFunction                = UInt16(54)
    const OpFunctionParameter       = UInt16(55)
    const OpFunctionEnd             = UInt16(56)
    const OpFunctionCall            = UInt16(57)
    const OpVariable                = UInt16(59)
    const OpLoad                    = UInt16(61)
    const OpStore                   = UInt16(62)
    const OpAccessChain             = UInt16(65)
    const OpDecorate                = UInt16(71)
    const OpMemberDecorate          = UInt16(72)
    const OpConvertFToU             = UInt16(109)
    const OpConvertFToS             = UInt16(110)
    const OpConvertSToF             = UInt16(111)
    const OpConvertUToF             = UInt16(112)
    const OpUConvert                = UInt16(113)
    const OpSConvert                = UInt16(114)
    const OpFConvert                = UInt16(115)
    const OpBitcast                 = UInt16(124)
    const OpSNegate                 = UInt16(126)
    const OpFNegate                 = UInt16(127)
    const OpIAdd                    = UInt16(128)
    const OpFAdd                    = UInt16(129)
    const OpISub                    = UInt16(130)
    const OpFSub                    = UInt16(131)
    const OpIMul                    = UInt16(132)
    const OpFMul                    = UInt16(133)
    const OpUDiv                    = UInt16(134)
    const OpSDiv                    = UInt16(135)
    const OpFDiv                    = UInt16(136)
    const OpUMod                    = UInt16(137)
    const OpSRem                    = UInt16(138)
    const OpFRem                    = UInt16(139)
    const OpLogicalEqual            = UInt16(164)
    const OpLogicalNotEqual         = UInt16(165)
    const OpLogicalOr               = UInt16(166)
    const OpLogicalAnd              = UInt16(167)
    const OpLogicalNot              = UInt16(168)
    const OpSelect                  = UInt16(169)
    const OpIEqual                  = UInt16(170)
    const OpINotEqual               = UInt16(171)
    const OpUGreaterThan            = UInt16(172)
    const OpSGreaterThan            = UInt16(173)
    const OpUGreaterThanEqual       = UInt16(174)
    const OpSGreaterThanEqual       = UInt16(175)
    const OpULessThan               = UInt16(176)
    const OpSLessThan               = UInt16(177)
    const OpULessThanEqual          = UInt16(178)
    const OpSLessThanEqual          = UInt16(179)
    const OpFOrdEqual               = UInt16(180)
    const OpFUnordEqual             = UInt16(181)
    const OpFOrdNotEqual            = UInt16(182)
    const OpFUnordNotEqual          = UInt16(183)
    const OpFOrdLessThan            = UInt16(184)
    const OpFUnordLessThan          = UInt16(185)
    const OpFOrdGreaterThan         = UInt16(186)
    const OpFUnordGreaterThan       = UInt16(187)
    const OpFOrdLessThanEqual       = UInt16(188)
    const OpFUnordLessThanEqual     = UInt16(189)
    const OpFOrdGreaterThanEqual    = UInt16(190)
    const OpFUnordGreaterThanEqual  = UInt16(191)
    const OpPhi                     = UInt16(245)
    const OpLoopMerge               = UInt16(246)
    const OpSelectionMerge          = UInt16(247)
    const OpLabel                   = UInt16(248)
    const OpBranch                  = UInt16(249)
    const OpBranchConditional       = UInt16(250)
    const OpReturn                  = UInt16(253)
    const OpReturnValue             = UInt16(254)
    const OpBitwiseOr               = UInt16(197)
    const OpBitwiseXor              = UInt16(198)
    const OpBitwiseAnd              = UInt16(199)
    const OpNot                     = UInt16(200)
    const OpBitFieldInsert          = UInt16(201)
    const OpBitFieldSExtract        = UInt16(202)
    const OpBitFieldUExtract        = UInt16(203)
    const OpBitReverse              = UInt16(204)
    const OpBitCount                = UInt16(205)
    const OpShiftRightLogical       = UInt16(194)
    const OpShiftRightArithmetic    = UInt16(195)
    const OpShiftLeftLogical        = UInt16(196)
    const OpDPdx                    = UInt16(207)
    const OpDPdy                    = UInt16(208)
    const OpFwidth                  = UInt16(209)
    const OpControlBarrier          = UInt16(224)
    const OpMemoryBarrier           = UInt16(225)
    const OpAtomicLoad              = UInt16(227)
    const OpAtomicStore             = UInt16(228)
    const OpAtomicExchange          = UInt16(229)
    const OpAtomicCompareExchange   = UInt16(230)
    const OpAtomicIIncrement        = UInt16(232)
    const OpAtomicIDecrement        = UInt16(233)
    const OpAtomicIAdd              = UInt16(234)
    const OpAtomicISub              = UInt16(235)
    const OpAtomicSMin              = UInt16(236)
    const OpAtomicUMin              = UInt16(237)
    const OpAtomicSMax              = UInt16(238)
    const OpAtomicUMax              = UInt16(239)
    const OpAtomicAnd               = UInt16(240)
    const OpAtomicOr                = UInt16(241)
    const OpAtomicXor               = UInt16(242)
    const OpConvertPtrToU           = UInt16(117)
    const OpConvertUToPtr           = UInt16(120)
    const OpCompositeExtract        = UInt16(81)
    const OpVectorExtractDynamic    = UInt16(77)
    const OpCompositeConstruct      = UInt16(80)
    # Ray tracing (SPV_KHR_ray_tracing)
    const OpTraceRayKHR             = UInt16(4445)
    const OpExecuteCallableKHR      = UInt16(4446)
    const OpIgnoreIntersectionKHR   = UInt16(4448)
    const OpTerminateRayKHR         = UInt16(4449)
    const OpReportIntersectionKHR   = UInt16(5334)
    const OpTypeAccelerationStructureKHR = UInt16(5341)
    const OpConstantNull            = UInt16(46)
    # Image/sampler instructions
    const OpTypeImage               = UInt16(25)
    const OpTypeSampler             = UInt16(26)
    const OpTypeSampledImage        = UInt16(27)
    const OpImageSampleImplicitLod  = UInt16(87)
    const OpImageSampleExplicitLod  = UInt16(88)
    # Geometry shader
    const OpEmitVertex              = UInt16(218)
    const OpEndPrimitive            = UInt16(219)
end

# ---- Capabilities ----
module Cap
    const Shader                        = UInt32(1)
    const Float16                       = UInt32(9)
    const Float64                       = UInt32(10)
    const Int8                          = UInt32(39)
    const Int64                         = UInt32(11)
    const Int16                         = UInt32(22)
    const StorageBuffer16BitAccess      = UInt32(4433)
    const VulkanMemoryModel             = UInt32(5345)
    const PhysicalStorageBufferAddresses = UInt32(5347)
    const VariablePointers              = UInt32(4442)
    const VariablePointersStorageBuffer = UInt32(4441)
    const Geometry                  = UInt32(2)
    const Tessellation              = UInt32(3)
    const InputAttachment           = UInt32(40)
    const RayTracingKHR             = UInt32(4479)
    const WorkgroupMemoryExplicitLayoutKHR = UInt32(4428)
    const WorkgroupMemoryExplicitLayout8BitAccessKHR = UInt32(4429)
    const WorkgroupMemoryExplicitLayout16BitAccessKHR = UInt32(4430)
end

# ---- Storage Classes ----
module SC
    const UniformConstant       = UInt32(0)
    const Input                 = UInt32(1)
    const Uniform               = UInt32(2)
    const Output                = UInt32(3)
    const Workgroup             = UInt32(4)
    const Private               = UInt32(6)
    const Function              = UInt32(7)
    const PushConstant          = UInt32(9)
    const StorageBuffer         = UInt32(12)
    const PhysicalStorageBuffer = UInt32(5349)
    # Ray tracing storage classes
    const RayPayloadKHR         = UInt32(5338)
    const HitAttributeKHR       = UInt32(5339)
    const IncomingRayPayloadKHR = UInt32(5342)
    const CallableDataKHR       = UInt32(5328)
    const IncomingCallableDataKHR = UInt32(5329)
    const ShaderRecordBufferKHR = UInt32(5343)
end

# ---- Decorations ----
module Dec
    const Block             = UInt32(2)
    const RowMajor          = UInt32(4)
    const ColMajor          = UInt32(5)
    const ArrayStride       = UInt32(6)
    const MatrixStride      = UInt32(7)
    const BuiltIn           = UInt32(11)
    const NoPerspective     = UInt32(13)
    const Flat              = UInt32(14)
    const Patch             = UInt32(15)
    const NonWritable       = UInt32(24)
    const NonReadable       = UInt32(25)
    const Location          = UInt32(30)
    const Binding           = UInt32(33)
    const DescriptorSet     = UInt32(34)
    const Offset            = UInt32(35)
    const Restrict          = UInt32(42)
    const Aliased           = UInt32(43)
end

# ---- Built-in Variables ----
module BuiltIn
    const Position                  = UInt32(0)
    const PointSize                 = UInt32(1)
    const VertexIndex               = UInt32(42)
    const InstanceIndex             = UInt32(43)
    const FragCoord                 = UInt32(15)
    const NumWorkgroups             = UInt32(24)
    const WorkgroupSize             = UInt32(25)
    const WorkgroupId               = UInt32(26)
    const LocalInvocationId         = UInt32(27)
    const GlobalInvocationId        = UInt32(28)
    const LocalInvocationIndex      = UInt32(29)
    const SubgroupSize              = UInt32(36)
    const SubgroupLocalInvocationId = UInt32(41)
    # Ray tracing built-ins
    const LaunchIdKHR               = UInt32(5319)
    const LaunchSizeKHR             = UInt32(5320)
    const WorldRayOriginKHR         = UInt32(5321)
    const WorldRayDirectionKHR      = UInt32(5322)
    const ObjectRayOriginKHR        = UInt32(5323)
    const ObjectRayDirectionKHR     = UInt32(5324)
    const RayTminKHR                = UInt32(5325)
    const RayTmaxKHR                = UInt32(5326)
    const IncomingRayFlagsKHR       = UInt32(5328)
    const HitKindKHR                = UInt32(5333)
    const InstanceCustomIndexKHR    = UInt32(5327)
    const ObjectToWorldKHR          = UInt32(5330)
    const WorldToObjectKHR          = UInt32(5331)
    const ClipDistance              = UInt32(3)
    const CullDistance              = UInt32(4)
    const InstanceId                = UInt32(6)
    const PrimitiveId               = UInt32(7)
    const InvocationId              = UInt32(8)
    const Layer                     = UInt32(9)
    const ViewportIndex             = UInt32(10)
    const TessLevelOuter            = UInt32(11)
    const TessLevelInner            = UInt32(12)
    const TessCoord                 = UInt32(13)
    const PatchVertices             = UInt32(14)
    const FrontFacing               = UInt32(17)
    const SampleId                  = UInt32(18)
    const SamplePosition            = UInt32(19)
    const SampleMask                = UInt32(20)
end

# ---- Execution Models ----
module ExecModel
    const Vertex                    = UInt32(0)
    const TessellationControl       = UInt32(1)
    const TessellationEvaluation    = UInt32(2)
    const Geometry                  = UInt32(3)
    const Fragment                  = UInt32(4)
    const GLCompute                 = UInt32(5)
    const RayGenerationKHR          = UInt32(5313)
    const IntersectionKHR           = UInt32(5314)
    const AnyHitKHR                 = UInt32(5315)
    const ClosestHitKHR             = UInt32(5316)
    const MissKHR                   = UInt32(5317)
    const CallableKHR               = UInt32(5318)
end

# ---- Execution Modes ----
module ExecMode
    const Invocations           = UInt32(0)
    const SpacingEqual          = UInt32(1)
    const SpacingFractionalEven = UInt32(2)
    const SpacingFractionalOdd  = UInt32(3)
    const VertexOrderCw         = UInt32(4)
    const VertexOrderCcw        = UInt32(5)
    const OriginUpperLeft       = UInt32(7)
    const OriginLowerLeft       = UInt32(8)
    const DepthReplacing        = UInt32(12)
    const PointMode             = UInt32(10)
    const LocalSize             = UInt32(17)
    const InputPoints           = UInt32(19)
    const InputLines            = UInt32(20)
    const InputLinesAdjacency   = UInt32(21)
    const Triangles             = UInt32(22)
    const InputTrianglesAdjacency = UInt32(23)
    const Quads                 = UInt32(24)
    const Isolines              = UInt32(25)
    const OutputVertices        = UInt32(26)
    const OutputPoints          = UInt32(27)
    const OutputLineStrip       = UInt32(28)
    const OutputTriangleStrip   = UInt32(29)
end

# ---- Addressing / Memory Models ----
module AddrModel
    const Logical                   = UInt32(0)
    const Physical32                = UInt32(1)
    const Physical64                = UInt32(2)
    const PhysicalStorageBuffer64   = UInt32(5348)
end

module MemModel
    const GLSL450   = UInt32(1)
    const Vulkan    = UInt32(3)
end

# ---- Scopes (for atomics) ----
module Scope
    const CrossDevice   = UInt32(0)
    const Device        = UInt32(1)
    const Workgroup     = UInt32(2)
    const Subgroup      = UInt32(3)
    const Invocation    = UInt32(4)
    const QueueFamily   = UInt32(5)
end

# ---- Memory Semantics (for atomics) ----
module MemSem
    const Relaxed           = UInt32(0x0)
    const Acquire           = UInt32(0x2)
    const Release           = UInt32(0x4)
    const AcquireRelease    = UInt32(0x8)
    const UniformMemory     = UInt32(0x40)     # SSBO/StorageBuffer/PhysicalStorageBuffer
    const WorkgroupMemory   = UInt32(0x100)    # Shared memory
    const ImageMemory       = UInt32(0x800)    # Image/texture
    const MakeAvailableKHR  = UInt32(0x2000)   # Vulkan memory model: make writes available
    const MakeVisibleKHR    = UInt32(0x4000)   # Vulkan memory model: make writes visible
end

# ---- Function Control ----
module FuncControl
    const None      = UInt32(0)
    const Inline    = UInt32(1)
    const DontInline = UInt32(2)
end

# ================================================================
# SPIR-V Module Builder
# ================================================================

"""
    SPIRVModule

Builder for SPIR-V binary modules. Manages ID allocation, section buffers,
type/constant deduplication, and binary serialization.

Sections are emitted in SPIR-V-required order:
1. Capabilities
2. Extensions
3. ExtInstImport
4. MemoryModel
5. EntryPoints
6. ExecutionModes
7. Debug (OpName, OpMemberName)
8. Annotations (OpDecorate, OpMemberDecorate)
9. Type declarations, Constants, Global variables
10. Function declarations
11. Function definitions
"""
mutable struct SPIRVModule
    # ID management
    next_id::UInt32

    # Section buffers (Vector{UInt32} words)
    capabilities::Vector{UInt32}
    extensions::Vector{UInt32}
    ext_inst_imports::Vector{UInt32}
    memory_model::Vector{UInt32}
    entry_points::Vector{UInt32}
    execution_modes::Vector{UInt32}
    debug::Vector{UInt32}
    annotations::Vector{UInt32}
    types_constants::Vector{UInt32}
    global_vars::Vector{UInt32}
    functions::Vector{UInt32}

    # Deduplication caches
    type_cache::Dict{Any, UInt32}       # type descriptor → SPIR-V ID
    constant_cache::Dict{Any, UInt32}   # (type_id, value) → SPIR-V ID

    # ExtInstImport IDs
    glsl_std_450_id::UInt32             # GLSL.std.450 extended instruction set

    # Capability tracking (avoid duplicates)
    declared_capabilities::Set{UInt32}
    declared_extensions::Set{String}

    # Source mapping: SPIR-V result ID → (julia_file, julia_line)
    # Populated during emission by record_source_location!()
    source_locations::Dict{UInt32, Tuple{String, Int}}
end

function SPIRVModule()
    mod = SPIRVModule(
        UInt32(1),  # IDs start at 1
        UInt32[], UInt32[], UInt32[], UInt32[],
        UInt32[], UInt32[], UInt32[], UInt32[],
        UInt32[], UInt32[], UInt32[],
        Dict{Any, UInt32}(),
        Dict{Any, UInt32}(),
        UInt32(0),
        Set{UInt32}(),
        Set{String}(),
        Dict{UInt32, Tuple{String, Int}}(),
    )
    return mod
end

"""
    fresh_id!(mod::SPIRVModule) -> UInt32

Allocate and return a fresh SPIR-V ID.
"""
function fresh_id!(mod::SPIRVModule)
    id = mod.next_id
    mod.next_id += UInt32(1)
    return id
end

# ---- Instruction encoding ----

"""
    encode_instruction!(buf::Vector{UInt32}, opcode::UInt16, operands::UInt32...)

Encode a SPIR-V instruction: [word_count << 16 | opcode, operands...]
"""
function encode_instruction!(buf::Vector{UInt32}, opcode::UInt16, operands...)
    word_count = UInt32(1 + length(operands))
    push!(buf, (word_count << 16) | UInt32(opcode))
    for op in operands
        push!(buf, UInt32(op))
    end
    return nothing
end

"""
    encode_string!(buf::Vector{UInt32}, s::String)

Encode a SPIR-V literal string (null-terminated, padded to UInt32 boundary).
Returns the words pushed (not including the initial instruction word).
"""
function encode_string_words!(buf::Vector{UInt32}, s::String)
    bytes = Vector{UInt8}(s)
    push!(bytes, 0x00)  # null terminator
    # Pad to 4-byte boundary
    while length(bytes) % 4 != 0
        push!(bytes, 0x00)
    end
    for i in 1:4:length(bytes)
        word = UInt32(bytes[i]) |
               (UInt32(bytes[min(i+1, length(bytes))]) << 8) |
               (UInt32(bytes[min(i+2, length(bytes))]) << 16) |
               (UInt32(bytes[min(i+3, length(bytes))]) << 24)
        push!(buf, word)
    end
    return length(bytes) ÷ 4
end

"""
    emit_name!(mod::SPIRVModule, id::UInt32, name::String)

Emit OpName for debug info.
"""
function emit_name!(mod::SPIRVModule, id::UInt32, name::String)
    start_len = length(mod.debug)
    push!(mod.debug, UInt32(0))  # placeholder for instruction word
    push!(mod.debug, id)
    nwords = encode_string_words!(mod.debug, name)
    total_words = UInt32(2 + nwords)
    mod.debug[start_len + 1] = (total_words << 16) | UInt32(Op.OpName)
    return nothing
end

# ---- Capability / Extension management ----

"""
    require_capability!(mod::SPIRVModule, cap::UInt32)

Declare a capability (deduplicated).
"""
function require_capability!(mod::SPIRVModule, cap::UInt32)
    if cap ∉ mod.declared_capabilities
        push!(mod.declared_capabilities, cap)
        encode_instruction!(mod.capabilities, Op.OpCapability, cap)
    end
    return nothing
end

"""
    require_extension!(mod::SPIRVModule, ext::String)

Declare an extension (deduplicated).
"""
function require_extension!(mod::SPIRVModule, ext::String)
    if ext ∉ mod.declared_extensions
        push!(mod.declared_extensions, ext)
        start_len = length(mod.extensions)
        push!(mod.extensions, UInt32(0))  # placeholder
        nwords = encode_string_words!(mod.extensions, ext)
        total_words = UInt32(1 + nwords)
        mod.extensions[start_len + 1] = (total_words << 16) | UInt32(Op.OpExtInstImport)
        # Fix: this should use a different opcode for OpExtension
        # OpExtension is opcode 10
        mod.extensions[start_len + 1] = (total_words << 16) | UInt32(10)
    end
    return nothing
end

"""
    setup_glsl_std_450!(mod::SPIRVModule)

Import GLSL.std.450 extended instruction set. Call once during module setup.
"""
function setup_glsl_std_450!(mod::SPIRVModule)
    if mod.glsl_std_450_id == 0
        mod.glsl_std_450_id = fresh_id!(mod)
        start_len = length(mod.ext_inst_imports)
        push!(mod.ext_inst_imports, UInt32(0))  # placeholder
        push!(mod.ext_inst_imports, mod.glsl_std_450_id)
        nwords = encode_string_words!(mod.ext_inst_imports, "GLSL.std.450")
        total_words = UInt32(2 + nwords)
        mod.ext_inst_imports[start_len + 1] = (total_words << 16) | UInt32(Op.OpExtInstImport)
    end
    return mod.glsl_std_450_id
end

# ---- Memory model ----

"""
    setup_memory_model!(mod::SPIRVModule; physical_storage_buffer=true)

Emit OpMemoryModel. For Vulkan with BDA: PhysicalStorageBuffer64 + Vulkan memory model.
"""
function setup_memory_model!(mod::SPIRVModule; physical_storage_buffer=true)
    addr = physical_storage_buffer ? AddrModel.PhysicalStorageBuffer64 : AddrModel.Logical
    encode_instruction!(mod.memory_model, UInt16(14), addr, MemModel.Vulkan)
    require_capability!(mod, Cap.VulkanMemoryModel)
    require_extension!(mod, "SPV_KHR_vulkan_memory_model")
    if physical_storage_buffer
        require_capability!(mod, Cap.PhysicalStorageBufferAddresses)
        require_extension!(mod, "SPV_KHR_physical_storage_buffer")
    end
    return nothing
end

# ---- Type emission (deduplicated) ----

function emit_type_void!(mod::SPIRVModule)
    key = :void
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeVoid, id)
        id
    end
end

function emit_type_bool!(mod::SPIRVModule)
    key = :bool
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeBool, id)
        id
    end
end

function emit_type_int!(mod::SPIRVModule, width::UInt32, signedness::UInt32)
    key = (:int, width, signedness)
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeInt, id, width, signedness)
        if width == 64
            require_capability!(mod, Cap.Int64)
        elseif width == 16
            require_capability!(mod, Cap.Int16)
        elseif width == 8
            require_capability!(mod, Cap.Int8)
        end
        id
    end
end

function emit_type_float!(mod::SPIRVModule, width::UInt32)
    key = (:float, width)
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeFloat, id, width)
        if width == 64
            require_capability!(mod, Cap.Float64)
        elseif width == 16
            require_capability!(mod, Cap.Float16)
        end
        id
    end
end

function emit_type_vector!(mod::SPIRVModule, component_type_id::UInt32, count::UInt32)
    key = (:vector, component_type_id, count)
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeVector, id, component_type_id, count)
        id
    end
end

function emit_type_array!(mod::SPIRVModule, element_type_id::UInt32, length_id::UInt32)
    key = (:array, element_type_id, length_id)
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeArray, id, element_type_id, length_id)
        id
    end
end

function emit_type_struct!(mod::SPIRVModule, member_type_ids::Vector{UInt32})
    key = (:struct, Tuple(member_type_ids))
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        word_count = UInt32(2 + length(member_type_ids))
        push!(mod.types_constants, (word_count << 16) | UInt32(Op.OpTypeStruct))
        push!(mod.types_constants, id)
        append!(mod.types_constants, member_type_ids)
        id
    end
end

function emit_type_pointer!(mod::SPIRVModule, storage_class::UInt32, pointee_type_id::UInt32)
    key = (:pointer, storage_class, pointee_type_id)
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypePointer, id, storage_class, pointee_type_id)
        id
    end
end

function emit_type_acceleration_structure!(mod::SPIRVModule)
    key = :acceleration_structure
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpTypeAccelerationStructureKHR, id)
        require_capability!(mod, Cap.RayTracingKHR)
        id
    end
end

function emit_type_function!(mod::SPIRVModule, return_type_id::UInt32, param_type_ids::Vector{UInt32}=UInt32[])
    key = (:function, return_type_id, Tuple(param_type_ids))
    get!(mod.type_cache, key) do
        id = fresh_id!(mod)
        word_count = UInt32(3 + length(param_type_ids))
        push!(mod.types_constants, (word_count << 16) | UInt32(Op.OpTypeFunction))
        push!(mod.types_constants, id)
        push!(mod.types_constants, return_type_id)
        append!(mod.types_constants, param_type_ids)
        id
    end
end

# ---- Constants (deduplicated) ----

function emit_constant_u32!(mod::SPIRVModule, value::UInt32)
    type_id = emit_type_int!(mod, UInt32(32), UInt32(0))
    key = (:const, type_id, value)
    get!(mod.constant_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstant, type_id, id, value)
        id
    end
end

function emit_constant_i32!(mod::SPIRVModule, value::Int32)
    type_id = emit_type_int!(mod, UInt32(32), UInt32(1))
    key = (:const, type_id, reinterpret(UInt32, value))
    get!(mod.constant_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstant, type_id, id, reinterpret(UInt32, value))
        id
    end
end

function emit_constant_f32!(mod::SPIRVModule, value::Float32)
    type_id = emit_type_float!(mod, UInt32(32))
    key = (:const, type_id, reinterpret(UInt32, value))
    get!(mod.constant_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstant, type_id, id, reinterpret(UInt32, value))
        id
    end
end

function emit_constant_u64!(mod::SPIRVModule, value::UInt64)
    type_id = emit_type_int!(mod, UInt32(64), UInt32(0))
    lo = UInt32(value & 0xFFFFFFFF)
    hi = UInt32((value >> 32) & 0xFFFFFFFF)
    key = (:const, type_id, lo, hi)
    get!(mod.constant_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstant, type_id, id, lo, hi)
        id
    end
end

function emit_constant_null!(mod::SPIRVModule, type_id::UInt32)
    key = (:const_null, type_id)
    get!(mod.constant_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, Op.OpConstantNull, type_id, id)
        id
    end
end

# ---- Decorations ----

function emit_decorate!(mod::SPIRVModule, target::UInt32, decoration::UInt32, operands::UInt32...)
    word_count = UInt32(3 + length(operands))
    push!(mod.annotations, (word_count << 16) | UInt32(Op.OpDecorate))
    push!(mod.annotations, target)
    push!(mod.annotations, decoration)
    append!(mod.annotations, operands)
    return nothing
end

function emit_member_decorate!(mod::SPIRVModule, struct_type::UInt32, member::UInt32, decoration::UInt32, operands::UInt32...)
    word_count = UInt32(4 + length(operands))
    push!(mod.annotations, (word_count << 16) | UInt32(Op.OpMemberDecorate))
    push!(mod.annotations, struct_type)
    push!(mod.annotations, member)
    push!(mod.annotations, decoration)
    append!(mod.annotations, operands)
    return nothing
end

# ---- Entry point ----

function emit_entry_point!(mod::SPIRVModule, exec_model::UInt32, func_id::UInt32, name::String, interface_ids::Vector{UInt32}=UInt32[])
    start_len = length(mod.entry_points)
    push!(mod.entry_points, UInt32(0))  # placeholder
    push!(mod.entry_points, exec_model)
    push!(mod.entry_points, func_id)
    nwords = encode_string_words!(mod.entry_points, name)
    append!(mod.entry_points, interface_ids)
    total_words = UInt32(3 + nwords + length(interface_ids))
    mod.entry_points[start_len + 1] = (total_words << 16) | UInt32(Op.OpEntryPoint)
    return nothing
end

function emit_execution_mode!(mod::SPIRVModule, entry_point::UInt32, mode::UInt32, operands::UInt32...)
    word_count = UInt32(3 + length(operands))
    push!(mod.execution_modes, (word_count << 16) | UInt32(Op.OpExecutionMode))
    push!(mod.execution_modes, entry_point)
    push!(mod.execution_modes, mode)
    append!(mod.execution_modes, operands)
    return nothing
end

# ---- Global variables ----

function emit_global_variable!(mod::SPIRVModule, type_id::UInt32, storage_class::UInt32; initializer::Union{Nothing, UInt32}=nothing)
    id = fresh_id!(mod)
    if initializer === nothing
        encode_instruction!(mod.global_vars, Op.OpVariable, type_id, id, storage_class)
    else
        encode_instruction!(mod.global_vars, Op.OpVariable, type_id, id, storage_class, initializer)
    end
    return id
end

# ================================================================
# Binary serialization
# ================================================================

"""
    serialize(mod::SPIRVModule) -> Vector{UInt8}

Serialize the SPIR-V module to a binary blob ready for spirv-val or VkShaderModule.
"""
function serialize(mod::SPIRVModule)
    # Compute total words
    total_words = 5  # header
    for section in (mod.capabilities, mod.extensions, mod.ext_inst_imports,
                    mod.memory_model, mod.entry_points, mod.execution_modes,
                    mod.debug, mod.annotations, mod.types_constants,
                    mod.global_vars, mod.functions)
        total_words += length(section)
    end

    words = Vector{UInt32}(undef, total_words)
    idx = 1

    # Header
    words[idx] = SPIRV_MAGIC; idx += 1
    words[idx] = SPIRV_VERSION_1_4; idx += 1
    words[idx] = SPIRV_GENERATOR; idx += 1
    words[idx] = mod.next_id; idx += 1  # bound
    words[idx] = UInt32(0); idx += 1    # reserved

    # Sections in order
    for section in (mod.capabilities, mod.extensions, mod.ext_inst_imports,
                    mod.memory_model, mod.entry_points, mod.execution_modes,
                    mod.debug, mod.annotations, mod.types_constants,
                    mod.global_vars, mod.functions)
        copyto!(words, idx, section, 1, length(section))
        idx += length(section)
    end

    return collect(reinterpret(UInt8, words))
end
