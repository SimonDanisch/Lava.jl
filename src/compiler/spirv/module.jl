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
    const OpString                  = UInt16(7)
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
    const OpInBoundsAccessChain     = UInt16(66)
    const OpPtrAccessChain          = UInt16(67)
    const OpInBoundsPtrAccessChain  = UInt16(70)
    const OpDecorate                = UInt16(71)
    const OpMemberDecorate          = UInt16(72)
    const OpConvertFToU             = UInt16(109)
    const OpConvertFToS             = UInt16(110)
    const OpConvertSToF             = UInt16(111)
    const OpConvertUToF             = UInt16(112)
    const OpUConvert                = UInt16(113)
    const OpSConvert                = UInt16(114)
    const OpFConvert                = UInt16(115)
    # Use-CHANGING cooperative-matrix conversion (`SPV_NV_cooperative_matrix2`).
    # `OpFConvert` is only legal between matrix types agreeing on scope, rows,
    # cols AND use; turning an fp32 Accumulator into an fp16 MatrixA — which is
    # what `flash_attn_cm2.comp` does twice, for `Qf16` and `P_A` — is this one.
    const OpCooperativeMatrixConvertNV = UInt16(5293)
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
    # Float atomics (SPV_EXT_shader_atomic_float_add / _min_max)
    const OpAtomicFAddEXT           = UInt16(6035)
    const OpAtomicFMinEXT           = UInt16(5614)
    const OpAtomicFMaxEXT           = UInt16(5615)
    # Subgroup / group-non-uniform ops (Vulkan 1.1 core)
    const OpGroupNonUniformElect                = UInt16(333)
    const OpGroupNonUniformAll                  = UInt16(334)
    const OpGroupNonUniformAny                  = UInt16(335)
    const OpGroupNonUniformBroadcast            = UInt16(337)
    const OpGroupNonUniformBroadcastFirst       = UInt16(338)
    const OpGroupNonUniformBallot               = UInt16(339)
    const OpGroupNonUniformShuffle              = UInt16(345)
    const OpGroupNonUniformShuffleXor           = UInt16(346)
    const OpGroupNonUniformShuffleUp            = UInt16(347)
    const OpGroupNonUniformShuffleDown          = UInt16(348)
    # SPV_KHR_subgroup_rotate: shuffle by a *delta*, where the source lane wraps
    # inside the subgroup (or inside a cluster). ShuffleUp/Down clamp instead of
    # wrapping, so a butterfly reduction needs this one or ShuffleXor.
    const OpGroupNonUniformRotateKHR            = UInt16(4431)
    const OpGroupNonUniformIAdd                 = UInt16(349)
    const OpGroupNonUniformFAdd                 = UInt16(350)
    const OpGroupNonUniformIMul                 = UInt16(351)
    const OpGroupNonUniformFMul                 = UInt16(352)
    const OpGroupNonUniformSMin                 = UInt16(353)
    const OpGroupNonUniformUMin                 = UInt16(354)
    const OpGroupNonUniformFMin                 = UInt16(355)
    const OpGroupNonUniformSMax                 = UInt16(356)
    const OpGroupNonUniformUMax                 = UInt16(357)
    const OpGroupNonUniformFMax                 = UInt16(358)
    const OpGroupNonUniformBitwiseAnd           = UInt16(359)
    const OpGroupNonUniformBitwiseOr            = UInt16(360)
    const OpGroupNonUniformBitwiseXor           = UInt16(361)
    const OpGroupNonUniformLogicalAnd           = UInt16(362)
    const OpGroupNonUniformLogicalOr            = UInt16(363)
    const OpGroupNonUniformLogicalXor           = UInt16(364)
    const OpConvertPtrToU           = UInt16(117)
    const OpConvertUToPtr           = UInt16(120)
    const OpCompositeExtract        = UInt16(81)
    const OpVectorExtractDynamic    = UInt16(77)
    const OpCompositeConstruct      = UInt16(80)
    # `emit.jl` has always *used* `Op.OpCompositeInsert` — it was simply never
    # declared here, so every kernel that reached that path died with
    # `UndefVarError: OpCompositeInsert not defined in Lava.Op` instead of
    # compiling. Building a vector value element by element is enough to hit it,
    # which is how it surfaced: `@localmem NTuple{2,VecElement{Float16}}`, the
    # shape a `vec2`-wide GEMM staging buffer needs.
    const OpCompositeInsert         = UInt16(82)
    # Ray tracing (SPV_KHR_ray_tracing)
    const OpTraceRayKHR             = UInt16(4445)
    const OpExecuteCallableKHR      = UInt16(4446)
    const OpIgnoreIntersectionKHR   = UInt16(4448)
    const OpTerminateRayKHR         = UInt16(4449)
    const OpReportIntersectionKHR   = UInt16(5334)
    const OpTypeAccelerationStructureKHR = UInt16(5341)
    const OpConstantNull            = UInt16(46)
    # `OpUndef` where `OpConstantNull` is illegal. An opaque handle type such as
    # `OpTypeTensorLayoutNV` has no null value — spirv-val says so directly,
    # "OpConstantNull Result Type cannot have a null value" — so a phi edge on
    # which a layout is dead needs an undef rather than a null.
    #
    # **1, not 52.** 52 is `OpSpecConstantOp`, and using it produced "End of
    # input reached while decoding OpSpecConstantOp: expected more operands"
    # — a decoder error nowhere near the mistake. Read out of a module that
    # `spirv-as` assembled from text containing an `OpUndef`, which is the same
    # method the tensor opcodes above use and the reason they were right first
    # time. This one was guessed and was not.
    const OpUndef                   = UInt16(1)
    # Bool constants are their own opcodes, not `OpConstant` with a 0/1 operand.
    # Needed by `OpTypeTensorViewNV`, whose `HasDimensions` parameter is a Bool
    # constant <id>. Observed, not assumed: a shader declaring both a
    # `tensorViewNV<2, true, …>` and a `<2, false, …>` emits 41 and 42 — and two
    # DISTINCT view types, which is why `HasDimensions` is part of the type key.
    const OpConstantTrue            = UInt16(41)
    const OpConstantFalse           = UInt16(42)
    # Image/sampler instructions
    const OpTypeImage               = UInt16(25)
    const OpTypeSampler             = UInt16(26)
    const OpTypeSampledImage        = UInt16(27)
    const OpImageSampleImplicitLod  = UInt16(87)
    const OpImageSampleExplicitLod  = UInt16(88)
    # Geometry shader
    const OpEmitVertex              = UInt16(218)
    const OpEndPrimitive            = UInt16(219)
    # SPV_KHR_cooperative_matrix — subgroup-scope matrix multiply-accumulate.
    # The matrix is an opaque type whose value is distributed across the lanes of
    # a subgroup; only these ops may produce or consume one.
    const OpTypeCooperativeMatrixKHR                      = UInt16(4456)
    const OpCooperativeMatrixLoadKHR                      = UInt16(4457)
    const OpCooperativeMatrixStoreKHR                     = UInt16(4458)
    const OpCooperativeMatrixMulAddKHR                    = UInt16(4459)
    const OpCooperativeMatrixLengthKHR                    = UInt16(4460)
    # SPV_NV_cooperative_matrix2. Both take a *function* <id> as an operand:
    #   OpCooperativeMatrixReduceNV     %type %res %matrix <reduce-mask literal> %func
    #   OpCooperativeMatrixPerElementOpNV %type %res %matrix %func [%extra...]
    # Numbers taken from what glslang emits for `coopMatPerElementNV` /
    # `coopMatReduceNV`, not from a remembered table.
    const OpCooperativeMatrixReduceNV                     = UInt16(5366)
    const OpCooperativeMatrixPerElementOpNV               = UInt16(5369)
    # Tensor addressing — the operand-staging half of coopmat2, and it is TWO
    # extensions, not one. The layout/view types and everything that builds them
    # come from `SPV_NV_tensor_addressing` (capability `TensorAddressingNV`);
    # only the load itself is `SPV_NV_cooperative_matrix2` (capability
    # `CooperativeMatrixTensorAddressingNV`). Requiring just the latter produces a
    # module the driver rejects.
    #
    #   %l  = OpCreateTensorLayoutNV                       (no operands)
    #   %l' = OpTensorLayoutSetDimensionNV %l %d0 %d1 …
    #   %s  = OpTensorLayoutSliceNV %l' %off0 %sz0 %off1 %sz1 …   (offset/size PAIRS)
    #   %v  = OpCreateTensorViewNV                         (no operands)
    #   %m  = OpCooperativeMatrixLoadTensorNV %ptr %object %s <memop> <tensorop>
    #
    # Three things the operand list does not advertise, all read off the binary:
    #
    #  * the TYPES take constant <id>s, not literals. `OpTypeTensorLayoutNV` is
    #    `%Dim %ClampMode` and `OpTypeTensorViewNV` is
    #    `%Dim %HasDimensions %p0 %p1 …`, every one of them an `OpConstant`.
    #  * `%object` on the load is **the matrix's existing value**, not a
    #    destination pointer — glslang emits an `OpLoad` of the target matrix and
    #    passes it in. It is what out-of-range elements keep, which is the whole
    #    point under a clamping layout; pass an `OpUndef` only where every element
    #    is known in range.
    #  * `%ptr` is an ordinary pointer (glslang emits `OpAccessChain`), so the
    #    GLSL element offset is folded into the pointer rather than passed on.
    #
    # Numbers read out of a SPIR-V binary that glslang produced for
    # `test/glsl/tensor_addressing_opcodes.comp`, matched to the disassembly by
    # result id — same method as the two above, not a remembered table.
    # `test/test_tensor_opcodes.jl` re-derives them from that shader and fails if
    # any constant here disagrees.
    const OpTypeTensorLayoutNV                            = UInt16(5370)
    const OpTypeTensorViewNV                              = UInt16(5371)
    const OpCreateTensorLayoutNV                          = UInt16(5372)
    const OpTensorLayoutSetDimensionNV                    = UInt16(5373)
    # The STRIDE between successive indices of each dimension, in elements. Without
    # it a layout describes a PACKED tensor, and attention's operands are not: `q`,
    # `k` and `v` arrive as a permuted view of one packed `(E, L, H, B)` block, so
    # the extent alone does not say where the next row is. `mul_mm_cm2.comp` and
    # `flash_attn_cm2.comp` both set it on every layout they build.
    const OpTensorLayoutSetStrideNV                       = UInt16(5374)
    const OpTensorLayoutSliceNV                           = UInt16(5375)
    # What a `TENSOR_CLAMP_CONSTANT` load substitutes OUT OF RANGE. The default
    # is zero, and zero is not always the useful identity: filling `V`'s padding
    # columns with ONE makes a row sum fall out of the `P x V` product's padding
    # columns, which is a per-key-block row reduction removed rather than made
    # cheaper. The operand is a single 32-bit <id> whatever the component type.
    const OpTensorLayoutSetClampValueNV                   = UInt16(5376)
    const OpCreateTensorViewNV                            = UInt16(5377)
    const OpCooperativeMatrixLoadTensorNV                 = UInt16(5367)
    # `TensorAddressingOperands` — the mask that trails a tensor load. Read off
    # `/usr/include/glslang/SPIRV/spirv.hpp11` (`TensorAddressingOperandsMask`)
    # rather than guessed: a wrong bit here does not fail to compile, it loads
    # the wrong elements and still returns finite numbers.
    const TENSOR_ADDR_NONE                                = UInt32(0x0)
    const TENSOR_ADDR_TENSORVIEW                          = UInt32(0x1)
    const TENSOR_ADDR_DECODEFUNC                          = UInt32(0x2)
    # The STORE, which is what makes a ragged OUTPUT legal — the clamping layout
    # bounds-checks writes exactly as it bounds-checks reads, so an edge tile
    # writes only its in-range elements instead of running off the end. Read off
    # a glslang binary for `coopMatStoreTensorNV`, not guessed from the load:
    #
    #   OpCooperativeMatrixStoreTensorNV %ptr %object %layout <memop> <tensorop>
    #
    # NO result type and NO result id — it is a store — so it is the load's
    # operand list minus its first two words. `%object` here IS the matrix being
    # written, the same operand position that on the load carries the value
    # out-of-range elements keep.
    const OpCooperativeMatrixStoreTensorNV                = UInt16(5368)
    # VK_KHR_ray_query opcodes (SPV_KHR_ray_query)
    const OpTypeRayQueryKHR                               = UInt16(4472)
    const OpRayQueryInitializeKHR                         = UInt16(4473)
    const OpRayQueryTerminateKHR                          = UInt16(4474)
    const OpRayQueryGenerateIntersectionKHR               = UInt16(4475)
    const OpRayQueryConfirmIntersectionKHR                = UInt16(4476)
    const OpRayQueryProceedKHR                            = UInt16(4477)
    const OpRayQueryGetIntersectionTypeKHR                = UInt16(4479)
    const OpRayQueryGetIntersectionTKHR                   = UInt16(6018)
    const OpRayQueryGetIntersectionInstanceCustomIndexKHR = UInt16(6019)
    const OpRayQueryGetIntersectionInstanceIdKHR          = UInt16(6020)
    const OpRayQueryGetIntersectionPrimitiveIndexKHR      = UInt16(6023)
    const OpRayQueryGetIntersectionBarycentricsKHR        = UInt16(6024)
    # Intersection-kind operand to GetIntersection* (and related ops)
    const RayQueryCandidateIntersectionKHR = UInt32(0)
    const RayQueryCommittedIntersectionKHR = UInt32(1)
    # ── SER (SPV_NV_shader_invocation_reorder) ──
    # The opaque HitObject type carries the result of a hitObjectTraceRayNV
    # call.  reorderThreadNV uses it (optionally) to coherently rebin warp
    # lanes, then hitObjectExecuteShaderNV invokes the closest-hit / miss
    # shader for the reordered thread.  This is what makes per-material
    # closesthit shaders deliver coherent warps under divergent rays.
    const OpHitObjectRecordHitMotionNV                = UInt16(5249)
    const OpHitObjectRecordHitWithIndexMotionNV       = UInt16(5250)
    const OpHitObjectRecordMissMotionNV               = UInt16(5251)
    const OpHitObjectGetWorldToObjectNV               = UInt16(5252)
    const OpHitObjectGetObjectToWorldNV               = UInt16(5253)
    const OpHitObjectGetObjectRayDirectionNV          = UInt16(5254)
    const OpHitObjectGetObjectRayOriginNV             = UInt16(5255)
    const OpHitObjectTraceRayMotionNV                 = UInt16(5256)
    const OpHitObjectGetShaderRecordBufferHandleNV    = UInt16(5257)
    const OpHitObjectGetShaderBindingTableRecordIndexNV = UInt16(5258)
    const OpHitObjectRecordEmptyNV                    = UInt16(5259)
    const OpHitObjectTraceRayNV                       = UInt16(5260)
    const OpHitObjectRecordHitNV                      = UInt16(5261)
    const OpHitObjectRecordHitWithIndexNV             = UInt16(5262)
    const OpHitObjectRecordMissNV                     = UInt16(5263)
    const OpHitObjectExecuteShaderNV                  = UInt16(5264)
    const OpHitObjectGetCurrentTimeNV                 = UInt16(5265)
    const OpHitObjectGetAttributesNV                  = UInt16(5266)
    const OpHitObjectGetHitKindNV                     = UInt16(5267)
    const OpHitObjectGetPrimitiveIndexNV              = UInt16(5268)
    const OpHitObjectGetGeometryIndexNV               = UInt16(5269)
    const OpHitObjectGetInstanceIdNV                  = UInt16(5270)
    const OpHitObjectGetInstanceCustomIndexNV         = UInt16(5271)
    const OpHitObjectGetWorldRayDirectionNV           = UInt16(5272)
    const OpHitObjectGetWorldRayOriginNV              = UInt16(5273)
    const OpHitObjectGetRayTMaxNV                     = UInt16(5274)
    const OpHitObjectGetRayTMinNV                     = UInt16(5275)
    const OpHitObjectIsEmptyNV                        = UInt16(5276)
    const OpHitObjectIsHitNV                          = UInt16(5277)
    const OpHitObjectIsMissNV                         = UInt16(5278)
    const OpReorderThreadWithHitObjectNV              = UInt16(5279)
    const OpReorderThreadWithHintNV                   = UInt16(5280)
    const OpTypeHitObjectNV                           = UInt16(5281)
end

# ---- Capabilities ----
module Cap
    const Shader                        = UInt32(1)
    const Float16                       = UInt32(9)
    const Float64                       = UInt32(10)
    const Int8                          = UInt32(39)
    const Int64                         = UInt32(11)
    const Int64Atomics                  = UInt32(12)
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
    # Float atomic ops (VK_EXT_shader_atomic_float / SPV_EXT_shader_atomic_float_add)
    const AtomicFloat32AddEXT           = UInt32(6033)
    const AtomicFloat64AddEXT           = UInt32(6034)
    const AtomicFloat16AddEXT           = UInt32(6095)
    # Subgroup ops (Vulkan 1.1 core)
    const GroupNonUniform               = UInt32(61)
    const GroupNonUniformVote           = UInt32(62)
    const GroupNonUniformArithmetic     = UInt32(63)
    const GroupNonUniformBallot         = UInt32(64)
    const GroupNonUniformShuffle        = UInt32(65)
    const GroupNonUniformShuffleRelative = UInt32(66)
    const GroupNonUniformClustered      = UInt32(67)
    const GroupNonUniformQuad           = UInt32(68)
    # SPV_KHR_subgroup_rotate (promoted to Vulkan 1.4). Needs the extension
    # declared as well as the capability.
    const GroupNonUniformRotateKHR      = UInt32(6026)
    # SPV_NV_cooperative_matrix2. Each sub-feature is a separate capability and
    # each has its own VkPhysicalDeviceCooperativeMatrix2FeaturesNV bit; see
    # `CoopMat2Caps` in runtime/device.jl. NVIDIA-only.
    const CooperativeMatrixReductionsNV            = UInt32(5430)
    # Value from /usr/include/glslang/SPIRV/spirv.hpp11, not inferred from the
    # neighbouring 5430 — adjacency is a coincidence, not a rule.
    const CooperativeMatrixConversionsNV           = UInt32(5431)
    const CooperativeMatrixPerElementOperationsNV  = UInt32(5432)
    # Tensor addressing needs BOTH of these, from two different extensions:
    # `TensorAddressingNV` (SPV_NV_tensor_addressing) for the layout/view types
    # and their builders, `CooperativeMatrixTensorAddressingNV`
    # (SPV_NV_cooperative_matrix2) for `OpCooperativeMatrixLoadTensorNV` itself.
    # Read off a glslang-produced binary, same as the opcodes.
    const CooperativeMatrixTensorAddressingNV      = UInt32(5433)
    const TensorAddressingNV                       = UInt32(5439)
    const RayQueryKHR                   = UInt32(4472)
    const CooperativeMatrixKHR          = UInt32(6022)
    # SER (SPV_NV_shader_invocation_reorder).  Value per the SPIR-V unified1
    # grammar (5383).  Earlier guesses (5288 / 5247) collided with
    # ComputeDerivativeGroupQuadsKHR and were invalid respectively.
    const ShaderInvocationReorderNV     = UInt32(5383)
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
    const Aliased           = UInt32(20)
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
    # 37 is SubgroupMaxSize, deliberately absent: spirv-val rejects it under the
    # Kernel (OpenCL) capability, so no Vulkan module may carry it. The value
    # lives host-side in `subgroup_size_control(ctx).max`.
    const NumSubgroups              = UInt32(38)
    const SubgroupId                = UInt32(40)
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

# ---- Memory Operand (on OpLoad / OpStore / OpCooperativeMatrix{Load,Store}KHR) ----
#
# Not the same enum as `MemSem` above, which is what a barrier or an atomic
# carries. This one qualifies a single access.
#
# `NonPrivatePointer` is the one that matters and the one that was missing.
# Lava emits `OpMemoryModel ... Vulkan`, and under the **Vulkan memory model** a
# barrier's `MakeAvailable`/`MakeVisible` apply *only* to accesses tagged
# non-private: an untagged access is by definition invisible to other
# invocations, so a workgroup barrier grants it nothing. Every `@localmem` store
# and load was untagged, which makes a staged kernel's whole premise — write
# shared, barrier, read what another invocation wrote — unenforced. It happened
# to work, until a cooperative-matrix GEMM at a 96-row block read stale shared
# memory and silently lost 4 of 32 k-terms per row.
#
# glslang tags every `shared` and buffer access this way when the Vulkan memory
# model is on, which is why the reference shaders do not hit it.
module MemOp
    const None              = UInt32(0x00)
    const Aligned           = UInt32(0x02)   # followed by a literal alignment
    const NonPrivatePointer = UInt32(0x20)
end

"""
    nonprivate(sc) -> MemOp.NonPrivatePointer | MemOp.None

The `NonPrivatePointer` bit for an access in storage class `sc`.

`Workgroup` is the storage every invocation in a group can see, so an access to
it has to say so; `Function` is per-invocation by construction and tagging it
would claim a sharing that does not exist. Global (`PhysicalStorageBuffer`)
accesses are left alone: they are made visible across *dispatches* by the API's
own barriers, which sit outside the shader memory model entirely.
"""
@inline nonprivate(sc::UInt32) = sc == SC.Workgroup ? MemOp.NonPrivatePointer : MemOp.None

# ---- Function Control ----
module FuncControl
    const None      = UInt32(0)
    const Inline    = UInt32(1)
    const DontInline = UInt32(2)
end

# ---- Group Operation (for OpGroupNonUniform* reductions/scans) ----
module GroupOp
    const Reduce              = UInt32(0)
    const InclusiveScan       = UInt32(1)
    const ExclusiveScan       = UInt32(2)
    const ClusteredReduce     = UInt32(3)
    const PartitionedReduceNV         = UInt32(6)
    const PartitionedInclusiveScanNV  = UInt32(7)
    const PartitionedExclusiveScanNV  = UInt32(8)
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
    setup_debug_printf!(mod::SPIRVModule) -> UInt32

Import the `NonSemantic.DebugPrintf` extended instruction set (and the required
`SPV_KHR_non_semantic_info` extension), returning its import id. Deduplicated via
`type_cache` so repeated printf call sites share one import. The result id is the
`<set>` operand of `OpExtInst … 1 …` (instruction 1 = DebugPrintf).
"""
function setup_debug_printf!(mod::SPIRVModule)
    key = (:debug_printf_extinst_set,)
    cached = get(mod.type_cache, key, UInt32(0))
    cached != 0 && return cached
    require_extension!(mod, "SPV_KHR_non_semantic_info")
    id = fresh_id!(mod)
    start_len = length(mod.ext_inst_imports)
    push!(mod.ext_inst_imports, UInt32(0))  # placeholder
    push!(mod.ext_inst_imports, id)
    nwords = encode_string_words!(mod.ext_inst_imports, "NonSemantic.DebugPrintf")
    total_words = UInt32(2 + nwords)
    mod.ext_inst_imports[start_len + 1] = (total_words << 16) | UInt32(Op.OpExtInstImport)
    mod.type_cache[key] = id
    return id
end

"""
    emit_op_string!(mod::SPIRVModule, s::AbstractString) -> UInt32

Emit `OpString` for `s` and return its id. SPIR-V's logical layout requires all
`OpString`s (debug section 7a) to precede `OpName`s (7b), so we prepend into the
debug buffer — every `OpName` Lava emits stays after them regardless of when the
printf is walked.
"""
function emit_op_string!(mod::SPIRVModule, s::AbstractString)
    id = fresh_id!(mod)
    words = UInt32[UInt32(0), id]  # placeholder + result id
    nwords = encode_string_words!(words, String(s))
    words[1] = (UInt32(2 + nwords) << 16) | UInt32(Op.OpString)
    prepend!(mod.debug, words)
    return id
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

"""
    emit_constant_bool!(mod, value) -> UInt32

`OpConstantTrue`/`OpConstantFalse`, deduplicated. A Bool constant is a distinct
opcode with no operand rather than an `OpConstant` carrying 0 or 1, so it cannot
go through `emit_constant_u32!`.
"""
function emit_constant_bool!(mod::SPIRVModule, value::Bool)
    type_id = emit_type_bool!(mod)
    key = (:constbool, type_id, value)
    get!(mod.constant_cache, key) do
        id = fresh_id!(mod)
        encode_instruction!(mod.types_constants,
                            value ? Op.OpConstantTrue : Op.OpConstantFalse,
                            type_id, id)
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
