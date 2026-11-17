# The SPIR-V module builder, used directly: a compute shader assembled
# instruction by instruction (`SPIRVModule`, the `emit_*!` functions,
# `encode_instruction!`) rather than compiled from Julia, then validated with
# spirv-val and disassembled.
#
# Moved here from Mantle's `test/vulkan/test_handwritten_spirv.jl` and
# `test_handwritten_rt.jl`. The compute shader's device half ran the module on a
# Vulkan device of its own through Vulkan.jl and checked output == 2 * input,
# which every kernel test in Mantle checks of its own module through the
# backend's pipeline path. The ray-tracing shaders' device half is Mantle's
# `test/test_rt_pipeline.jl`, with the shaders written in Julia.

using Test, Lava

"""
    build_times2_shader(; local_size=64) -> Vector{UInt8}

Generate SPIR-V binary for a compute shader that doubles every element.
Layout:
  - Binding 0: input StorageBuffer (readonly, runtime array of f32)
  - Binding 1: output StorageBuffer (runtime array of f32)
  - GlobalInvocationId.x used for indexing
"""
function build_times2_shader(; local_size=64)
    mod = Lava.SPIRVModule()

    # Capabilities & memory model
    Lava.require_capability!(mod, Lava.Cap.Shader)
    Lava.setup_memory_model!(mod; physical_storage_buffer=false)

    # Types
    void_t  = Lava.emit_type_void!(mod)
    f32_t   = Lava.emit_type_float!(mod, UInt32(32))
    u32_t   = Lava.emit_type_int!(mod, UInt32(32), UInt32(0))
    uvec3_t = Lava.emit_type_vector!(mod, u32_t, UInt32(3))

    # OpTypeRuntimeArray (opcode 29) — not yet in builder API
    rta_f32_t = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.types_constants, UInt16(29), rta_f32_t, f32_t)
    Lava.emit_decorate!(mod, rta_f32_t, Lava.Dec.ArrayStride, UInt32(4))

    # Input struct { float data[]; } — Block decorated
    input_struct_t = Lava.emit_type_struct!(mod, UInt32[rta_f32_t])
    Lava.emit_decorate!(mod, input_struct_t, Lava.Dec.Block)
    Lava.emit_member_decorate!(mod, input_struct_t, UInt32(0), Lava.Dec.Offset, UInt32(0))

    # Output struct — manually emitted to avoid type deduplication with input struct
    output_struct_t = Lava.fresh_id!(mod)
    push!(mod.types_constants, (UInt32(3) << 16) | UInt32(Lava.Op.OpTypeStruct))
    push!(mod.types_constants, output_struct_t)
    push!(mod.types_constants, rta_f32_t)
    Lava.emit_decorate!(mod, output_struct_t, Lava.Dec.Block)
    Lava.emit_member_decorate!(mod, output_struct_t, UInt32(0), Lava.Dec.Offset, UInt32(0))

    # Pointer types
    ptr_input_t    = Lava.emit_type_pointer!(mod, Lava.SC.StorageBuffer, input_struct_t)
    ptr_output_t   = Lava.emit_type_pointer!(mod, Lava.SC.StorageBuffer, output_struct_t)
    ptr_f32_sb_t   = Lava.emit_type_pointer!(mod, Lava.SC.StorageBuffer, f32_t)
    ptr_uvec3_in_t = Lava.emit_type_pointer!(mod, Lava.SC.Input, uvec3_t)
    func_t         = Lava.emit_type_function!(mod, void_t)

    # Constants
    const_0_u32 = Lava.emit_constant_u32!(mod, UInt32(0))
    const_2_f32 = Lava.emit_constant_f32!(mod, Float32(2.0))

    # Global variables
    input_var = Lava.emit_global_variable!(mod, ptr_input_t, Lava.SC.StorageBuffer)
    Lava.emit_decorate!(mod, input_var, Lava.Dec.DescriptorSet, UInt32(0))
    Lava.emit_decorate!(mod, input_var, Lava.Dec.Binding, UInt32(0))
    Lava.emit_decorate!(mod, input_var, Lava.Dec.NonWritable)

    output_var = Lava.emit_global_variable!(mod, ptr_output_t, Lava.SC.StorageBuffer)
    Lava.emit_decorate!(mod, output_var, Lava.Dec.DescriptorSet, UInt32(0))
    Lava.emit_decorate!(mod, output_var, Lava.Dec.Binding, UInt32(1))

    gl_global_id = Lava.emit_global_variable!(mod, ptr_uvec3_in_t, Lava.SC.Input)
    Lava.emit_decorate!(mod, gl_global_id, Lava.Dec.BuiltIn, Lava.BuiltIn.GlobalInvocationId)

    # Entry point & execution mode
    func_id = Lava.fresh_id!(mod)
    Lava.emit_entry_point!(mod, Lava.ExecModel.GLCompute, func_id, "main",
        UInt32[gl_global_id, input_var, output_var])
    Lava.emit_execution_mode!(mod, func_id, Lava.ExecMode.LocalSize,
        UInt32(local_size), UInt32(1), UInt32(1))

    # Function body
    label_id = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunction,
        void_t, func_id, Lava.FuncControl.None, func_t)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLabel, label_id)

    # %gid_vec = OpLoad %v3uint %gl_GlobalInvocationID
    gid_vec = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, uvec3_t, gid_vec, gl_global_id)

    # %gid_x = OpCompositeExtract %uint %gid_vec 0
    gid_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, UInt16(81), u32_t, gid_x, gid_vec, UInt32(0))

    # %input_ptr = OpAccessChain %ptr_f32_sb %input_var %0 %gid_x
    input_elem_ptr = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpAccessChain,
        ptr_f32_sb_t, input_elem_ptr, input_var, const_0_u32, gid_x)

    # %val = OpLoad %float %input_ptr
    input_val = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, f32_t, input_val, input_elem_ptr)

    # %result = OpFMul %float %val %const_2
    result_val = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFMul, f32_t, result_val, input_val, const_2_f32)

    # %output_ptr = OpAccessChain %ptr_f32_sb %output_var %0 %gid_x
    output_elem_ptr = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpAccessChain,
        ptr_f32_sb_t, output_elem_ptr, output_var, const_0_u32, gid_x)

    # OpStore %output_ptr %result
    Lava.encode_instruction!(mod.functions, Lava.Op.OpStore, output_elem_ptr, result_val)

    Lava.encode_instruction!(mod.functions, Lava.Op.OpReturn)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunctionEnd)

    # Debug names
    Lava.emit_name!(mod, func_id, "main")
    Lava.emit_name!(mod, input_var, "input_buf")
    Lava.emit_name!(mod, output_var, "output_buf")
    Lava.emit_name!(mod, gl_global_id, "gl_GlobalInvocationID")

    return Lava.serialize(mod)
end

@testset "hand-built shader modules" begin
    # output[gid] = input[gid] * 2, two storage-buffer descriptors.
    @testset "compute: times 2" begin
        spirv_binary = build_times2_shader()
        @test length(spirv_binary) > 0
        @test length(spirv_binary) % 4 == 0  # must be UInt32 aligned

        # Check magic number
        magic = reinterpret(UInt32, spirv_binary[1:4])[1]
        @test magic == 0x07230203

        # Validate with spirv-val
        @test_nowarn Lava.validate_spirv(spirv_binary)

        # Disassemble and check key instructions
        disasm = Lava.disassemble_spirv(spirv_binary)
        @test occursin("OpEntryPoint GLCompute", disasm)
        @test occursin("OpExecutionMode %main LocalSize 64 1 1", disasm)
        @test occursin("OpFMul", disasm)
        @test occursin("BuiltIn GlobalInvocationId", disasm)
        @test occursin("StorageBuffer", disasm)
    end
end
