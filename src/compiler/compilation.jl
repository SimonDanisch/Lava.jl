# Main compilation pipeline for Lava.jl
#
# Pipeline: Julia function → GPUCompiler → LLVM IR → LLVM passes → custom SPIR-V emitter
#
# Two entry points:
#   lava_compile_to_llvm()  — returns LLVM IR string (for debugging)
#   lava_compile_to_spirv() — returns validated SPIR-V binary

# ── Compilation result types ──

struct LavaLLVMResult
    ir::String
    entry_name::String
    workgroup_size::NTuple{3,Int}
end

struct LavaSPIRVResult
    spirv_bytes::Vector{UInt8}
    entry_name::String
    workgroup_size::NTuple{3,Int}
    ir::String  # LLVM IR for debugging
end

"""
GPU-ready compilation result with BDA argument buffer layout.
"""
struct LavaGPUKernel
    spirv_bytes::Vector{UInt8}
    entry_name::String
    workgroup_size::NTuple{3,Int}
    push_info::PushConstantInfo
    ir::String
end

# ── LLVM IR compilation (GPUCompiler → LLVM Module) ──

"""
    lava_compile_to_llvm(f, tt; workgroup_size=(64,1,1)) -> LavaLLVMResult

Compile a Julia function to LLVM IR via GPUCompiler. Returns IR string.
"""
function lava_compile_to_llvm(@nospecialize(f), @nospecialize(tt);
                              workgroup_size::NTuple{3,Int} = (64, 1, 1))
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry_name = LLVM.name(meta.entry)
        ir = string(mod)
        return LavaLLVMResult(ir, entry_name, workgroup_size)
    end
end

# ── Full pipeline: Julia → SPIR-V ──

"""
    lava_compile_to_spirv(f, tt; workgroup_size=(64,1,1), validate=true) -> LavaSPIRVResult

Full compilation pipeline: Julia function → LLVM IR → LLVM passes → SPIR-V emission → validation.

Returns `LavaSPIRVResult` with validated SPIR-V binary, entry name, and LLVM IR for debugging.
"""
function lava_compile_to_spirv(@nospecialize(f), @nospecialize(tt);
                               workgroup_size::NTuple{3,Int} = (64, 1, 1),
                               validate::Bool = true)
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # ── Stage 1: LLVM passes ──
        _run_llvm_passes!(mod, entry_fn)

        # Save IR for debugging (after passes, before emission)
        ir = string(mod)

        # ── Stage 2: Custom SPIR-V emission ──
        spirv_bytes = _emit_spirv_from_llvm(mod, entry_name, workgroup_size)

        # ── Stage 3: Validation ──
        if validate
            _validate_spirv(spirv_bytes)
        end

        return LavaSPIRVResult(spirv_bytes, entry_name, workgroup_size, ir)
    end
end

"""
    lava_compile_gpu(f, tt; workgroup_size=(64,1,1), validate=true) -> LavaGPUKernel

Full GPU-ready compilation pipeline with BDA entry wrapper.
The resulting SPIR-V has a void() entry point that loads arguments from a BDA buffer.

The `push_info` field describes the argument buffer layout for `pack_kernel_args`.
"""
function lava_compile_gpu(@nospecialize(f), @nospecialize(tt);
                           workgroup_size::NTuple{3,Int} = (64, 1, 1),
                           validate::Bool = true)
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)

    GPUCompiler.JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry_fn = meta.entry
        entry_name = LLVM.name(entry_fn)

        # ── Stage 0: BDA entry wrapper ──
        # Must happen BEFORE passes — the wrapper becomes the new entry point
        push_info = wrap_entry_for_vulkan!(mod, entry_fn; workgroup_size)
        wrapper_name = push_info.wrapper_name

        # The wrapper function is now the entry point
        wrapper_fn = LLVM.functions(mod)[wrapper_name]

        # ── Stage 1: LLVM passes ──
        _run_llvm_passes!(mod, wrapper_fn)

        # Save IR for debugging (always available at /tmp/lava_last.ll)
        ir = string(mod)
        write("/tmp/lava_last.ll", ir)

        # ── Stage 2: Custom SPIR-V emission ──
        spirv_bytes = _emit_spirv_from_llvm(mod, wrapper_name, workgroup_size)

        # Save SPIR-V for debugging (always available at /tmp/lava_last.spv)
        write("/tmp/lava_last.spv", spirv_bytes)

        # ── Stage 3: Validation ──
        if validate
            _validate_spirv(spirv_bytes, ir)
        end

        return LavaGPUKernel(spirv_bytes, wrapper_name, workgroup_size, push_info, ir)
    end
end

# ── Stage 1: LLVM Pass Pipeline ──

function _run_llvm_passes!(mod::LLVM.Module, entry_fn::LLVM.Function)
    # ── CFG cleanup ──
    # Remove constructs that SPIR-V can't handle
    _replace_freeze!(mod)
    _strip_assume!(mod)

    # Remove trap/unreachable from error paths (GPUCompiler's lower_throw!)
    GPUCompiler.rm_trap!(mod)
    _replace_unreachable!(mod)
    _strip_noreturn!(mod)

    # ── Force-inline all internal functions ──
    # GPU shaders are single-function programs. GPUCompiler generates helper
    # functions (error throwing, boxing, etc.) that must be inlined into the
    # entry function. After inlining, the error paths become dead code.
    _force_inline_all!(mod, entry_fn)

    # ── Post-inlining optimization ──
    # After force-inlining, newly visible optimization opportunities appear:
    # - The BDA wrapper stores args into individual allocas, then the inlined
    #   kernel body constructs tuples from them. Without SROA, the tuple
    #   construction creates GEPs that read across alloca boundaries (using the
    #   full original struct type on a decomposed alloca). This is valid with
    #   opaque pointers on CPU but invalid in SPIR-V where allocas are separate.
    # - SROA decomposes these intermediate allocas, eliminating the cross-alloca reads.
    LLVM.run!(LLVM.InstCombinePass(), mod)
    LLVM.run!(LLVM.SROAPass(), mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)

    # ── Fix inttoptr address spaces after SROA ──
    # SROA eliminates allocas and creates `inttoptr i64 %bda_val to ptr` (addrspace 0)
    # for BDA pointer fields. These should be addrspace 1 (PhysicalStorageBuffer)
    # for correct SPIR-V emission. Convert them and update all downstream uses.
    _fix_inttoptr_addrspace!(mod)

    # ── Remove Julia runtime artifacts from inlined error paths ──
    # After force-inlining, error/boxing helpers may reference Julia runtime:
    # - `load i64, ptr @jl_int64_type` (type tags for boxing)
    # - `store i64, ptr inttoptr(1)` (GC tag slot writes)
    # These are dead error paths that will never execute on GPU. Remove them
    # so the SPIR-V emitter doesn't need to handle runtime declarations.
    _remove_julia_runtime_artifacts!(mod)

    # ── Lower LLVM intrinsics unsupported by SPIR-V ──
    # memcpy → typed loads/stores, lifetime markers → removed
    _lower_unsupported_intrinsics!(mod)

    # ── Fix GEPs with mismatched source types on allocas ──
    # After SROA + inlining, some GEPs reference the original full tuple type
    # through a smaller alloca pointer. Convert these to byte-offset GEPs so the
    # lift_byte_geps pass can properly convert them using the alloca's type.
    _fix_gep_alloca_type_mismatches!(mod)

    # ── Flatten chained GEPs on allocas ──
    # Pattern: gep i8 alloca -4 → gep i32 result %idx  →  gep i8 alloca (-4 + idx*4)
    # This handles Julia's 1-based MArray indexing where the base is shifted.
    _flatten_chained_geps_on_allocas!(mod)

    # ── Lift byte-offset GEPs to typed GEPs ──
    # Julia accesses struct fields via `getelementptr i8, ptr %p, i64 <offset>`.
    # SPIR-V needs typed GEPs into the struct. Run 3x as later passes may create more.
    _lift_byte_geps_on_allocas!(mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    _lift_byte_geps_on_allocas!(mod)
    LLVM.run!(LLVM.InstCombinePass(), mod)
    _lift_byte_geps_on_allocas!(mod)

    # ── Combine consecutive same-type GEPs ──
    # Patterns like `gep T, (gep T, p, i), j` → `gep T, p, add(i, j)`.
    # This avoids chained OpPtrAccessChain which some drivers handle incorrectly.
    _combine_chained_geps!(mod)

    # ── Structured control flow ──
    # SPIR-V requires structured CF. Run the full structurize pipeline.
    run_structurize_cfg_pipeline!(mod)

    # ── Lift byte-offset GEPs on workgroup globals ──
    # Convert `gep i8, @shared, <offset>` ConstantExpr to typed struct-member GEPs.
    # Must run before decompose passes so the emitter sees proper typed access patterns.
    dl = LLVM.datalayout(mod)
    # ── Decompose workgroup typepun copies ──
    # LLVM may optimize shared memory struct copies (shared[i] = shared[j]) into raw
    # integer block copies. Detect and replace with per-field typed copies.
    _decompose_workgroup_typepun_copies!(mod, dl)

    # ── Decompose composite workgroup accesses ──
    # Struct loads/stores on addrspace(3) must be decomposed into scalar ops
    # because shared memory is flattened to scalar arrays in SPIR-V.
    _decompose_composite_workgroup_accesses!(mod, dl)

    # ── Decompose type-punned alloca loads ──
    # LLVM memcpy lowering may create `load i64, ptr %alloca_of_struct` which
    # reads raw bytes from a padded struct. For workgroup stores, decompose
    # the entire copy into per-field struct stores. For other uses, replace
    # with first-field load + zext.
    _decompose_typepun_alloca_loads!(mod, dl)

    # Re-run composite workgroup decomposition: the typepun pass may have created
    # new struct stores to workgroup memory that need field-by-field decomposition.
    _decompose_composite_workgroup_accesses!(mod, dl)

    # Decompose type-punned loads through GEPs into struct allocas.
    # LLVM memcpy optimization creates `load i64, ptr %gep_to_float_field` —
    # reads crossing struct field boundaries. Decompose into per-field loads + pack.
    _decompose_typepun_gep_loads!(mod, dl)

    # LLVM may create loads where the type differs from the alloca type
    # (e.g., `load i16` from `alloca { [2 x i8] }`). SPIR-V requires strict
    # type matching, so rewrite these to byte-by-byte extraction.
    _fix_alloca_type_mismatched_loads!(mod)

    # LLVM SROA/memcpy lowering creates `store i32, ptr %alloca_of_[16 x i64]`.
    # SPIR-V requires the stored value type to match the pointer's pointee type.
    # Rewrite these stores to drill into the alloca type via GEP.
    _fix_alloca_type_mismatched_stores!(mod, dl)

    # Lower chained mismatched-type GEPs on allocas.
    # Julia's MArray/StaticArray patterns create chains like:
    #   %base = getelementptr i32, ptr %alloca_[16 x i64], i64 -1
    #   %elem = getelementptr i32, ptr %base, i64 %var
    #   store i32 %val, ptr %elem
    # Lower to proper element-level access with runtime index computation.
    _lower_chained_mismatched_geps!(mod)

    # ── Lower byte-offset GEP chains on MArray allocas ──
    # After InstCombine splits flattened byte-offset GEPs back into chains:
    #   %gep1 = gep i8, ptr %alloca_[16xi64], %dynamic
    #   %gep2 = gep i8, ptr %gep1, -4
    #   store i32 %val, ptr %gep2
    # Lower to proper element-level access with shift/mask (no integer divide).
    _lower_byte_gep_chain_on_allocas!(mod)

    # ── Lift byte-offset GEPs on workgroup globals ──
    # The decompose passes above may create byte-offset ConstantExpr GEPs like
    # `gep i8, @shared, <offset>` when splitting struct loads from workgroup globals.
    # Convert these to typed struct-member GEPs so the emitter produces proper OpAccessChain.
    _lift_byte_geps_on_workgroup_globals!(mod, dl)

    # ── Final cleanup: fix GEPs with mismatched source types on allocas ──
    # LLVM's SROA/memcpy lowering can create typed GEPs using wrong source types
    # (e.g., `gep [3 x float]` on `alloca [3 x i32]`). Convert these to byte-offset
    # GEPs followed by the lift pass to normalize types.
    _fix_gep_alloca_type_mismatches!(mod)
    _lift_byte_geps_on_allocas!(mod)

    return nothing
end

"""
Force-inline all internal functions into the entry function.

Marks all non-entry, non-declaration functions as `alwaysinline`, runs the
AlwaysInliner pass, then removes dead functions with GlobalDCE.
After this, only the entry function (and LLVM intrinsic declarations) remain.
"""
function _force_inline_all!(mod::LLVM.Module, entry_fn::LLVM.Function)
    entry_name = LLVM.name(entry_fn)

    for fn in LLVM.functions(mod)
        fn_name = LLVM.name(fn)

        # Skip entry function, declarations (no body), and LLVM intrinsics
        fn_name == entry_name && continue
        isempty(LLVM.blocks(fn)) && continue
        startswith(fn_name, "llvm.") && continue

        # Remove noinline, add alwaysinline
        attrs = LLVM.function_attributes(fn)
        delete!(attrs, LLVM.EnumAttribute("noinline"))
        push!(attrs, LLVM.EnumAttribute("alwaysinline"))
    end

    # Run inliner
    LLVM.run!(LLVM.AlwaysInlinerPass(), mod)

    # Remove now-dead internal functions
    LLVM.run!(LLVM.GlobalDCEPass(), mod)

    return nothing
end

# ── Stage 2: SPIR-V Emission ──

"""
Emit SPIR-V binary from an LLVM module. Handles type mapping, instruction emission,
entry point setup, and serialization.
"""
function _emit_spirv_from_llvm(llvm_mod::LLVM.Module, entry_name::String,
                                workgroup_size::NTuple{3,Int})
    # Build pointee type map (opaque pointer → typed pointer recovery)
    ptm = build_pointee_type_map(llvm_mod)

    # Create SPIR-V module and type context
    spirv_mod = SPIRVModule()
    type_ctx = SPIRVTypeContext(spirv_mod, ptm)

    # Setup module header
    setup_memory_model!(spirv_mod; physical_storage_buffer=true)
    require_capability!(spirv_mod, Cap.Shader)
    require_capability!(spirv_mod, Cap.VariablePointers)
    require_extension!(spirv_mod, "SPV_KHR_variable_pointers")

    # Build struct pointer member type map (resolves ptr members in structs)
    build_struct_ptr_member_types!(type_ctx, llvm_mod)

    # Pre-collect all types used in the module
    collect_module_types!(type_ctx, llvm_mod)

    # Create emitter state
    state = SPIRVEmitterState(spirv_mod, type_ctx)

    # Find the entry function
    entry_fn = LLVM.functions(llvm_mod)[entry_name]

    # Emit global variables (if any — needed for builtin inputs, etc.)
    interface_ids = _emit_globals!(state, llvm_mod)

    # Check if entry function has parameters
    fn_ty = LLVM.function_type(entry_fn)
    n_params = length(collect(LLVM.parameters(fn_ty)))

    if n_params == 0
        # No parameters — emit directly as entry point
        func_id = emit_function!(state, entry_fn; is_entry=true)
    else
        # Entry functions in Vulkan SPIR-V cannot have parameters.
        # Create a parameterless wrapper that calls the kernel.
        # For now, pass undef values for parameters (real BDA wrapper comes later).
        func_id = _emit_entry_wrapper!(state, entry_fn)
    end

    # Setup entry point and execution modes
    emit_entry_point!(spirv_mod, ExecModel.GLCompute, func_id, "main", interface_ids)
    emit_execution_mode!(spirv_mod, func_id, ExecMode.LocalSize,
                         UInt32(workgroup_size[1]), UInt32(workgroup_size[2]), UInt32(workgroup_size[3]))

    # Add debug name
    emit_name!(spirv_mod, func_id, entry_name)

    # Add Block + MemberOffset decorations for PSB-pointed struct types
    decorate_psb_struct_layouts!(type_ctx, llvm_mod)

    # Serialize to binary
    return serialize(spirv_mod)
end

"""
Emit a parameterless entry wrapper that calls the kernel function.
For now, passes undef values for parameters. The real BDA entry wrapper will
load arguments from a device-memory buffer via PhysicalStorageBuffer.
"""
function _emit_entry_wrapper!(state::SPIRVEmitterState, entry_fn::LLVM.Function)
    mod = state.mod

    # First, emit the original function as a non-entry function
    inner_func_id = emit_function!(state, entry_fn; is_entry=false)

    # Create wrapper function type: void()
    void_ty = emit_type_void!(mod)
    wrapper_fn_ty = emit_type_function!(mod, void_ty)

    # Create wrapper function
    wrapper_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpFunction, void_ty, wrapper_id, FuncControl.None, wrapper_fn_ty)

    # Entry block
    label_id = fresh_id!(mod)
    encode_instruction!(mod.functions, Op.OpLabel, label_id)

    # Build undef arguments for each parameter
    arg_ids = UInt32[]
    for param in LLVM.parameters(entry_fn)
        param_ty = LLVM.value_type(param)
        if param_ty isa LLVM.PointerType
            param_spirv_ty = map_pointer_type_for_value!(state.type_ctx, param)
        else
            param_spirv_ty = map_type!(state.type_ctx, param_ty)
        end
        # Create OpUndef for each parameter
        undef_id = fresh_id!(mod)
        encode_instruction!(mod.types_constants, UInt16(1), param_spirv_ty, undef_id)  # OpUndef
        push!(arg_ids, undef_id)
    end

    # Call the inner function
    word_count = UInt32(4 + length(arg_ids))
    push!(mod.functions, (word_count << 16) | UInt32(Op.OpFunctionCall))
    push!(mod.functions, void_ty)
    call_result_id = fresh_id!(mod)
    push!(mod.functions, call_result_id)
    push!(mod.functions, inner_func_id)
    append!(mod.functions, arg_ids)

    # Return
    encode_instruction!(mod.functions, Op.OpReturn)
    encode_instruction!(mod.functions, Op.OpFunctionEnd)

    return wrapper_id
end

"""
Emit global variables from the LLVM module. Returns interface variable IDs for OpEntryPoint.
Handles push constant globals (addrspace 2) and builtin globals (addrspace 7).
"""
function _emit_globals!(state::SPIRVEmitterState, llvm_mod::LLVM.Module)
    interface_ids = UInt32[]

    for gv in LLVM.globals(llvm_mod)
        gv_ty = LLVM.value_type(gv)
        gv_ty isa LLVM.PointerType || continue
        as = LLVM.addrspace(gv_ty)

        if as == 2
            # Push constant global (addrspace 2)
            var_id = _emit_push_constant_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 3
            # Workgroup/shared memory global (addrspace 3)
            var_id = _emit_workgroup_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 7
            # Input global (addrspace 7) — SPIR-V builtins
            var_id = _emit_builtin_global!(state, gv)
            push!(interface_ids, var_id)
        elseif as == 1
            # Julia constant global (addrspace 1) — lookup tables like _j_const_N
            var_id = _emit_constant_global!(state, gv)
            if var_id !== nothing
                push!(interface_ids, var_id)  # SPIR-V 1.4+ requires all globals in interface
            end
        end
    end

    return interface_ids
end

"""
Emit a push constant global variable in SPIR-V.
Creates the struct type with Block decoration, pointer type, and OpVariable.
"""
function _emit_push_constant_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_value_ty = LLVM.global_value_type(gv)

    # Map the struct type
    struct_spirv_id = map_type!(state.type_ctx, gv_value_ty)

    # Add Block decoration to the struct type (required for PushConstant)
    emit_decorate!(mod, struct_spirv_id, Dec.Block)

    # Add MemberOffset decorations
    if gv_value_ty isa LLVM.StructType
        members = LLVM.elements(gv_value_ty)
        offset = UInt32(0)
        for (i, member_ty) in enumerate(members)
            emit_member_decorate!(mod, struct_spirv_id, UInt32(i - 1), Dec.Offset, offset)
            offset += UInt32(_compute_type_size(member_ty))
        end
    end

    # Create pointer type: OpTypePointer PushConstant %struct_type
    ptr_ty = map_pointer_type!(state.type_ctx, struct_spirv_id, SC.PushConstant)

    # Create OpVariable
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.PushConstant)

    # Register in value_map so loads can find it
    state.value_map[gv] = var_id

    # Register pointee type in PTM for downstream pointer type resolution
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    return var_id
end

"""
Emit a workgroup (shared memory) global variable in SPIR-V.
Creates OpVariable with Workgroup storage class for addrspace(3) globals.
These are used by KA's @localmem for shared memory within a workgroup.

Workgroup variables must NOT have explicit layout decorations (ArrayStride,
Offset, Block) unless VK_KHR_workgroup_memory_explicit_layout is enabled.
We create FRESH type IDs (via `map_workgroup_type!`) separate from the main
cache, so PSB types keep their decorations while workgroup types stay clean.
"""
function _emit_workgroup_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_value_ty = LLVM.global_value_type(gv)

    # Create FRESH type IDs for workgroup usage (no layout decorations).
    # This is critical when the same LLVM type (e.g. [3 x float]) is used in both
    # PSB (needs ArrayStride) and Workgroup (must NOT have ArrayStride).
    pointee_spirv = map_workgroup_type!(state.type_ctx, gv_value_ty)

    # Create pointer type: OpTypePointer Workgroup %fresh_type
    ptr_ty = map_pointer_type!(state.type_ctx, pointee_spirv, SC.Workgroup)

    # Create OpVariable with Workgroup storage class
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Workgroup)

    # Register in value_map so references can find it
    state.value_map[gv] = var_id

    # Register pointee type in PTM for downstream GEP/load/store resolution
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    # Debug name
    gv_name = LLVM.name(gv)
    if !isempty(gv_name)
        emit_name!(mod, var_id, gv_name)
    end

    return var_id
end

# ── Builtin name → SPIR-V BuiltIn decoration mapping ──
const _SPIRV_BUILTIN_MAP = Dict{String, UInt32}(
    "__spirv_BuiltInGlobalInvocationId"   => BuiltIn.GlobalInvocationId,
    "__spirv_BuiltInLocalInvocationId"    => BuiltIn.LocalInvocationId,
    "__spirv_BuiltInWorkgroupId"          => BuiltIn.WorkgroupId,
    "__spirv_BuiltInNumWorkgroups"        => BuiltIn.NumWorkgroups,
    "__spirv_BuiltInWorkgroupSize"        => BuiltIn.WorkgroupSize,
    "__spirv_BuiltInLocalInvocationIndex" => BuiltIn.LocalInvocationIndex,
)

"""
Emit a builtin Input global variable in SPIR-V.
Creates the appropriate type (vec3<u32> or u32), pointer type, OpVariable,
and BuiltIn decoration based on the global's name.
"""
function _emit_builtin_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_name = LLVM.name(gv)
    gv_value_ty = LLVM.global_value_type(gv)

    # Look up builtin decoration from name
    builtin_id = get(_SPIRV_BUILTIN_MAP, gv_name, nothing)
    if builtin_id === nothing
        error("Unknown SPIR-V builtin global: $gv_name")
    end

    # Map the value type (e.g., <3 x i32> → OpTypeVector(i32, 3), or i32)
    value_spirv_id = map_type!(state.type_ctx, gv_value_ty)

    # Create pointer type: OpTypePointer Input %value_type
    ptr_ty = map_pointer_type!(state.type_ctx, value_spirv_id, SC.Input)

    # Create OpVariable in Input storage class
    var_id = fresh_id!(mod)
    encode_instruction!(mod.global_vars, Op.OpVariable, ptr_ty, var_id, SC.Input)

    # Add BuiltIn decoration
    emit_decorate!(mod, var_id, Dec.BuiltIn, builtin_id)

    # Add debug name
    emit_name!(mod, var_id, gv_name)

    # Register in value_map so loads can find it
    state.value_map[gv] = var_id

    # Register pointee type in PTM
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    return var_id
end

"""
Emit a Julia constant global (addrspace 1) as a Private-scope SPIR-V variable
with a constant initializer. These are lookup tables generated by Julia's
math library (e.g., polynomial coefficients for log/exp/pow).

Strategy: Create an OpConstantComposite for the array data, then declare
an OpVariable in Private storage class with the composite as initializer.
"""
function _emit_constant_global!(state::SPIRVEmitterState, gv::LLVM.GlobalVariable)
    mod = state.mod
    gv_value_ty = LLVM.global_value_type(gv)
    gv_name = LLVM.name(gv)

    # Must be an array type
    if !(gv_value_ty isa LLVM.ArrayType)
        @warn "Skipping non-array constant global: $gv_name (type: $gv_value_ty)"
        return
    end

    # Must have an initializer
    init = LLVM.initializer(gv)
    if init === nothing
        @warn "Skipping constant global without initializer: $gv_name"
        return
    end

    # Map the array type to SPIR-V
    arr_spirv_ty = map_type!(state.type_ctx, gv_value_ty)

    # Build the composite constant recursively
    composite_id = _emit_llvm_constant!(state, init, gv_value_ty)

    # Create pointer type: OpTypePointer Private %array_type
    ptr_ty = map_pointer_type!(state.type_ctx, arr_spirv_ty, SC.Private)

    # Create OpVariable with Private SC and initializer
    var_id = fresh_id!(mod)
    word_count = UInt32(5)  # result_type, result_id, storage_class, initializer
    push!(mod.global_vars, (word_count << 16) | UInt32(Op.OpVariable))
    push!(mod.global_vars, ptr_ty)
    push!(mod.global_vars, var_id)
    push!(mod.global_vars, UInt32(SC.Private))
    push!(mod.global_vars, composite_id)

    # Register in value_map
    state.value_map[gv] = var_id

    # Register pointee type in PTM
    set_pointee_type!(state.type_ctx.ptm, gv, gv_value_ty; priority=5)

    # Debug name
    emit_name!(mod, var_id, gv_name)

    return var_id
end

"""
Emit an LLVM constant value as a SPIR-V constant, returning its ID.
Handles ConstantInt, ConstantFP, ConstantDataArray, ConstantArray, ConstantStruct.
"""
function _emit_llvm_constant!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.LLVMType)
    # Scalar constants
    if val isa LLVM.ConstantInt || val isa LLVM.ConstantFP ||
       val isa LLVM.UndefValue || val isa LLVM.PoisonValue ||
       val isa LLVM.ConstantAggregateZero
        return map_constant!(state.type_ctx, val)
    end

    # ConstantDataArray/ConstantDataVector — packed scalar data
    if val isa LLVM.ConstantDataSequential
        return _emit_constant_data_array!(state, val, ty)
    end

    # ConstantArray — array of aggregate elements
    if ty isa LLVM.ArrayType
        return _emit_constant_array!(state, val, ty)
    end

    # ConstantStruct
    if ty isa LLVM.StructType
        return _emit_constant_struct!(state, val, ty)
    end

    error("Unsupported LLVM constant type for emission: $(typeof(val)), LLVM type: $ty")
end

"""Emit a ConstantDataArray (packed scalar data) as OpConstantComposite."""
function _emit_constant_data_array!(state::SPIRVEmitterState, val::LLVM.ConstantDataSequential, ty::LLVM.ArrayType)
    n = LLVM.length(ty)
    elem_ty = LLVM.eltype(ty)
    arr_spirv_ty = map_type!(state.type_ctx, ty)

    # Extract each element via LLVMGetElementAsConstant
    elem_ids = UInt32[]
    for i in 0:(n-1)
        elem_ref = LLVM.API.LLVMGetElementAsConstant(val, UInt32(i))
        elem_val = LLVM.Value(elem_ref)::LLVM.Constant
        elem_id = map_constant!(state.type_ctx, elem_val)
        push!(elem_ids, elem_id)
    end

    # Emit OpConstantComposite
    composite_id = fresh_id!(state.mod)
    word_count = UInt32(3 + length(elem_ids))
    push!(state.mod.types_constants, (word_count << 16) | UInt32(Op.OpConstantComposite))
    push!(state.mod.types_constants, arr_spirv_ty)
    push!(state.mod.types_constants, composite_id)
    append!(state.mod.types_constants, elem_ids)

    return composite_id
end

"""Emit a ConstantArray (array of aggregate elements) as OpConstantComposite."""
function _emit_constant_array!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.ArrayType)
    n = LLVM.length(ty)
    elem_ty = LLVM.eltype(ty)
    arr_spirv_ty = map_type!(state.type_ctx, ty)

    ops = LLVM.operands(val)
    elem_ids = UInt32[]
    for i in 1:n
        elem_id = _emit_llvm_constant!(state, ops[i]::LLVM.Constant, elem_ty)
        push!(elem_ids, elem_id)
    end

    # Emit OpConstantComposite
    composite_id = fresh_id!(state.mod)
    word_count = UInt32(3 + length(elem_ids))
    push!(state.mod.types_constants, (word_count << 16) | UInt32(Op.OpConstantComposite))
    push!(state.mod.types_constants, arr_spirv_ty)
    push!(state.mod.types_constants, composite_id)
    append!(state.mod.types_constants, elem_ids)

    return composite_id
end

"""Emit a ConstantStruct as OpConstantComposite."""
function _emit_constant_struct!(state::SPIRVEmitterState, val::LLVM.Constant, ty::LLVM.StructType)
    members = LLVM.elements(ty)
    struct_spirv_ty = map_type!(state.type_ctx, ty)

    ops = LLVM.operands(val)
    member_ids = UInt32[]
    for (i, member_ty) in enumerate(members)
        member_id = _emit_llvm_constant!(state, ops[i]::LLVM.Constant, member_ty)
        push!(member_ids, member_id)
    end

    # Emit OpConstantComposite
    composite_id = fresh_id!(state.mod)
    word_count = UInt32(3 + length(member_ids))
    push!(state.mod.types_constants, (word_count << 16) | UInt32(Op.OpConstantComposite))
    push!(state.mod.types_constants, struct_spirv_ty)
    push!(state.mod.types_constants, composite_id)
    append!(state.mod.types_constants, member_ids)

    return composite_id
end

# ── Stage 3: SPIR-V Validation ──

"""
Validate SPIR-V binary using spirv-val. On failure, writes debug artifacts
to /tmp/lava_last.{spv,dis,ll} and throws with a focused error excerpt.
"""
function _validate_spirv(spirv_bytes::Vector{UInt8}, llvm_ir::String="")
    spirv_val = SPIRV_Tools_jll.spirv_val()
    spirv_dis_cmd = SPIRV_Tools_jll.spirv_dis()
    spv_path = "/tmp/lava_last.spv"

    # Validate — capture stderr via temp file (spirv-val writes errors to stderr)
    val_err_file = tempname()
    p = run(pipeline(`$spirv_val --target-env vulkan1.3 --scalar-block-layout $spv_path`;
                      stderr=val_err_file, stdout=devnull); wait=false)
    wait(p)
    val_errors = isfile(val_err_file) ? read(val_err_file, String) : ""
    rm(val_err_file; force=true)
    p.exitcode == 0 && return nothing

    # ── Validation failed — build a useful error message ──

    # Disassemble and save
    dis = try
        read(`$spirv_dis_cmd --no-color $spv_path`, String)
    catch
        ""
    end
    if !isempty(dis)
        write("/tmp/lava_last.dis", dis)
    end

    # Extract the error line numbers and show context from disassembly
    excerpt = ""
    if !isempty(dis)
        dis_lines = split(dis, '\n')
        error_line_nums = Int[]
        for m in eachmatch(r"line (\d+):", val_errors)
            push!(error_line_nums, parse(Int, m.captures[1]))
        end
        unique!(error_line_nums)

        if !isempty(error_line_nums)
            io = IOBuffer()
            for lnum in error_line_nums[1:min(3, end)]  # at most 3 error sites
                lo = max(1, lnum - 8)
                hi = min(length(dis_lines), lnum + 5)
                println(io, "  ┌─ SPIR-V around line $lnum:")
                for j in lo:hi
                    marker = j == lnum ? " >> " : "    "
                    println(io, "  │$marker$j: ", dis_lines[j])
                end
                println(io, "  └───")
            end
            excerpt = String(take!(io))
        end
    end

    error("""
    SPIR-V validation failed!

    spirv-val:
    $(strip(val_errors))

    $(isempty(excerpt) ? "" : excerpt)
    Debug files:
      LLVM IR:     /tmp/lava_last.ll
      SPIR-V bin:  /tmp/lava_last.spv
      SPIR-V dis:  /tmp/lava_last.dis
    """)
end

"""
Disassemble SPIR-V binary to text for debugging.
"""
function disassemble_spirv(spirv_bytes::Vector{UInt8})
    spirv_dis = SPIRV_Tools_jll.spirv_dis()
    tmpfile = tempname() * ".spv"
    try
        write(tmpfile, spirv_bytes)
        return read(`$spirv_dis --no-color $tmpfile`, String)
    finally
        rm(tmpfile; force=true)
    end
end

# ── Lower unsupported LLVM intrinsics ──

"""
Lower LLVM intrinsics that SPIR-V cannot represent:
- llvm.memcpy → typed load + store (using destination alloca's type)
- llvm.lifetime.start/end → removed (no-op)
"""
function _lower_unsupported_intrinsics!(mod::LLVM.Module)
    to_erase = LLVM.Instruction[]

    for fn in LLVM.functions(mod)
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.CallInst || continue
                called = LLVM.called_operand(inst)
                called isa LLVM.Function || continue
                fname = LLVM.name(called)

                if startswith(fname, "llvm.memcpy")
                    _lower_memcpy!(inst)
                    push!(to_erase, inst)
                elseif startswith(fname, "llvm.memset")
                    _lower_memset!(inst)
                    push!(to_erase, inst)
                elseif startswith(fname, "llvm.lifetime")
                    push!(to_erase, inst)
                end
            end
        end
    end

    for inst in to_erase
        LLVM.erase!(inst)
    end

    # Remove dead intrinsic declarations
    for fn in collect(LLVM.functions(mod))
        fname = LLVM.name(fn)
        if (startswith(fname, "llvm.memcpy") || startswith(fname, "llvm.memset") ||
            startswith(fname, "llvm.lifetime")) && isempty(LLVM.uses(fn))
            LLVM.erase!(fn)
        end
    end
end

"""
Lower a single memcpy call to a typed load + store.
If the destination is an alloca, use the alloca's type for the load/store
so SPIR-V doesn't need to bitcast structs.
"""
function _lower_memcpy!(inst::LLVM.CallInst)
    ops = LLVM.operands(inst)
    dst = ops[1]
    src = ops[2]

    # Try to find the alloca's element type for typed load/store
    copy_type = nothing
    if dst isa LLVM.AllocaInst
        copy_type = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(dst))
    end

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        if copy_type !== nothing
            # Typed load/store using the alloca's type
            val = LLVM.load!(builder, copy_type, src, "memcpy_val")
            LLVM.store!(builder, val, dst)
        else
            # Fallback: byte-by-byte copy using i64 chunks
            len_val = ops[3]
            if !(len_val isa LLVM.ConstantInt)
                error("Cannot lower memcpy with non-constant length: $inst")
            end
            nbytes = convert(Int, len_val)
            T_i64 = LLVM.Int64Type()
            T_i8 = LLVM.Int8Type()
            offset = 0
            while offset + 8 <= nbytes
                src_ptr = LLVM.gep!(builder, T_i8, src, [LLVM.ConstantInt(T_i64, offset)])
                dst_ptr = LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, offset)])
                val = LLVM.load!(builder, T_i64, src_ptr)
                LLVM.alignment!(val, 8)
                st = LLVM.store!(builder, val, dst_ptr)
                LLVM.alignment!(st, 8)
                offset += 8
            end
            while offset < nbytes
                src_ptr = LLVM.gep!(builder, T_i8, src, [LLVM.ConstantInt(T_i64, offset)])
                dst_ptr = LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, offset)])
                val = LLVM.load!(builder, LLVM.Int8Type(), src_ptr)
                LLVM.store!(builder, val, dst_ptr)
                offset += 1
            end
        end
    end
end

"""
Lower a single memset call to explicit stores.
SPIR-V has no memset intrinsic, so we replace with i32 stores (4-byte chunks)
plus i8 tail stores. For addrspace 0 (allocas), uses GEP-based addressing.
"""
function _lower_memset!(inst::LLVM.CallInst)
    ops = LLVM.operands(inst)
    dst = ops[1]
    fill_val = ops[2]  # i8
    len_val = ops[3]

    if !(len_val isa LLVM.ConstantInt)
        error("Cannot lower memset with non-constant length: $inst")
    end
    nbytes = convert(Int, len_val)

    T_i8 = LLVM.Int8Type()
    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()

    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, inst)

        # Build fill word: replicate i8 val to i32
        val32 = LLVM.zext!(builder, fill_val, T_i32)
        v1 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 8))
        val32 = LLVM.or!(builder, val32, v1)
        v2 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 16))
        val32 = LLVM.or!(builder, val32, v2)

        # Store i32 chunks via byte-offset GEPs
        n_words = nbytes ÷ 4
        for i in 0:(n_words-1)
            off = i * 4
            ptr = if off == 0
                dst
            else
                LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, off)])
            end
            st = LLVM.store!(builder, val32, ptr)
            LLVM.alignment!(st, 4)
        end

        # Handle tail bytes
        for i in (n_words*4):(nbytes-1)
            ptr = LLVM.gep!(builder, T_i8, dst, [LLVM.ConstantInt(T_i64, i)])
            st = LLVM.store!(builder, fill_val, ptr)
            LLVM.alignment!(st, 1)
        end
    end
end

# ── Compiler caches ──

const _compiler_cache = Dict{Any, Any}()
const _kernel_cache = Dict{UInt, Any}()
const _compile_lock = ReentrantLock()
