# SPIR-V objects for the functions left after LLVM inlining. An edit to a caller can reuse
# its unchanged helpers, including across sessions; spirv-link resolves their imports before
# the kernel's ordinary optimization and validation. Globals live in one shared object.

function function_linkage!(mod::SPIRVModule, id::UInt32, name::String, import_symbol::Bool)
    words = UInt32[id, 41]
    encode_string_words!(words, name)
    push!(words, import_symbol ? UInt32(1) : UInt32(0))
    encode_instruction!(mod.annotations, Op.OpDecorate, words...)
end

function linked_interfaces(bytes::Vector{UInt8}, sources)
    words = reinterpret(UInt32, bytes)
    globals = UInt32[]
    i = 6
    while i <= length(words)
        n, op = Int(words[i] >> 16), UInt16(words[i] & 0xffff)
        op == Op.OpVariable && words[i+3] != SC.Function && push!(globals, words[i+2])
        i += n
    end
    builtin_annotations = UInt32[]
    locations = Dict{UInt32, Tuple{String, Int}}()
    i = 6
    while i <= length(words)
        n, op = Int(words[i] >> 16), UInt16(words[i] & 0xffff)
        if op == Op.OpName
            name = String(filter(!iszero, reinterpret(UInt8, collect(words[i+2:i+n-1]))))
            if startswith(name, "lava_link_builtin_")
                kind = parse(UInt32, name[length("lava_link_builtin_")+1:end])
                encode_instruction!(builtin_annotations, Op.OpDecorate, words[i+1], Dec.BuiltIn, kind)
            elseif startswith(name, "lava_link_source_")
                object, instruction = parse.(Int, split(name[length("lava_link_source_")+1:end], '_'))
                loc = sources[object][instruction]
                loc === nothing || (locations[words[i+1]] = loc)
            end
        end
        i += n
    end
    out = collect(words[1:5])
    added = false
    i = 6
    while i <= length(words)
        n, op = Int(words[i] >> 16), UInt16(words[i] & 0xffff)
        if !added && 19 <= op <= 39
            append!(out,builtin_annotations)
            added = true
        end
        if op == Op.OpName && startswith(
                String(filter(!iszero, reinterpret(UInt8, collect(words[i+2:i+n-1])))), "lava_link_source_")
            # These names only carry relocation information through spirv-link.
        elseif op == Op.OpEntryPoint
            j = i+3
            while all(b -> b != 0, reinterpret(UInt8, UInt32[words[j]])); j += 1; end
            prefix = collect(words[i+1:j])
            encode_instruction!(out, op, prefix..., unique(globals)...)
        else
            append!(out, words[i:i+n-1])
        end
        i += n
    end
    return collect(reinterpret(UInt8, out)), locations
end

function tag_function_sources(bytes::Vector{UInt8}, object::Int)
    words = reinterpret(UInt32, bytes)
    out = collect(words[1:5])
    i = 6
    while i <= length(words)
        n, op = Int(words[i] >> 16), UInt16(words[i] & 0xffff)
        if op == Op.OpName
            name = String(filter(!iszero, reinterpret(UInt8, collect(words[i+2:i+n-1]))))
            if startswith(name, "lava_link_source_")
                ordinal = name[length("lava_link_source_")+1:end]
                operands = UInt32[words[i+1]]
                encode_string_words!(operands, "lava_link_source_$(object)_$ordinal")
                encode_instruction!(out, op, operands...)
                i += n
                continue
            end
        end
        append!(out, words[i:i+n-1])
        i += n
    end
    return collect(reinterpret(UInt8, out))
end

function declare_function!(state::SPIRVEmitterState, fn::LLVM.Function, symbol::String)
    mod, ctx = state.mod, state.type_ctx
    ret = map_type!(ctx, fn.function_type.return_type)
    params = UInt32[function_parameter_type!(state, p) for p in LLVM.parameters(fn)]
    ft = emit_type_function!(mod, ret, params)
    id = state.value_map[fn]
    encode_instruction!(mod.functions, Op.OpFunction, ret, id, FuncControl.None, ft)
    for ty in params
        encode_instruction!(mod.functions, Op.OpFunctionParameter, ty, fresh_id!(mod))
    end
    encode_instruction!(mod.functions, Op.OpFunctionEnd)
    function_linkage!(mod, id, symbol, true)
end

function copy_compute_emitter(template::SPIRVEmitterState)
    mod = deepcopy(template.mod)
    t = template.type_ctx
    ctx = SPIRVTypeContext(copy(t.llvm_to_spirv), copy(t.pointer_types), mod,
        PointeeTypeMap(copy(t.ptm.map)), copy(t.struct_ptr_members), copy(t.workgroup_type_map))
    state = SPIRVEmitterState(mod, ctx)
    state.value_map = copy(template.value_map)
    state.inline_callbacks = copy(template.inline_callbacks)
    state.wg_wrapped_vars = copy(template.wg_wrapped_vars)
    state.rt_tlas_var_id = template.rt_tlas_var_id
    state.rt_accel_type_id = template.rt_accel_type_id
    state.ray_query_type_id = template.ray_query_type_id
    state.entry_interface_ids = copy(template.entry_interface_ids)
    state.data_layout = template.data_layout
    state.features = template.features
    return state
end

function function_object(llvm_mod::LLVM.Module, fn::LLVM.Function, symbols, state, interfaces;
                         entry::Bool, wg::NTuple{3,Int})
    mod, ctx = state.mod, state.type_ctx
    state.source_instruction_indices = Dict(inst => i for (i, inst) in
        enumerate(inst for block in fn.blocks for inst in block.instructions))
    require_capability!(mod, UInt32(5)) # Linkage: spirv-link removes it.
    for callee in collect_reachable_callees(fn)
        callee === fn && continue
        declare_function!(state, callee, symbols[callee])
    end
    id = entry && !isempty(LLVM.parameters(fn)) ? emit_entry_wrapper!(state, fn) :
         emit_function!(state, fn; is_entry=entry)
    if entry
        emit_entry_point!(mod, ExecModel.GLCompute, id, "main", interfaces)
        emit_execution_mode!(mod, id, ExecMode.LocalSize, UInt32.(wg)...)
    else
        function_linkage!(mod, id, symbols[fn], false)
    end
    emit_name!(mod, id, fn.name)
    for (result, (_, ordinal)) in mod.source_locations
        emit_name!(mod, result, "lava_link_source_$ordinal")
    end
    decorate_psb_struct_layouts!(ctx, llvm_mod)
    # Split the globals from the function. Every object imports them; a single
    # globals object supplies their definitions for the linked kernel.
    # SPIRV-Tools' linker explicitly ignores symbols decorated BuiltIn. Carry that
    # decoration as a name until imports have been resolved, then restore it in
    # linked_interfaces. Source: SPIRV-Tools/source/link/linker.cpp, GetImportExportPairs.
    builtin_ids = Dict{UInt32,UInt32}()
    annotations = UInt32[]
    i = 1
    while i <= length(mod.annotations)
        n = Int(mod.annotations[i] >> 16)
        if UInt16(mod.annotations[i] & 0xffff) == Op.OpDecorate && mod.annotations[i+2] == Dec.BuiltIn
            builtin_ids[mod.annotations[i+1]] = mod.annotations[i+3]
        else
            append!(annotations,mod.annotations[i:i+n-1])
        end
        i += n
    end
    empty!(mod.annotations)
    append!(mod.annotations,annotations)
    globals = entry ? deepcopy(mod) : nothing
    module_ids = Set{UInt32}()
    if entry
        empty!(globals.functions)
        empty!(globals.entry_points)
        empty!(globals.execution_modes)
        empty!(globals.debug)
        for (id, kind) in builtin_ids
            emit_name!(globals, id, "lava_link_builtin_$kind")
        end
        i = 1
        while i <= length(mod.types_constants)
            n = Int(mod.types_constants[i] >> 16)
            op = UInt16(mod.types_constants[i] & 0xffff)
            constant = op in (UInt16(1), UInt16(41), UInt16(42), UInt16(43), UInt16(44),
                              UInt16(45), UInt16(46), UInt16(48), UInt16(49), UInt16(50),
                              UInt16(51), UInt16(52))
            push!(module_ids, mod.types_constants[i + (constant ? 2 : 1)])
            i += n
        end
    end
    i = 1
    declarations = UInt32[]
    while i <= length(mod.global_vars)
        n = Int(mod.global_vars[i] >> 16)
        id = mod.global_vars[i+2]
        entry && push!(module_ids, id)
        name = "lava_global_$(length(declarations))"
        function_linkage!(mod,id,name,true)
        entry && function_linkage!(globals, id, name, false)
        encode_instruction!(declarations,Op.OpVariable,mod.global_vars[i+1],id,mod.global_vars[i+3])
        i += n
    end
    empty!(mod.global_vars)
    append!(mod.global_vars,declarations)
    entry || return serialize(mod), UInt8[]
    annotations = UInt32[]
    i = 1
    while i <= length(globals.annotations)
        n = Int(globals.annotations[i] >> 16)
        globals.annotations[i+1] in module_ids && append!(annotations, globals.annotations[i:i+n-1])
        i += n
    end
    empty!(globals.annotations)
    append!(globals.annotations, annotations)
    return serialize(mod), serialize(globals)
end

function function_module_skeleton(mod::LLVM.Module)
    c = copy(mod)
    for fn in c.functions
        isempty(fn.blocks) || empty!(fn)
    end
    LLVM.strip_debuginfo!(c)
    return c
end

function isolated_function_ir(mod::LLVM.Module, fn::LLVM.Function, skeleton, symbols)
    c = copy(skeleton)
    try
        values = Dict{LLVM.Value,LLVM.Value}()
        for f in mod.functions
            values[f] = c.functions[f.name]
        end
        for g in LLVM.globals(mod)
            values[g] = LLVM.globals(c)[g.name]
        end
        target = c.functions[fn.name]
        for (a, b) in zip(LLVM.parameters(fn), LLVM.parameters(target))
            values[a] = b
        end
        LLVM.clone_into!(target, fn; value_map=values,
                         changes=LLVM.API.LLVMCloneFunctionChangeTypeDifferentModule)
        for f in LLVM.functions(c)
            f.linkage = LLVM.Linkage.External
            if f !== target
                original = mod.functions[f.name]
                haskey(symbols,original) && (f.name = symbols[original])
            end
        end
        LLVM.run!(LLVM.StripDeadPrototypesPass(),c)
        LLVM.run!(LLVM.GlobalDCEPass(),c)
        return canonical_ir(c)
    finally
        LLVM.dispose(c)
    end
end

function function_global_interface(mod::LLVM.Module, skeleton)
    c = copy(skeleton)
    try
        # StripDeadPrototypes also deletes external globals once the function
        # bodies are gone, including the builtin inputs that every object imports.
        # Remove unused function declarations explicitly so those globals survive.
        for fn in collect(c.functions)
            isempty(fn.uses) && LLVM.erase!(fn)
        end
        # Objects import globals by their order of emission, including globals
        # used by other functions. Their declarations, initializers and reachability
        # must agree before a helper object can be reused in another module.
        return ObjCache.keyhash(1, canonical_ir(c), serialized([reached(gv) for gv in LLVM.globals(mod)]))
    finally
        LLVM.dispose(c)
    end
end

function function_fingerprint(mod::LLVM.Module, fn::LLVM.Function, skeleton, symbols, state, globals)
    ptm = state.type_ctx.ptm
    vals = [collect(LLVM.parameters(fn)); [i for b in fn.blocks for i in b.instructions]]
    pointees = [(string(get(ptm.map, v, nothing)), trace_pointer_to_alloca(v),
                 v isa LLVM.Argument && param_called_with_alloca_arg(v))
                for v in vals if v.value_type isa LLVM.PointerType]
    ctx = state.type_ctx
    members = sort!([(string(t),i,string(p),a) for ((t,i),(p,a)) in ctx.struct_ptr_members])
    return ObjCache.keyhash(6, isolated_function_ir(mod, fn, skeleton, symbols), globals,
                           serialized((pointees, members, fn in state.inline_callbacks)))
end

function emit_function_objects(llvm_mod, entry, reachable, workgroup_size; enable_ray_query)
    for scc in strongly_connected_components(reachable)
        length(scc) > 1 && error("Mutual recursion is not supported in SPIR-V: " * join((f.name for f in scc), " -> "))
    end
    symbols = Dict{LLVM.Function,String}()
    emitted = Set{String}()
    objects = Vector{UInt8}[]
    sources = Vector{Union{Nothing,Tuple{String,Int}}}[]
    globals = UInt8[]
    keep = !generating_output() && !kernel_dump_wanted()
    # Type recovery and global discovery scan the whole LLVM module. Do them once;
    # helpers get independent copies of the prepared buffers and maps. The entry,
    # emitted last, can consume the original state directly.
    template, interfaces = compute_emitter(llvm_mod; enable_ray_query)
    # Copying the complete LLVM body once per tiny helper dwarfs emitting it.
    # Keep the declarations and globals once, then clone only the helper's body.
    skeleton = function_module_skeleton(llvm_mod)
    try
        global_interface = function_global_interface(llvm_mod, skeleton)
        for fn in reachable
            key = if fn === entry
                UInt8[]
            else
                rawkey = function_fingerprint(llvm_mod, fn, skeleton, symbols, template, global_interface)
                ObjCache.keyhash(PERSISTED_SCHEMA, rawkey,
                                 serialized((compiler_identity(), workgroup_size, enable_ray_query)))
            end
            symbols[fn] = fn === entry ? "main" : "lava_function_" * bytes2hex(key)
            if fn !== entry
                if symbols[fn] in emitted
                    COMPILE_STATS.function_hits += 1
                    continue
                end
                push!(emitted, symbols[fn])
            end
            found = keep && fn !== entry ? ObjCache.get("Lava/function", key) : nothing
            if found !== nothing
                COMPILE_STATS.function_hits += 1
                code = found
            else
                fn !== entry && (COMPILE_STATS.function_misses += 1)
                state = fn === entry ? template : copy_compute_emitter(template)
                code, data = function_object(llvm_mod, fn, symbols, state, interfaces;
                                             entry = fn === entry, wg = workgroup_size)
                fn === entry && (globals = data)
                keep && fn !== entry && ObjCache.put!("Lava/function", key, code)
            end
            push!(sources, Union{Nothing,Tuple{String,Int}}[
                extract_source_location(template, inst) for block in fn.blocks for inst in block.instructions])
            push!(objects, tag_function_sources(code, length(sources)))
        end
    finally
        LLVM.dispose(skeleton)
    end
    push!(objects, globals)
    return mktempdir() do dir
        paths = [joinpath(dir, "function$i.spv") for i in eachindex(objects)]
        for (path, bytes) in zip(paths, objects)
            write(path, bytes)
            record_subprocess!("write spirv for spirv-link", 0.0, length(bytes))
        end
        output = joinpath(dir, "linked.spv")
        errpath = joinpath(dir, "link-errors.txt")
        command = `$(SPIRV_Tools_jll.spirv_link()) --target-env spv1.4 $paths -o $output`
        process = lava_run(pipeline(command; stdout = devnull, stderr = errpath); label = "spirv-link")
        process_exited(process) && process.exitcode == 0 ||
            error("SPIR-V function linking failed: " * read(errpath, String))
        return linked_interfaces(read(output), sources)
    end
end
