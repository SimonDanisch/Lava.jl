# SPIR-V objects for the functions left after LLVM inlining. An edit to a caller can reuse
# its unchanged helpers, including across sessions; spirv-link resolves their imports before
# the kernel's ordinary optimization and validation. Globals live in one shared object.

function function_linkage!(mod::SPIRVModule, id::UInt32, name::String, import_symbol::Bool)
    words = UInt32[id, 41]
    encode_string_words!(words, name)
    push!(words, import_symbol ? UInt32(1) : UInt32(0))
    encode_instruction!(mod.annotations, Op.OpDecorate, words...)
end

function linked_interfaces(bytes::Vector{UInt8})
    words = reinterpret(UInt32, bytes)
    globals = UInt32[]
    i = 6
    while i <= length(words)
        n, op = Int(words[i] >> 16), UInt16(words[i] & 0xffff)
        op == Op.OpVariable && words[i+3] != SC.Function && push!(globals, words[i+2])
        i += n
    end
    builtin_annotations = UInt32[]
    i = 6
    while i <= length(words)
        n, op = Int(words[i] >> 16), UInt16(words[i] & 0xffff)
        if op == Op.OpName
            name = String(filter(!iszero, reinterpret(UInt8, collect(words[i+2:i+n-1]))))
            if startswith(name, "lava_link_builtin_")
                kind = parse(UInt32, name[length("lava_link_builtin_")+1:end])
                encode_instruction!(builtin_annotations, Op.OpDecorate, words[i+1], Dec.BuiltIn, kind)
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
        if op == Op.OpEntryPoint
            j = i+3
            while all(b -> b != 0, reinterpret(UInt8, UInt32[words[j]])); j += 1; end
            prefix = collect(words[i+1:j])
            encode_instruction!(out, op, prefix..., unique(globals)...)
        else
            append!(out, words[i:i+n-1])
        end
        i += n
    end
    collect(reinterpret(UInt8, out))
end

function declare_function!(state::SPIRVEmitterState, fn::LLVM.Function, symbol::String)
    mod, ctx = state.mod, state.type_ctx
    ret = map_type!(ctx, fn.function_type.return_type)
    params = UInt32[p.value_type isa LLVM.PointerType ? map_pointer_type_for_value!(ctx, p) :
                   map_type!(ctx, p.value_type) for p in LLVM.parameters(fn)]
    ft = emit_type_function!(mod, ret, params)
    id = state.value_map[fn]
    encode_instruction!(mod.functions, Op.OpFunction, ret, id, FuncControl.None, ft)
    for ty in params
        encode_instruction!(mod.functions, Op.OpFunctionParameter, ty, fresh_id!(mod))
    end
    encode_instruction!(mod.functions, Op.OpFunctionEnd)
    function_linkage!(mod, id, symbol, true)
end

function function_object(llvm_mod::LLVM.Module, fn::LLVM.Function, symbols;
                         entry::Bool, wg::NTuple{3,Int}, enable_ray_query::Bool)
    state, interfaces = compute_emitter(llvm_mod; enable_ray_query)
    mod, ctx = state.mod, state.type_ctx
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

function function_fingerprint(mod::LLVM.Module, fn::LLVM.Function, symbols)
    c = copy(mod)
    try
        for f in LLVM.functions(c)
            if f.name != fn.name
                isempty(f.blocks) || empty!(f)
                f.linkage = LLVM.Linkage.External
                original = mod.functions[f.name]
                haskey(symbols,original) && (f.name = symbols[original])
            else
                f.linkage = LLVM.Linkage.External
            end
        end
        LLVM.run!(LLVM.StripDeadPrototypesPass(),c)
        ptm = build_pointee_type_map(mod)
        vals = [collect(LLVM.parameters(fn)); [i for b in fn.blocks for i in b.instructions]]
        pointees = [(string(get(ptm.map, v, nothing)),trace_pointer_to_alloca(v),
                     v isa LLVM.Argument && param_called_with_alloca_arg(v))
                    for v in vals if v.value_type isa LLVM.PointerType]
        ctx = SPIRVTypeContext(SPIRVModule(),ptm)
        build_struct_ptr_member_types!(ctx,mod)
        members = sort!([(string(t),i,string(p),a) for ((t,i),(p,a)) in ctx.struct_ptr_members])
        callbacks = SPIRVEmitterState(ctx.mod, ctx)
        collect_inline_callbacks!(callbacks, mod)
        globals = [reached(gv) for gv in LLVM.globals(mod)]
        ObjCache.keyhash(1, canonical_ir(c),
                         serialized((pointees, members, globals, fn in callbacks.inline_callbacks)))
    finally
        LLVM.dispose(c)
    end
end


function emit_function_objects(llvm_mod, entry, reachable, workgroup_size; enable_ray_query)
    for scc in strongly_connected_components(reachable)
        length(scc) > 1 && error("Mutual recursion is not supported in SPIR-V: " * join((f.name for f in scc), " -> "))
    end
    symbols = Dict{LLVM.Function,String}()
    emitted = Set{String}()
    objects = Vector{UInt8}[]
    globals = UInt8[]
    keep = !generating_output() && !kernel_dump_wanted()
    for fn in reachable
        key = if fn === entry
            UInt8[]
        else
            rawkey = function_fingerprint(llvm_mod, fn, symbols)
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
            push!(objects, found)
        else
            fn !== entry && (COMPILE_STATS.function_misses += 1)
            code, data = function_object(llvm_mod, fn, symbols; entry = fn === entry,
                                         wg = workgroup_size, enable_ray_query)
            push!(objects, code)
            fn === entry && (globals = data)
            keep && fn !== entry && ObjCache.put!("Lava/function", key, code)
        end
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
        return linked_interfaces(read(output)), Dict{UInt32,Tuple{String,Int}}()
    end
end
