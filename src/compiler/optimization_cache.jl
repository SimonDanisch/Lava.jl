# The optimized module is reusable before the expensive LLVM pass pipeline runs.
# Keep relocation identities and source locations with the input: neither host
# addresses nor stale debug locations may cross a session boundary.

function optimization_debug_locations(mod::LLVM.Module)
    memo = Dict{LLVM.API.LLVMMetadataRef,Union{Nothing,Tuple{String,Int}}}()
    return [((loc = instruction_diloc(inst)); loc === nothing ? nothing : diloc_outermost(loc, memo))
            for fn in mod.functions for block in fn.blocks for inst in block.instructions]
end

# Materialized Julia constants are appended in a nondeterministic order. Their
# first use, including references from other initializers, gives a stable order
# without conflating distinct globals that happen to contain the same value.
function optimization_globals(mod::LLVM.Module)
    seen = Set{LLVM.Value}()
    ordered = LLVM.GlobalVariable[]
    function visit(value)
        value in seen && return
        push!(seen, value)
        if value isa LLVM.GlobalVariable
            push!(ordered, value)
            value.initializer === nothing || visit(value.initializer)
        elseif value isa LLVM.User && !(value isa LLVM.Function)
            foreach(visit, LLVM.operands(value))
        end
    end
    for fn in mod.functions, block in fn.blocks, inst in block.instructions
        foreach(visit, LLVM.operands(inst))
    end
    foreach(visit, LLVM.globals(mod))
    return ordered
end

function optimization_input(mod::LLVM.Module)
    c = copy(mod)
    try
        for gv in optimization_globals(c)
            LLVM.move!(gv, LLVM.at_end(c.globals))
        end
        return canonical_ir(c)
    finally
        LLVM.dispose(c)
    end
end

optimization_symbols(mod::LLVM.Module) =
    [String[f.name for f in mod.functions]; String[g.name for g in optimization_globals(mod)]]

# Relocation sites use job-local symbols. Key them by the same symbol order as
# the canonical IR; their generated names and manifest order carry no semantics.
function optimization_relocations(names::Vector{String}, relocs::GPUCompiler.Relocations)
    isempty(relocs) && return ()
    symbols = Dict(name => i for (i, name) in enumerate(names))
    records = map(relocs.records) do record
        site = get(symbols, record.name, 0)
        (; kind = record.kind, site, name = site == 0 ? record.name : "",
           offset = record.offset, target = record.target)
    end
    return sort!(records; by = record -> (record.site, record.name, record.offset, Int(record.kind)))
end
optimization_relocations(mod::LLVM.Module, relocs::GPUCompiler.Relocations) =
    optimization_relocations(optimization_symbols(mod), relocs)

function restore_optimization_symbols!(mod::LLVM.Module, old, current)
    names = Dict(zip(old, current))
    renames = Tuple{LLVM.GlobalValue,String}[]
    for value in [LLVM.GlobalValue[f for f in mod.functions]; LLVM.GlobalValue[g for g in LLVM.globals(mod)]]
        replacement = get(names, value.name, value.name)
        replacement == value.name && continue
        push!(renames, (value, replacement))
        value.name = "__lava_cached_symbol_$(length(renames))"
    end
    for (value, name) in renames
        value.name = name
    end
    return names
end

# GPUCompiler owns the module handle. Replace its contents, including named
# metadata and the data layout, rather than returning a different module.
function replace_optimization_ir!(mod::LLVM.Module, restored::LLVM.Module)
    for name in collect(keys(mod.metadata))
        delete!(mod.metadata, name)
    end
    functions = collect(mod.functions)
    globals = collect(LLVM.globals(mod))
    for fn in functions
        isempty(fn.blocks) || empty!(fn)
        fn.personality = nothing
    end
    for gv in globals
        gv.initializer = nothing
    end
    for value in [LLVM.GlobalValue[functions...]; LLVM.GlobalValue[globals...]]
        LLVM.replace_uses!(value, LLVM.UndefValue(value.value_type))
        LLVM.erase!(value)
    end
    empty!(mod.inline_asm)
    mod.datalayout = restored.datalayout
    mod.triple = restored.triple
    LLVM.link!(mod, restored)
    return nothing
end

function optimize_uncached!(job::LavaCompilerJob, mod::LLVM.Module, relocs::GPUCompiler.Relocations; opt_level)
    return invoke(GPUCompiler.optimize!, Tuple{GPUCompiler.CompilerJob,LLVM.Module,GPUCompiler.Relocations},
                  job, mod, relocs; opt_level)
end

function GPUCompiler.optimize!(job::LavaCompilerJob, mod::LLVM.Module,
                              relocs::GPUCompiler.Relocations; opt_level = 2)
    keep = HAS_OBJECT_CACHE && !generating_output() && !kernel_dump_wanted() &&
           isempty(LLVM.aliases(mod)) && isempty(LLVM.ifuncs(mod))
    keep || return optimize_uncached!(job, mod, relocs; opt_level)
    config = job.config
    settings = [(name, getfield(config, name)) for name in fieldnames(typeof(config)) if name !== :cache_owner]
    current = optimization_symbols(mod)
    key = ObjCache.keyhash(2, optimization_input(mod),
                          serialized((compiler_identity(), Base.module_build_id(GPUCompiler), settings,
                                      opt_level, optimization_debug_locations(mod), optimization_relocations(current, relocs))))
    found = ObjCache.get("Lava/optimized-llvm", key)
    if found === nothing
        COMPILE_STATS.llvm_misses += 1
        optimize_uncached!(job, mod, relocs; opt_level)
        GPUCompiler.prune_dead_relocations!(mod, relocs)
        if isempty(LLVM.aliases(mod)) && isempty(LLVM.ifuncs(mod))
            # Normalize through bitcode and linking on both misses and hits.
            # Intrinsic attributes and metadata must reach downstream keys in
            # the same form whichever path supplied the optimized module.
            io = IOBuffer()
            write(io, mod)
            replace_optimization_ir!(mod, parse(LLVM.Module, take!(io)))
            io = IOBuffer()
            write(io, mod)
            entry = (; bitcode = take!(io), symbols = current, relocations = copy(relocs.records))
            ObjCache.put!("Lava/optimized-llvm", key, serialized(entry))
        end
    else
        COMPILE_STATS.llvm_hits += 1
        entry = Serialization.deserialize(IOBuffer(found))
        restored = parse(LLVM.Module, entry.bitcode)
        names = restore_optimization_symbols!(restored, entry.symbols, current)
        replace_optimization_ir!(mod, restored)
        originals = Dict((r.name, r.offset) => r for r in relocs.records)
        GPUCompiler.check_mutable(relocs, "restore")
        empty!(relocs.records)
        for record in entry.relocations
            name = get(names, record.name, record.name)
            original = get(originals, (name, record.offset), nothing)
            target = original === nothing ? record.target : original.target
            GPUCompiler.add_relocation!(relocs, GPUCompiler.Relocation(record.kind, name, record.offset, target))
        end
        GPUCompiler.prune_dead_relocations!(mod, relocs)
    end
    return nothing
end
