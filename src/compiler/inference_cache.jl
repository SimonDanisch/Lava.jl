# An earlier persistent lookup, before Julia emits LLVM. Fingerprint the exact
# inferred CodeInfos supplied to jl_emit_native, closed under direct invokes.
# CodeInstance worlds, host entry pointers and cached analysis results do not
# describe generated code and never enter the key. Constants, native ABI flags,
# debug information, layouts and compilation settings do.
#
# This is deliberately a partial domain: an opaque handle, unresolved direct
# callee or unknown mutable value falls back to the LLVM-keyed cache. Nothing
# deserializes compiler IR or installs an inferred CodeInstance from disk.

@static if HAS_OBJECT_CACHE && VERSION >= v"1.12.0-DEV.1823" &&
           isdefined(GPUCompiler.CompilerCaching, :get_codeinfos)

struct IRKeyUnsupported <: Exception
    value_type::Any
end

mutable struct IRKeyState
    io::IOBuffer
    serializer::Serialization.Serializer{IOBuffer}
    objects::IdDict{Any,Int}
    types::IdDict{Any,Int}
    sources::IdDict{Core.CodeInstance,Core.CodeInfo}
    world::UInt
end
function IRKeyState(pairs, world)
    io = IOBuffer()
    IRKeyState(io, Serialization.Serializer(io), IdDict{Any,Int}(), IdDict{Any,Int}(),
             IdDict{Core.CodeInstance,Core.CodeInfo}(pairs), world)
end
irkey_atom!(s, x) = Serialization.serialize(s.serializer, x)
irkey_tag!(s, x::UInt8) = write(s.io, x)

function irkey_reference!(s, x)
    id = get(s.objects, x, 0)
    if id != 0
        irkey_tag!(s, 0x00)
        irkey_atom!(s, id)
        return true
    end
    irkey_tag!(s, 0x01)
    s.objects[x] = length(s.objects) + 1
    return false
end

function irkey!(s, x::Module)
    irkey_tag!(s, 0x02)
    irkey_reference!(s, x) && return
    irkey_atom!(s, Base.fullname(x))
    root = Base.moduleroot(x)
    pkg = Base.PkgId(root)
    irkey_atom!(s, string(pkg.uuid))
    # Runtime-created modules carry a per-process build ID. Their inferred
    # instructions and type layouts, rather than that nonce, identify them.
    irkey_atom!(s, pkg.uuid === nothing && root !== Base && root !== Core ? nothing : Base.module_build_id(root))
end
function irkey!(s, x::DataType)
    irkey_tag!(s, 0x03)
    id = get(s.types, x, 0)
    irkey_atom!(s, id)
    id != 0 && return
    s.types[x] = length(s.types) + 1
    irkey!(s, parentmodule(x))
    irkey_atom!(s, nameof(x))
    irkey!(s, x.parameters)
    irkey_atom!(s, isconcretetype(x))
    irkey_atom!(s, isprimitivetype(x))
    irkey_atom!(s, ismutabletype(x))
    irkey_atom!(s, isbitstype(x))
    if isconcretetype(x) && parentmodule(x) !== Core
        irkey_atom!(s, sizeof(x))
        irkey_atom!(s, Base.datatype_alignment(x))
        irkey_atom!(s, fieldnames(x))
        for i in 1:fieldcount(x)
            irkey_atom!(s, fieldoffset(x, i))
            irkey!(s, fieldtype(x, i))
        end
    end
end
function irkey!(s, x::UnionAll)
    irkey_tag!(s, 0x04)
    irkey!(s, x.var)
    irkey!(s, x.body)
end
function irkey!(s, x::Union)
    irkey_tag!(s, 0x05)
    irkey!(s, x.a)
    irkey!(s, x.b)
end
function irkey!(s, x::TypeVar)
    irkey_tag!(s, 0x06)
    irkey_reference!(s, x) && return
    irkey_atom!(s, x.name)
    irkey!(s, x.lb)
    irkey!(s, x.ub)
end
function irkey!(s, x::Core.TypeofBottom)
    irkey_tag!(s, 0x07)
end
function irkey!(s, x::Union{Symbol,String})
    x isa String && position(s.io) + sizeof(x) > 32 * 1024^2 && throw(IRKeyUnsupported(String))
    irkey_tag!(s, 0x08)
    irkey_atom!(s, x)
end
Base.@nospecializeinfer function irkey!(s, @nospecialize(x::Tuple))
    irkey_tag!(s, 0x09)
    irkey_atom!(s, length(x))
    for i in 1:length(x)
        irkey!(s, getfield(x, i))
    end
end
function irkey!(s, x::Core.SimpleVector)
    irkey_tag!(s, 0x0a)
    irkey_atom!(s, length(x))
    foreach(y -> irkey!(s, y), x)
end
function irkey!(s, x::Array)
    position(s.io) + sizeof(x) > 32 * 1024^2 && throw(IRKeyUnsupported(typeof(x)))
    irkey_tag!(s, 0x0b)
    irkey_reference!(s, x) && return
    irkey!(s, typeof(x))
    irkey_atom!(s, size(x))
    if isprimitivetype(eltype(x)) && !(eltype(x) <: Ptr)
        write(s.io, reinterpret(UInt8, vec(x)))
    else
        for i in eachindex(x)
            position(s.io) > 32 * 1024^2 && throw(IRKeyUnsupported(typeof(x)))
            assigned = isassigned(x, i)
            irkey_atom!(s, assigned)
            assigned && irkey!(s, x[i])
        end
    end
end
function irkey!(s, x::GlobalRef)
    irkey_tag!(s, 0x0c)
    irkey_reference!(s, x) && return
    irkey!(s, x.mod)
    irkey_atom!(s, x.name)
    # Global values contribute even if the binding itself stays unchanged.
    Base.invoke_in_world(s.world, isdefined, x.mod, x.name) || throw(IRKeyUnsupported(GlobalRef))
    irkey!(s, Base.invoke_in_world(s.world, getglobal, x.mod, x.name))
end
function irkey!(s, x::Core.MethodInstance)
    irkey_tag!(s, 0x0d)
    irkey_reference!(s, x) && return
    irkey!(s, x.def)
    irkey!(s, x.specTypes)
    irkey!(s, x.sparam_vals)
end
function irkey!(s, x::Method)
    x.is_for_opaque_closure && throw(IRKeyUnsupported(Method))
    irkey_tag!(s, 0x0e)
    irkey_reference!(s, x) && return
    irkey!(s, x.module)
    irkey_atom!(s, x.name)
    irkey_atom!(s, x.file)
    irkey_atom!(s, x.line)
    irkey!(s, x.sig)
    irkey_atom!(s, x.nargs)
    irkey_atom!(s, x.isva)
    irkey_atom!(s, x.is_for_opaque_closure)
    irkey_atom!(s, x.purity)
    irkey_atom!(s, x.constprop)
    irkey_atom!(s, x.nospecializeinfer)
    irkey_atom!(s, x.nkw)
    irkey_atom!(s, x.nospecialize)
    irkey_atom!(s, isdefined(x, :ccallable))
    isdefined(x, :ccallable) && irkey!(s, x.ccallable)
end
function irkey!(s, x::Core.CodeInstance)
    irkey_tag!(s, 0x0f)
    irkey_reference!(s, x) && return
    irkey!(s, x.def)
    irkey!(s, x.rettype)
    irkey_atom!(s, isdefined(x, :rettype_const))
    isdefined(x, :rettype_const) && irkey!(s, x.rettype_const)
    irkey_atom!(s, Core.Compiler.use_const_api(x))
    source = get(s.sources, x, nothing)
    source === nothing && throw(IRKeyUnsupported(Core.CodeInstance))
    irkey!(s, source)
end
function irkey!(s, x::Core.CodeInfo)
    position(s.io) > 32 * 1024^2 && throw(IRKeyUnsupported(Core.CodeInfo))
    irkey_tag!(s, 0x10)
    irkey_reference!(s, x) && return
    for name in fieldnames(Core.CodeInfo)
        # These control inference validity rather than generated code. Inlined
        # dependencies are already represented by their resulting instructions.
        name in (:edges, :min_world, :max_world, :method_for_inference_limit_heuristics) && continue
        irkey_atom!(s, name)
        assigned = isdefined(x, name)
        irkey_atom!(s, assigned)
        assigned && irkey!(s, getfield(x, name))
    end
end
function irkey!(s, x::Expr)
    irkey_tag!(s, 0x11)
    irkey_reference!(s, x) && return
    irkey_atom!(s, x.head)
    irkey!(s, x.args)
end
function irkey!(s, x::Ptr)
    throw(IRKeyUnsupported(typeof(x)))
end
Base.@nospecializeinfer function irkey!(s, @nospecialize(x))
    T = typeof(x)
    irkey_tag!(s, 0x12)
    irkey!(s, T)
    if isprimitivetype(T)
        write(s.io, Ref(x))
        return
    end
    # Mutable compiler IR nodes are containers, not process handles. Unknown
    # mutable values deliberately disable this earlier cache.
    ismutabletype(T) && !(T <: Union{Core.PhiNode,Core.PhiCNode}) && throw(IRKeyUnsupported(T))
    isstructtype(T) || throw(IRKeyUnsupported(T))
    irkey_reference!(s, x) && return
    irkey_atom!(s, fieldcount(T))
    for i in 1:fieldcount(T)
        assigned = isdefined(x, i)
        irkey_atom!(s, assigned)
        assigned && irkey!(s, getfield(x, i))
    end
end

"""Portable identity of the inferred inputs to LLVM emission, or `nothing` for an unsupported input."""
function inferred_kernel_key(job)
    interp = GPUCompiler.get_interpreter(job)
    root = GPUCompiler.drive_inference!(interp, job.source)
    root === nothing && return nothing
    pairs = GPUCompiler.CompilerCaching.get_codeinfos(interp, root)
    s = IRKeyState(pairs, job.world)
    try
        # A compilable/vararg-widened CI may be shared by distinct concrete
        # entry signatures. The entry ABI still depends on the actual job.
        irkey!(s, job.source)
        irkey!(s, root)
        for (ci, _) in pairs
            irkey!(s, ci)
        end
        config = job.config
        irkey!(s, [(name, getfield(config, name)) for name in fieldnames(typeof(config)) if name !== :cache_owner])
        irkey_atom!(s, compiler_identity())
        irkey_atom!(s, Base.module_build_id(GPUCompiler))
        irkey_atom!(s, (Sys.ARCH, Sys.WORD_SIZE, Base.GIT_VERSION_INFO.commit))
        options = Base.JLOptions()
        irkey_atom!(s, (options.opt_level, options.opt_level_min, options.debug_level,
                       options.check_bounds, options.can_inline, options.fast_math,
                       options.polly, options.image_codegen, options.strip_metadata))
        return ObjCache.keyhash(1, take!(s.io))
    catch err
        err isa IRKeyUnsupported || rethrow()
        return nothing
    end
end

else

inferred_kernel_key(job) = nothing

end
