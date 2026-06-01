# On-kernel printf for debugging GPU kernels.
#
# Lowers to a NonSemantic.DebugPrintf OpExtInst (see compiler/spirv/emit.jl
# `emit_debug_printf!`). The format string is hex-encoded into the callee name so
# it survives GPUCompiler without needing an LLVM string global, and the runtime
# values are passed as ordinary call arguments.
#
# Usage inside a kernel:
#     @lava_printf "tid=%u  sdx=%d  off=%lu\n" UInt32(i) sdx off
#
# Output appears via the validation layer's debug-printf messenger; enable it with
# `Lava.enable_debug_printf!()` (resets the device with the layer feature on) and
# read messages back with `Lava.get_validation_messages()` / they are also
# @info-logged by the debug-utils callback.
#
# Format specifiers follow Vulkan debug printf. Match the specifier WIDTH to the
# argument width or the layer warns:
#   Int32/UInt32 (and narrower) → %d / %i / %u / %x / %X
#   Int64/UInt64                → %ld / %lu / %lx
#   Float32 (and Float16)       → %f / %e / %g
#   Float64                     → %lf / %le / %lg   (the `l` is required for 64-bit)
# Lava passes each value at its canonical width (small ints widen to i32,
# Float16 widens to float). Vector forms %v<N><t> work if you pass the components.

export @lava_printf

# Julia arg type → (LLVM type string, canonical Julia type passed to llvmcall).
@inline function _lava_printf_canon(@nospecialize(T))
    T <: Bool                               && return ("i32", Int32)
    T <: Union{Int8, Int16, Int32}          && return ("i32", Int32)
    T <: Union{UInt8, UInt16, UInt32}       && return ("i32", UInt32)
    T <: Union{Int64, Int}                  && return ("i64", Int64)
    T <: UInt64                             && return ("i64", UInt64)
    T <: Union{Float16, Float32}            && return ("float", Float32)
    T <: Float64                            && return ("double", Float64)
    error("@lava_printf: unsupported argument type $T — only integer and " *
          "floating-point scalars are supported (widen/convert vectors yourself).")
end

@generated function _lava_printf_impl(::Val{FMT}, args::Vararg{Any, N}) where {FMT, N}
    fmt = String(FMT)
    llvm_types = Vector{String}(undef, N)
    canon_types = Vector{Any}(undef, N)
    for i in 1:N
        lty, cty = _lava_printf_canon(args[i])
        llvm_types[i] = lty
        canon_types[i] = cty
    end

    hexfmt = bytes2hex(Vector{UInt8}(fmt))
    # Disambiguate same-format / different-signature call sites: LLVM rejects two
    # declarations of one name with differing parameter types.
    sigcode = bytes2hex(Vector{UInt8}(join(llvm_types, ",")))
    fname = "_lava_debug_printf_" * hexfmt * "__" * sigcode

    params    = join(["$(llvm_types[i]) %a$(i - 1)" for i in 1:N], ", ")
    call_args = params
    decl_ps   = join(llvm_types, ", ")

    ir = """
        declare void @$fname($decl_ps) #0
        define void @entry($params) #0 {
            call void @$fname($call_args)
            ret void
        }
        attributes #0 = { alwaysinline }
    """

    conv = [:(Base.convert($(canon_types[i]), args[$i])) for i in 1:N]
    tup  = Expr(:curly, :Tuple, canon_types...)
    return :(Base.llvmcall(($ir, "entry"), Cvoid, $tup, $(conv...)))
end

"""
    @lava_printf "format" args...

Print from inside a GPU kernel via Vulkan debug printf. The format must be a
string literal; arguments are matched positionally to its specifiers. Enable
output with `Lava.enable_debug_printf!()`.

```julia
@kernel function k!(out)
    i = @index(Global)
    @lava_printf "thread %u writing %f\\n" UInt32(i) out[i]
end
```
"""
macro lava_printf(fmt, args...)
    fmt isa AbstractString ||
        throw(ArgumentError("@lava_printf: format must be a string literal"))
    impl = GlobalRef(@__MODULE__, :_lava_printf_impl)
    return Expr(:call, impl, :(Val($(QuoteNode(Symbol(fmt))))), map(esc, args)...)
end

# ── Backend-independent KernelAbstractions.@print support ──
#
# KA lowers `@print(items...)` to `KernelAbstractions.__print(items...)`, interning
# string literals as `Val{Symbol}` and leaving runtime values as-is; the default
# __print just calls `Base.print` (the CPU path). We override __print for the Lava
# device so `@print` (the portable API, same as CUDA/AMDGPU) works on Lava: build a
# printf format string with specifiers auto-selected from the runtime arg types and
# route through the same DebugPrintf path as @lava_printf. Enable output with
# `Lava.enable_debug_printf!()`.

# Runtime value type → printf specifier, matching _lava_printf_impl's canonical width.
@inline function _ka_print_spec(@nospecialize(T))
    T <: Bool                          && return "%d"
    T <: Union{Int8, Int16, Int32}     && return "%d"
    T <: Union{UInt8, UInt16, UInt32}  && return "%u"
    T <: Union{Int64, Int}             && return "%ld"
    T <: UInt64                        && return "%lu"
    T <: Union{Float16, Float32}       && return "%f"
    T <: Float64                       && return "%lf"
    error("@print: unsupported argument type $T on Lava — only integer and " *
          "floating-point scalars are supported.")
end

@lava_device_override @generated function KernelAbstractions.__print(items...)
    fmt = IOBuffer()
    argexprs = Any[]
    for i in 1:length(items)
        T = items[i]
        if T <: Val
            # Literal text (string/number/symbol). Escape % so it is not parsed as
            # a printf specifier by the validation layer.
            lit = string(T.parameters[1])
            print(fmt, replace(lit, "%" => "%%"))
        else
            print(fmt, _ka_print_spec(T))
            push!(argexprs, :(items[$i]))
        end
    end
    fmtsym = QuoteNode(Symbol(String(take!(fmt))))
    impl = GlobalRef(@__MODULE__, :_lava_printf_impl)
    return Expr(:call, impl, :(Val($fmtsym)), argexprs...)
end
