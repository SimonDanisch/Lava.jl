# Subgroup / group-non-uniform intrinsics for Lava.jl
#
# Direct bindings to SPIR-V OpGroupNonUniform* ops.  The emitter dispatches
# on the LLVM function name `_lava_subgroup_<kind>_<op>_<suffix>` and maps it
# to the correct OpGroupNonUniform* with the right GroupOperation.
#
# `Subgroup` scope on Vulkan maps to a wavefront on AMD (64 lanes on GCN/RDNA
# up to RDNA2; 32 or 64 on RDNA3 depending on the shader — RX 7900 XTX prefers
# 64 for compute by default).  Use `subgroup_size()` at runtime if you need the
# actual count; don't hard-code it.
#
# ### Quick reference
# All reduction-style ops have the default `reduce` variant and explicit
# `inclusive_scan` / `exclusive_scan` variants.
#
#   subgroup_add(x)               reduce-add across the subgroup
#   subgroup_inclusive_scan_add(x) prefix-scan (inclusive)
#   subgroup_exclusive_scan_add(x) prefix-scan (exclusive)
#
# Same for `mul`, `min`, `max`, `and`, `or`, `xor` (the latter three are
# integer-only — float variants would need bitcast).
#
# ### Non-arithmetic
#   subgroup_elect()              true on exactly one lane per subgroup
#   subgroup_broadcast_first(x)   value from first active lane → all lanes
#   subgroup_all(p)               AND across the subgroup
#   subgroup_any(p)               OR  across the subgroup
#
# ### Lane identity and the shuffle family
#   subgroup_size()               lanes per subgroup, as actually compiled
#   subgroup_lane()               this lane's 0-based index
#   subgroup_shuffle(x, id)       absolute lane
#   subgroup_shuffle_xor(x, mask) butterfly partner
#   subgroup_shuffle_up/down(x,d) neighbour at ±d, UNDEFINED off the end
#   subgroup_rotate(x, d)         neighbour at +d, wrapping (needs the extension)
#   subgroup_broadcast(x, id)     absolute lane, `id` must be uniform
# See the block above their definitions for the three ways these bite.

# ── Code-gen helper ──
# Builds an @inline function that does a single llvmcall with the right LLVM
# signature. The emitter will pattern-match on the function name.

macro subgroup_arith_op(jl_name, intrinsic_name, T_jl, T_ir)
    jl_sym = Symbol(jl_name)
    llvm_name = string(intrinsic_name)
    ir = """
        declare $(T_ir) @$(llvm_name)($(T_ir)) #0
        define $(T_ir) @entry($(T_ir) %v) #0 {
            %r = call $(T_ir) @$(llvm_name)($(T_ir) %v)
            ret $(T_ir) %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        @inline function $(esc(jl_sym))(v::$(esc(T_jl)))
            Base.llvmcall(($ir, "entry"), $(esc(T_jl)), Tuple{$(esc(T_jl))}, v)
        end
        push!(KNOWN_INTRINSICS, $llvm_name)
    end
end

# ── Reduction variants (single-input → single-output) ──
# Type matrix: each reduce op for the types the SPIR-V ext supports.

# add
@subgroup_arith_op subgroup_add _lava_subgroup_reduce_add_f32 Float32 "float"
@subgroup_arith_op subgroup_add _lava_subgroup_reduce_add_f64 Float64 "double"
@subgroup_arith_op subgroup_add _lava_subgroup_reduce_add_i32 Int32   "i32"
@subgroup_arith_op subgroup_add _lava_subgroup_reduce_add_u32 UInt32  "i32"
@subgroup_arith_op subgroup_add _lava_subgroup_reduce_add_i64 Int64   "i64"
@subgroup_arith_op subgroup_add _lava_subgroup_reduce_add_u64 UInt64  "i64"

# mul
@subgroup_arith_op subgroup_mul _lava_subgroup_reduce_mul_f32 Float32 "float"
@subgroup_arith_op subgroup_mul _lava_subgroup_reduce_mul_f64 Float64 "double"
@subgroup_arith_op subgroup_mul _lava_subgroup_reduce_mul_i32 Int32   "i32"
@subgroup_arith_op subgroup_mul _lava_subgroup_reduce_mul_u32 UInt32  "i32"

# min
@subgroup_arith_op subgroup_min _lava_subgroup_reduce_min_f32 Float32 "float"
@subgroup_arith_op subgroup_min _lava_subgroup_reduce_min_f64 Float64 "double"
@subgroup_arith_op subgroup_min _lava_subgroup_reduce_min_i32 Int32   "i32"
@subgroup_arith_op subgroup_min _lava_subgroup_reduce_min_u32 UInt32  "i32"

# max
@subgroup_arith_op subgroup_max _lava_subgroup_reduce_max_f32 Float32 "float"
@subgroup_arith_op subgroup_max _lava_subgroup_reduce_max_f64 Float64 "double"
@subgroup_arith_op subgroup_max _lava_subgroup_reduce_max_i32 Int32   "i32"
@subgroup_arith_op subgroup_max _lava_subgroup_reduce_max_u32 UInt32  "i32"

# and/or/xor (integer only)
@subgroup_arith_op subgroup_and _lava_subgroup_reduce_and_i32 Int32  "i32"
@subgroup_arith_op subgroup_and _lava_subgroup_reduce_and_u32 UInt32 "i32"
@subgroup_arith_op subgroup_or  _lava_subgroup_reduce_or_i32  Int32  "i32"
@subgroup_arith_op subgroup_or  _lava_subgroup_reduce_or_u32  UInt32 "i32"
@subgroup_arith_op subgroup_xor _lava_subgroup_reduce_xor_i32 Int32  "i32"
@subgroup_arith_op subgroup_xor _lava_subgroup_reduce_xor_u32 UInt32 "i32"

# ── Scan variants (inclusive / exclusive prefix-scan) ──

@subgroup_arith_op subgroup_inclusive_scan_add _lava_subgroup_inclusive_scan_add_f32 Float32 "float"
@subgroup_arith_op subgroup_inclusive_scan_add _lava_subgroup_inclusive_scan_add_i32 Int32   "i32"
@subgroup_arith_op subgroup_inclusive_scan_add _lava_subgroup_inclusive_scan_add_u32 UInt32  "i32"

@subgroup_arith_op subgroup_exclusive_scan_add _lava_subgroup_exclusive_scan_add_f32 Float32 "float"
@subgroup_arith_op subgroup_exclusive_scan_add _lava_subgroup_exclusive_scan_add_i32 Int32   "i32"
@subgroup_arith_op subgroup_exclusive_scan_add _lava_subgroup_exclusive_scan_add_u32 UInt32  "i32"

# ── Elect: returns true on exactly one active lane per subgroup ──

@inline function subgroup_elect()
    raw = Base.llvmcall(("""
        declare i1 @_lava_subgroup_elect() #0
        define i8 @entry() #0 {
            %v = call i1 @_lava_subgroup_elect()
            %e = zext i1 %v to i8
            ret i8 %e
        }
        attributes #0 = { alwaysinline convergent }
    """, "entry"), UInt8, Tuple{})
    raw != 0x00
end
push!(KNOWN_INTRINSICS, "_lava_subgroup_elect")

# ── Broadcast-first: value from lowest active lane spreads to all lanes ──

macro broadcast_first_op(T_jl, T_ir, suffix)
    llvm_name = "_lava_subgroup_broadcast_first_$(suffix)"
    ir = """
        declare $(T_ir) @$(llvm_name)($(T_ir)) #0
        define $(T_ir) @entry($(T_ir) %v) #0 {
            %r = call $(T_ir) @$(llvm_name)($(T_ir) %v)
            ret $(T_ir) %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    quote
        @inline function subgroup_broadcast_first(v::$(esc(T_jl)))
            Base.llvmcall(($ir, "entry"), $(esc(T_jl)), Tuple{$(esc(T_jl))}, v)
        end
        push!(KNOWN_INTRINSICS, $llvm_name)
    end
end

@broadcast_first_op Float32 "float"  f32
@broadcast_first_op Float64 "double" f64
@broadcast_first_op Int32   "i32"    i32
@broadcast_first_op UInt32  "i32"    u32
@broadcast_first_op Int64   "i64"    i64
@broadcast_first_op UInt64  "i64"    u64

# ── Boolean votes: all / any lanes' predicate ──

@inline function subgroup_all(pred::Bool)
    raw = pred ? UInt8(1) : UInt8(0)
    out = Base.llvmcall(("""
        declare i1 @_lava_subgroup_all(i1) #0
        define i8 @entry(i8 %pi) #0 {
            %p = trunc i8 %pi to i1
            %r = call i1 @_lava_subgroup_all(i1 %p)
            %e = zext i1 %r to i8
            ret i8 %e
        }
        attributes #0 = { alwaysinline convergent }
    """, "entry"), UInt8, Tuple{UInt8}, raw)
    out != 0x00
end
push!(KNOWN_INTRINSICS, "_lava_subgroup_all")

# ── Shuffle family: read another lane's value ──
#
# `subgroup_size()` and `subgroup_lane()` are the two halves you need to use
# these: the lane index is 0-based and so are all the selectors.
#
#   subgroup_shuffle(v, id)        value from lane `id`                (absolute)
#   subgroup_shuffle_xor(v, mask)  value from lane `subgroup_lane() ⊻ mask`
#   subgroup_shuffle_up(v, d)      value from lane `subgroup_lane() - d`
#   subgroup_shuffle_down(v, d)    value from lane `subgroup_lane() + d`
#   subgroup_rotate(v, d)          value from lane `(subgroup_lane() + d) % size`
#   subgroup_broadcast(v, id)      like shuffle, but `id` must be subgroup-uniform
#
# Three things bite:
#
# 1. `up` and `down` do NOT clamp. When the source lane is outside the subgroup
#    the SPIR-V result is *undefined* — not zero — so a scan built on them has
#    to mask by lane index itself. `shuffle_xor` and `rotate` stay inside the
#    subgroup for any selector and need no mask.
# 2. Reading an *inactive* lane is undefined. These are `convergent`, but that
#    only stops the compiler from sinking them into divergent control flow; it
#    does not make a lane that took the other branch readable.
# 3. The subgroup is 32 lanes here and 32 *or* 64 on RDNA3. A butterfly written
#    as `for m in (1,2,4,8,16)` silently reduces half a wave64. Loop to
#    `subgroup_size()`.
#
# `subgroup_rotate` additionally needs VK_KHR_shader_subgroup_rotate; check
# `vk_context().subgroup_rotate_available` before compiling a kernel that uses
# it. The other five are Vulkan 1.1 core.

"""
    subgroup_size() -> UInt32

Lanes per subgroup, as the shader is actually running. Never hard-code 32.
"""
@inline subgroup_size() = lava_subgroup_size()

"""
    subgroup_lane() -> UInt32

This invocation's **0-based** index within its subgroup.
"""
@inline subgroup_lane() = lava_subgroup_local_id()

macro subgroup_shuffle_op(jl_name, kind, T_jl, T_ir, suffix)
    llvm_name = "_lava_subgroup_$(kind)_$(suffix)"
    ir = """
        declare $(T_ir) @$(llvm_name)($(T_ir), i32) #0
        define $(T_ir) @entry($(T_ir) %v, i32 %s) #0 {
            %r = call $(T_ir) @$(llvm_name)($(T_ir) %v, i32 %s)
            ret $(T_ir) %r
        }
        attributes #0 = { alwaysinline convergent }
    """
    jl_sym = Symbol(jl_name)
    quote
        @inline function $(esc(jl_sym))(v::$(esc(T_jl)), sel::Integer)
            Base.llvmcall(($ir, "entry"), $(esc(T_jl)), Tuple{$(esc(T_jl)), UInt32},
                          v, UInt32(sel))
        end
        push!(KNOWN_INTRINSICS, $llvm_name)
    end
end

for (jl_name, kind) in ((:subgroup_shuffle,      "shuffle"),
                        (:subgroup_shuffle_xor,  "shuffle_xor"),
                        (:subgroup_shuffle_up,   "shuffle_up"),
                        (:subgroup_shuffle_down, "shuffle_down"),
                        (:subgroup_rotate,       "rotate"),
                        (:subgroup_broadcast,    "broadcast"))
    for (T_jl, T_ir, suffix) in ((:Float32, "float",  "f32"),
                                 (:Float64, "double", "f64"),
                                 (:Int32,   "i32",    "i32"),
                                 (:UInt32,  "i32",    "u32"),
                                 (:Int64,   "i64",    "i64"),
                                 (:UInt64,  "i64",    "u64"))
        @eval @subgroup_shuffle_op $jl_name $kind $T_jl $T_ir $suffix
    end
end

@inline function subgroup_any(pred::Bool)
    raw = pred ? UInt8(1) : UInt8(0)
    out = Base.llvmcall(("""
        declare i1 @_lava_subgroup_any(i1) #0
        define i8 @entry(i8 %pi) #0 {
            %p = trunc i8 %pi to i1
            %r = call i1 @_lava_subgroup_any(i1 %p)
            %e = zext i1 %r to i8
            ret i8 %e
        }
        attributes #0 = { alwaysinline convergent }
    """, "entry"), UInt8, Tuple{UInt8}, raw)
    out != 0x00
end
push!(KNOWN_INTRINSICS, "_lava_subgroup_any")
