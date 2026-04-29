# LLVM pass: retype Function-storage allocas based on uniformly-typed access.
#
# Background
# ----------
# Julia's MArray / StaticArrays.MVector lowering produces Function-storage
# allocas typed as `[N x i64]` regardless of the logical element type.  The
# resulting IR has the alloca as a byte buffer but every load/store + GEP
# carries the LOGICAL access type (`float`, `i32`, `[3 x float]`, etc.).
# SPIR-V is strictly typed, so the emitter has to reconcile this mismatch
# at every access site — resulting in a long tail of type-pun edge cases.
#
# This pass does the reconciliation ONCE, at the alloca level: when every
# load/store/GEP rooted at an alloca uses a single scalar type T at
# sizeof(T)-aligned byte offsets, retype the alloca to `[total_bytes/sizeof(T) x T]`
# and rewrite all GEPs to typed `[N x T]` indexing.  After the pass, the
# alloca and its access pattern match — the SPIR-V emitter sees clean
# OpAccessChain + OpLoad/Store with no fixups needed.
#
# Algorithm (deterministic, no heuristics)
# ----------------------------------------
# For each alloca A in entry block:
#   1. Collect every load/store reachable from A through GEPs.  Bail on
#      any non-GEP/load/store user (PHI of pointer, function call, bitcast
#      to integer via inttoptr, etc.) — fall back to existing passes.
#   2. For each load/store, compute its byte offset relative to A and the
#      scalar access type.  Bail if the offset isn't a compile-time integer
#      or a simple `base + dyn * stride` expression that the rewriter
#      can reproduce.
#   3. Let T = unique scalar type appearing in all loads/stores.  Bail if
#      types disagree (heterogeneous case — defer to a later phase).
#   4. Verify sizeof(T) divides total alloca size and every byte offset.
#      Bail otherwise.
#   5. Replace alloca with a new `alloca [total/sizeof(T) x T]`.  For each
#      load/store, emit a fresh `gep T, new_alloca, byte_offset/sizeof(T)`
#      and rewrite the load/store pointer.  Erase original GEPs and alloca.

using LLVM
using LLVM: API

"""
    retype_uniform_typed_allocas!(mod::LLVM.Module, dl::LLVM.DataLayout)

For every Function-storage alloca whose loads/stores form a uniform-typed
access pattern, retype the alloca to `[N x T]` and rewrite GEPs accordingly.
Allocas that don't satisfy the invariant (heterogeneous types, escapes,
pointer phis, etc.) are left untouched — the existing byte-pun handling
in prepare_vulkan.jl serves as the fallback.
"""
function retype_uniform_typed_allocas!(mod::LLVM.Module, dl::LLVM.DataLayout)
    for fn in LLVM.functions(mod)
        isempty(LLVM.blocks(fn)) && continue
        retype_function_allocas!(fn, dl)
    end
end

# Walk a value's use chain transitively, collecting every (load/store, byte_offset, access_type)
# rooted at `alloca`.  Returns nothing on bail (use we can't follow / non-GEP root).
struct AccessSite
    inst::LLVM.Instruction       # the load or store
    is_load::Bool
    pointer::LLVM.Value          # the pointer operand of inst (may be the alloca or a GEP)
    byte_offset::Int             # static byte offset from alloca; -1 if dynamic-only
    dyn_index_chain::Vector{Tuple{LLVM.Value, Int}}   # (dynamic_index_value, stride_in_bytes) pairs
    access_type::LLVM.LLVMType   # type of the value loaded/stored
end

struct AllocaInfo
    alloca::LLVM.AllocaInst
    total_bytes::Int
    accesses::Vector{AccessSite}
    geps::Vector{LLVM.GetElementPtrInst}   # all GEPs to be erased after rewrite
end

# Collect uses transitively.  Each pointer-typed instruction in the use chain
# must be a GEP we can interpret.  Returns AllocaInfo or nothing.
function analyze_alloca_uses(alloca::LLVM.AllocaInst, dl::LLVM.DataLayout)
    alloca_ty = LLVM.LLVMType(API.LLVMGetAllocatedType(alloca))
    # Only consider Function-storage allocas — addrspace 0 in opaque-pointer LLVM.
    # The retype pass is sound for any storage class but we scope it conservatively
    # to where we know it helps (MVector / MArray patterns).
    total_bytes = Int(API.LLVMABISizeOfType(dl, alloca_ty))
    total_bytes > 0 || return nothing

    accesses = AccessSite[]
    geps = LLVM.GetElementPtrInst[]

    # BFS over (pointer_value, byte_offset_const, dyn_chain).
    # Each entry represents "this pointer points to base+byte_offset_const+sum(dyn*stride)".
    queue = Tuple{LLVM.Value, Int, Vector{Tuple{LLVM.Value, Int}}}[(alloca, 0, Tuple{LLVM.Value,Int}[])]
    visited = Set{LLVM.Value}()

    while !isempty(queue)
        ptr, base_off, dyn_chain = popfirst!(queue)
        ptr in visited && continue
        push!(visited, ptr)

        for use in LLVM.uses(ptr)
            user = LLVM.user(use)
            if user isa LLVM.LoadInst
                # Pointer must be the use's operand 0
                LLVM.operands(user)[1] === ptr || return nothing
                push!(accesses, AccessSite(user, true, ptr, base_off, copy(dyn_chain),
                                            LLVM.value_type(user)))
            elseif user isa LLVM.StoreInst
                # Pointer is operand 1, value is operand 0
                ops = LLVM.operands(user)
                ops[2] === ptr || continue   # value-operand use, not a pointer use
                push!(accesses, AccessSite(user, false, ptr, base_off, copy(dyn_chain),
                                            LLVM.value_type(ops[1])))
            elseif user isa LLVM.GetElementPtrInst
                # Compute the GEP's offset relative to current ptr
                gep_const_off, gep_dyn = analyze_gep_offset(user, dl)
                gep_const_off === nothing && return nothing
                push!(geps, user)
                new_dyn = copy(dyn_chain)
                append!(new_dyn, gep_dyn)
                push!(queue, (user, base_off + gep_const_off, new_dyn))
            elseif user isa LLVM.CallInst
                # Allow llvm.lifetime.{start,end} and llvm.memset/memcpy intrinsics
                # only if they don't create new pointer aliases we can't follow.
                # llvm.lifetime.* is a no-op for our purposes.
                fn = LLVM.called_operand(user)
                if fn isa LLVM.Function
                    name = LLVM.name(fn)
                    if startswith(name, "llvm.lifetime.")
                        continue   # ignore lifetime markers
                    end
                end
                return nothing   # unknown call — bail
            else
                return nothing   # phi / bitcast / inttoptr / etc. — bail conservatively
            end
        end
    end

    return AllocaInfo(alloca, total_bytes, accesses, geps)
end

# Decompose a GEP's index chain into (constant_byte_offset, dynamic_chain).
# Returns (Int, Vector{(Value, Int)}) on success, (nothing, _) on bail.
function analyze_gep_offset(gep::LLVM.GetElementPtrInst, dl::LLVM.DataLayout)
    src_ty = LLVM.LLVMType(API.LLVMGetGEPSourceElementType(gep))
    ops = LLVM.operands(gep)
    n_ops = length(ops)
    n_ops >= 2 || return (nothing, Tuple{LLVM.Value,Int}[])

    const_off = 0
    dyn_chain = Tuple{LLVM.Value,Int}[]
    current_ty = src_ty

    for i in 2:n_ops
        idx = ops[i]
        if i == 2
            # First index: stepping through `current_ty`-sized blocks
            stride = Int(API.LLVMABISizeOfType(dl, current_ty))
            if idx isa LLVM.ConstantInt
                const_off += Int(convert(Int64, idx)) * stride
            else
                push!(dyn_chain, (idx, stride))
            end
            # Type doesn't change — first index is array-like stride
        else
            # Subsequent indices: drill into struct/array
            if current_ty isa LLVM.StructType
                idx isa LLVM.ConstantInt || return (nothing, dyn_chain)
                fi = Int(convert(Int64, idx)) + 1
                elems = LLVM.elements(current_ty)
                1 <= fi <= length(elems) || return (nothing, dyn_chain)
                # Compute member offset
                member_off = Int(API.LLVMOffsetOfElement(dl, current_ty, fi - 1))
                const_off += member_off
                current_ty = elems[fi]
            elseif current_ty isa LLVM.ArrayType
                stride = Int(API.LLVMABISizeOfType(dl, LLVM.eltype(current_ty)))
                if idx isa LLVM.ConstantInt
                    const_off += Int(convert(Int64, idx)) * stride
                else
                    push!(dyn_chain, (idx, stride))
                end
                current_ty = LLVM.eltype(current_ty)
            else
                return (nothing, dyn_chain)   # can't drill into scalar
            end
        end
    end

    return (const_off, dyn_chain)
end

# Helper: is this LLVM type a scalar we can use as the alloca's element type?
@inline function is_retypeable_scalar(T::LLVM.LLVMType)
    T isa LLVM.IntegerType ||
    T isa LLVM.LLVMFloat ||
    T isa LLVM.LLVMDouble ||
    T isa LLVM.LLVMHalf
end

# Pick a scalar type T satisfying:
#   - every access type is a scalar (int / float / half / double)
#   - T is the SMALLEST scalar size appearing in accesses (smallest stride wins)
#   - sizeof(T) divides total_bytes
#   - sizeof(T) divides every static byte offset
#   - every dynamic stride in any access's dyn_chain is a multiple of sizeof(T)
#   - every access's type-size is a multiple of sizeof(T) (so wider accesses
#     decompose cleanly via OpBitcast at access sites)
#
# When all accesses share one type, T = that type — no bitcasts needed.  When
# accesses mix sizes (e.g. LLVM combined 3 i32 stores into i64 + i32), T = the
# smallest, and the rewrite emits OpBitcast pointer at wider-access sites.
function pick_uniform_type(accesses::Vector{AccessSite}, total_bytes::Int, dl::LLVM.DataLayout)
    isempty(accesses) && return nothing

    # All access types must be scalars
    for a in accesses
        is_retypeable_scalar(a.access_type) || return nothing
    end

    # Find the smallest access type by size; use that as T.
    smallest_T = accesses[1].access_type
    smallest_sz = Int(API.LLVMABISizeOfType(dl, smallest_T))
    for a in accesses
        sz_a = Int(API.LLVMABISizeOfType(dl, a.access_type))
        if sz_a < smallest_sz
            smallest_T = a.access_type
            smallest_sz = sz_a
        end
    end
    smallest_sz > 0 || return nothing

    # When sizes are equal, prefer the type that appears MOST FREQUENTLY in
    # the access pattern.  For an MVector{N, Vec3f} where most accesses are
    # float (component reads/writes) and only a few are i32 (from LLVM's
    # bitcast-store optimization), picking float minimizes the number of
    # OpBitcast pointer cast sites at the SPIR-V emit.
    type_counts = Dict{LLVM.LLVMType, Int}()
    for a in accesses
        sz_a = Int(API.LLVMABISizeOfType(dl, a.access_type))
        if sz_a == smallest_sz
            type_counts[a.access_type] = get(type_counts, a.access_type, 0) + 1
        end
    end
    if !isempty(type_counts)
        best_count = maximum(values(type_counts))
        for (T, cnt) in type_counts
            if cnt == best_count
                smallest_T = T
                break
            end
        end
    end

    sz = smallest_sz
    total_bytes % sz == 0 || return nothing

    # Verify divisibility constraints.  Strides that aren't multiples of sz
    # are OK — the rewrite computes total byte offset and divides by sz at the
    # end, which is a single shift for power-of-2 sz.
    debug = get(ENV, "LAVA_DEBUG_RETYPE", "") == "1"
    for a in accesses
        a_sz = Int(API.LLVMABISizeOfType(dl, a.access_type))
        if a_sz % sz != 0
            debug && @info "retype: pick BAIL access_size_not_multiple" a_ty=string(a.access_type) a_sz sz
            return nothing
        end
        if a.byte_offset >= 0 && a.byte_offset % sz != 0
            debug && @info "retype: pick BAIL offset_not_aligned" off=a.byte_offset sz
            return nothing
        end
    end

    # We require sz to be a power of 2 so the byte-offset → element-idx divide
    # is a single LLVM `lshr`.  Power-of-2 covers all real Julia scalar types
    # (i1/i8/i16/i32/i64, f16/f32/f64).
    is_pow2 = sz > 0 && (sz & (sz - 1)) == 0
    is_pow2 || return nothing

    return smallest_T
end

# Rewrite the alloca and all its accesses to use the new typed form.
function rewrite_alloca!(info::AllocaInfo, T::LLVM.LLVMType, dl::LLVM.DataLayout)
    sz = Int(API.LLVMABISizeOfType(dl, T))
    n_elems = info.total_bytes ÷ sz
    new_arr_ty = LLVM.ArrayType(T, n_elems)

    # Skip if the alloca already has the desired type.
    current_ty = LLVM.LLVMType(API.LLVMGetAllocatedType(info.alloca))
    if current_ty == new_arr_ty
        return false
    end

    # Insert new alloca right before the old one
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, info.alloca)
        # Inherit alignment from old alloca
        old_align = LLVM.alignment(info.alloca)
        new_alloca = LLVM.alloca!(builder, new_arr_ty)
        LLVM.alignment!(new_alloca, old_align)
        # Copy over name for debug readability
        old_name = LLVM.name(info.alloca)
        if !isempty(old_name)
            LLVM.name!(new_alloca, old_name * "_retyped")
        end

        # For each access, build a fresh GEP from new_alloca and rewrite the pointer.
        # Strategy: compute total BYTE offset (constant + sum(dyn * stride)) at
        # the access site, then shift right by log2(sz) to convert to an
        # element index.  InstCombine will fold (idx * stride) >> log2(sz) into
        # (idx * (stride / sz)) when stride is a multiple of sz, and into a
        # cheap shift when stride == 1.
        sz_log2 = trailing_zeros(sz)
        i64 = LLVM.Int64Type()
        zero32 = LLVM.ConstantInt(LLVM.Int32Type(), 0)

        # Helper: build the byte-offset expression for an access site.
        function build_byte_offset(site::AccessSite)
            byte_offset_expr = LLVM.ConstantInt(i64, site.byte_offset)
            for (dyn, stride) in site.dyn_index_chain
                dyn_i64 = if LLVM.value_type(dyn) isa LLVM.IntegerType &&
                             LLVM.width(LLVM.value_type(dyn)) < 64
                    LLVM.sext!(builder, dyn, i64)
                elseif LLVM.value_type(dyn) isa LLVM.IntegerType &&
                       LLVM.width(LLVM.value_type(dyn)) > 64
                    LLVM.trunc!(builder, dyn, i64)
                else
                    dyn
                end
                if stride == 1
                    byte_offset_expr = LLVM.add!(builder, byte_offset_expr, dyn_i64)
                else
                    stride_const = LLVM.ConstantInt(i64, stride)
                    scaled = LLVM.mul!(builder, dyn_i64, stride_const)
                    byte_offset_expr = LLVM.add!(builder, byte_offset_expr, scaled)
                end
            end
            return byte_offset_expr
        end

        # Helper: emit `gep T, new_alloca, [0, byte_offset >> log2(sz)]`.
        function emit_gep_at_byte_offset(byte_offset_expr)
            idx_value = if sz_log2 == 0
                byte_offset_expr
            else
                shift_const = LLVM.ConstantInt(i64, sz_log2)
                LLVM.ashr!(builder, byte_offset_expr, shift_const)
            end
            return LLVM.gep!(builder, new_arr_ty, new_alloca, LLVM.Value[zero32, idx_value])
        end

        # Strategy:
        # - When access size == sz (T's size): rewrite the load/store pointer
        #   to a fresh GEP at the right byte offset.
        # - When access size > sz (wider, e.g. i64 store on [N x i32] alloca):
        #   DECOMPOSE into multiple T-sized accesses.  This avoids OpBitcast
        #   on Function-storage pointers, which is invalid SPIR-V under logical
        #   addressing.
        for site in info.accesses
            LLVM.position!(builder, site.inst)
            access_sz = Int(API.LLVMABISizeOfType(dl, site.access_type))
            byte_offset_expr = build_byte_offset(site)

            if access_sz == sz
                # Same-size: rewrite pointer in place.
                new_gep = emit_gep_at_byte_offset(byte_offset_expr)
                ptr_op_idx = site.is_load ? 1 : 2
                API.LLVMSetOperand(site.inst, UInt32(ptr_op_idx - 1), new_gep)
            else
                # Wider access: decompose into n_parts T-sized accesses.
                @assert access_sz % sz == 0
                n_parts = access_sz ÷ sz
                t_bw = sz * 8   # bit width of T (only int handled for now)
                @assert site.access_type isa LLVM.IntegerType
                @assert T isa LLVM.IntegerType   # decomposition currently only for int → int
                t_ty = T

                # Compute n_parts byte offsets and matching GEPs.
                geps = LLVM.Value[]
                for p in 0:(n_parts - 1)
                    p_off = if p == 0
                        byte_offset_expr
                    else
                        LLVM.add!(builder, byte_offset_expr, LLVM.ConstantInt(i64, p * sz))
                    end
                    push!(geps, emit_gep_at_byte_offset(p_off))
                end

                if site.is_load
                    # Load each T-sized chunk, zext + shift + or to assemble the wide value.
                    wide_ty = site.access_type
                    parts = LLVM.Value[]
                    for g in geps
                        v = LLVM.load!(builder, t_ty, g)
                        push!(parts, v)
                    end
                    # Combine: result = sum(parts[i] zext to wide << (i*t_bw))
                    combined = LLVM.zext!(builder, parts[1], wide_ty)
                    for i in 2:n_parts
                        ext = LLVM.zext!(builder, parts[i], wide_ty)
                        shift_const = LLVM.ConstantInt(wide_ty, (i - 1) * t_bw)
                        shifted = LLVM.shl!(builder, ext, shift_const)
                        combined = LLVM.or!(builder, combined, shifted)
                    end
                    LLVM.replace_uses!(site.inst, combined)
                    LLVM.erase!(site.inst)
                else
                    # Store: split the wide value into n_parts T-chunks, store each.
                    val = LLVM.operands(site.inst)[1]   # value being stored
                    wide_ty = site.access_type
                    for i in 1:n_parts
                        chunk = if i == 1 && t_bw == LLVM.width(wide_ty)
                            val   # no truncation needed
                        else
                            shifted = if i == 1
                                val
                            else
                                shift_const = LLVM.ConstantInt(wide_ty, (i - 1) * t_bw)
                                LLVM.lshr!(builder, val, shift_const)
                            end
                            if LLVM.width(wide_ty) > t_bw
                                LLVM.trunc!(builder, shifted, t_ty)
                            else
                                shifted
                            end
                        end
                        LLVM.store!(builder, chunk, geps[i])
                    end
                    LLVM.erase!(site.inst)
                end
            end
        end
    end

    # Erase the old GEPs in REVERSE BFS order (children first, then parents).
    # Forward order leaves parent GEPs with their child-GEP uses still attached,
    # so parents never reach `isempty(uses)` and stay in the IR — leaving the
    # OLD alloca live with zombie users that confuse downstream analysis.
    for gep in reverse(info.geps)
        if isempty(LLVM.uses(gep))
            LLVM.erase!(gep)
        end
    end
    if isempty(LLVM.uses(info.alloca))
        LLVM.erase!(info.alloca)
    end

    return true
end

function retype_function_allocas!(fn::LLVM.Function, dl::LLVM.DataLayout)
    # Allocas live in the entry block by Julia/LLVM convention
    entry = first(LLVM.blocks(fn))
    candidates = LLVM.AllocaInst[]
    for inst in LLVM.instructions(entry)
        inst isa LLVM.AllocaInst || continue
        push!(candidates, inst)
    end

    debug = get(ENV, "LAVA_DEBUG_RETYPE", "") == "1"
    for alloca in candidates
        info = analyze_alloca_uses(alloca, dl)
        if info === nothing
            debug && @info "retype: BAIL analyze_uses" name=LLVM.name(alloca)
            continue
        end
        T = pick_uniform_type(info.accesses, info.total_bytes, dl)
        if T === nothing
            if debug
                type_strings = unique(string(a.access_type) for a in info.accesses)
                @info "retype: BAIL non-uniform" name=LLVM.name(alloca) type_strings n=length(info.accesses)
            end
            continue
        end
        if rewrite_alloca!(info, T, dl)
            debug && @info "retype: REWROTE" name=LLVM.name(alloca) T
        end
    end
end
