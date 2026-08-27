# Device-side intrinsics for Lava.jl
#
# Thread indexing builtins via llvmcall.
# These produce loads from addrspace(7) globals with __spirv_BuiltIn* names.
# The custom SPIR-V emitter recognizes these globals and emits Input variables
# with appropriate BuiltIn decorations.
#
# Convention: 1-based Julia indices, UInt32 return type, dim argument 1-based.

# ── 3D Builtin Variables ──
# Each produces a load from `<3 x i32>` global, extracting one component.

const BUILTIN_3D = (
    :lava_global_invocation_id => :__spirv_BuiltInGlobalInvocationId,
    :lava_local_invocation_id  => :__spirv_BuiltInLocalInvocationId,
    :lava_workgroup_id         => :__spirv_BuiltInWorkgroupId,
    :lava_num_workgroups       => :__spirv_BuiltInNumWorkgroups,
    :lava_workgroup_size       => :__spirv_BuiltInWorkgroupSize,
)

for (jl_name, spirv_name) in BUILTIN_3D
    gvar = "@$spirv_name"
    ir = """
        $gvar = external addrspace(7) global <3 x i32>
        define i32 @entry(i32 %dim) #0 {
            %vec = load <3 x i32>, ptr addrspace(7) $gvar, align 16
            %x = extractelement <3 x i32> %vec, i32 %dim
            ret i32 %x
        }
        attributes #0 = { alwaysinline }
    """

    # 3D version: lava_foo(dim) where dim is 1-based
    @eval @inline function $jl_name(dim::Integer=1)
        Base.llvmcall(($ir, "entry"), UInt32, Tuple{UInt32}, UInt32(dim - 1))
    end

    # Convenience: lava_foo_x(), lava_foo_y(), lava_foo_z()
    x_name = Symbol(jl_name, :_x)
    y_name = Symbol(jl_name, :_y)
    z_name = Symbol(jl_name, :_z)
    @eval @inline $x_name() = $jl_name(1)
    @eval @inline $y_name() = $jl_name(2)
    @eval @inline $z_name() = $jl_name(3)
end

# ── 1D Builtin: LocalInvocationIndex ──

@inline function lava_local_invocation_index()
    Base.llvmcall(("""
        @__spirv_BuiltInLocalInvocationIndex = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInLocalInvocationIndex, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

# ── 1D Builtins: subgroup size and lane index ──
#
# Both are 0-based hardware values and are deliberately NOT converted to Julia's
# 1-based convention: they are lane *selectors*, and every subgroup op that
# consumes one (shuffle, broadcast, rotate) is specified in terms of the raw
# SPIR-V numbering. Adding one here would make `subgroup_shuffle(v, lane())`
# read the neighbouring lane.
#
# `lava_subgroup_size()` is the real running width, which is 32 on this NVIDIA
# card but 32 *or* 64 on RDNA3 depending on how the driver compiled the shader.
# Nothing may hard-code it — see the `lane ÷ 32` note at COOPMAT_SUBGROUP.

@inline function lava_subgroup_size()
    Base.llvmcall(("""
        @__spirv_BuiltInSubgroupSize = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInSubgroupSize, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

# The rest of the subgroup set `KernelInterface` declares, minus SubgroupMaxSize:
# spirv-val rejects that one under the Kernel (OpenCL) capability, so Vulkan
# cannot have it device-side at all — `subgroup_size_control(ctx).max` is the
# host-side answer. Both of these are compute builtins carrying
# Cap.GroupNonUniform (see SPIRV_BUILTIN_CAPABILITY). Values are raw and 0-based
# where the builtin is an index; KernelInterface's 1-based convention is applied
# at its own boundary.
for (jl_name, spirv_name) in ((:lava_num_subgroups, :__spirv_BuiltInNumSubgroups),
                              (:lava_subgroup_id,   :__spirv_BuiltInSubgroupId))
    ir = """
        @$(spirv_name) = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @$(spirv_name), align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """
    @eval @inline function $jl_name()
        Base.llvmcall(($ir, "entry"), UInt32, Tuple{})
    end
end

@inline function lava_subgroup_local_id()
    Base.llvmcall(("""
        @__spirv_BuiltInSubgroupLocalInvocationId = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInSubgroupLocalInvocationId, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

# ── Workgroup Barrier ──

@inline function lava_workgroup_barrier()
    Base.llvmcall(("""
        declare void @llvm.spv.group.memory.barrier.with.group.sync() #0
        define void @entry() #0 {
            call void @llvm.spv.group.memory.barrier.with.group.sync()
            ret void
        }
        attributes #0 = { convergent nounwind }
    """, "entry"), Cvoid, Tuple{})
end

# Register barrier intrinsic so GPUCompiler doesn't complain
push!(KNOWN_INTRINSICS, "llvm.spv.group.memory.barrier.with.group.sync")
