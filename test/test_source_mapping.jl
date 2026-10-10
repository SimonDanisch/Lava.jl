# Source maps (SPIR-V result id → Julia file and line) and the compile errors
# built from GPUCompiler's `InvalidIRError` (`LavaCompilationError`).
#
# Covers:
# - Source map population during SPIR-V emission
# - Inlined-at chain walking (user code, not Base internals)
# - Complex kernels (structs, control flow, math)
# - File-based kernels (real file paths)
# - LavaCompilationError for common GPU-incompatible patterns, on `lava_compile`
#   and on the job path a launch compiles through
# - Call chains through user helpers, deduplicated
#
# Moved here from Mantle's `test/vulkan/test_source_mapping.jl`. Its two device
# testsets did not move: "OOM validation messages don't leak" is Mantle's
# `test/vulkan/test_tolerated_alloc_failure.jl`, which asserts the same thing
# with the ring drained first; "source mapping doesn't break kernel execution"
# ran a broadcast, a `sum` and a struct kernel, and every compile builds a source
# map, so every kernel test in Mantle runs that path.

using Test, Lava
using Lava: LavaDeviceArray, lava_compile, CompilationResult,
            LavaCompilationError, shorten_path

# Simple kernel: load, compute, store
function srcmap_add!(A::LavaDeviceArray{Float32,1}, val::Float32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds A[i] = A[i] + val
    return nothing
end

# Complex kernel: multiple ops, control flow
function srcmap_complex!(A::LavaDeviceArray{Float32,1}, B::LavaDeviceArray{Float32,1},
                         scale::Float32, threshold::Float32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds begin
        x = A[i]
        y = B[i]
        result = x * scale + y * (1.0f0 - scale)
        if result > threshold
            A[i] = result
        else
            A[i] = threshold
        end
    end
    return nothing
end

# Struct kernel: NTuple/struct field access
struct SrcMapVec3
    x::Float32
    y::Float32
    z::Float32
end

function srcmap_struct!(dst::LavaDeviceArray{SrcMapVec3,1},
                        src::LavaDeviceArray{SrcMapVec3,1},
                        scale::Float32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds begin
        v = src[i]
        dst[i] = SrcMapVec3(v.x * scale, v.y * scale, v.z * scale)
    end
    return nothing
end

# Integer kernel: different types
function srcmap_int!(A::LavaDeviceArray{Int32,1}, mask::Int32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds A[i] = A[i] & mask
    return nothing
end

# Multi-array kernel
function srcmap_multi!(A::LavaDeviceArray{Float32,1}, B::LavaDeviceArray{Float32,1},
                       C::LavaDeviceArray{Float32,1})
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds C[i] = A[i] * B[i]
    return nothing
end

# ── Intentionally broken kernels for error testing ──

# Heap allocation (rand uses RNG state allocation)
function srcmap_bad_alloc!(A::LavaDeviceArray{Float32,1})
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    x = rand()
    @inbounds A[i] = Float32(x)
    return nothing
end

# The same, compiled only through the launch path (`compile_or_lookup`).
function srcmap_launch_bad_alloc!(A::LavaDeviceArray{Float32,1})
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    x = rand()
    @inbounds A[i] = Float32(x)
    return nothing
end

# Type instability (Any element type → dynamic dispatch)
function srcmap_bad_unstable!(A::LavaDeviceArray{Any,1})
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds A[i] = A[i] * 2
    return nothing
end

# Non-const global access
srcmap_mutable_global = 42
function srcmap_bad_global!(A::LavaDeviceArray{Float32,1})
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds A[i] = Float32(srcmap_mutable_global)
    return nothing
end

# ── Deep call chain error patterns ──

# Pattern A: String interpolation buried 3 levels deep in physics simulation
struct SrcMapParticle
    x::Float32; y::Float32; vx::Float32; vy::Float32
end

function srcmap_update_velocity(p::SrcMapParticle, dt::Float32)
    SrcMapParticle(p.x + p.vx * dt, p.y + p.vy * dt, p.vx, p.vy)
end

function srcmap_check_collision(p::SrcMapParticle)
    if p.x < 0.0f0
        msg = "collision at $(p.x)"  # String interpolation → heap alloc
        return length(msg)
    end
    return Int(0)
end

function srcmap_physics_step(p::SrcMapParticle, dt::Float32)
    p2 = srcmap_update_velocity(p, dt)
    n = srcmap_check_collision(p2)
    return SrcMapParticle(p2.x, p2.y, p2.vx * (1.0f0 - Float32(n) * 0.01f0), p2.vy)
end

function srcmap_deep_physics!(particles::LavaDeviceArray{SrcMapParticle,1}, dt::Float32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds particles[i] = srcmap_physics_step(particles[i], dt)
    return nothing
end

# Pattern B: Abstract field accessed 4 levels deep in a scene renderer
struct SrcMapSceneObject
    transform::NTuple{4, Float32}
    material_id::Int32
    data  # Any-typed — the bug
end

function srcmap_get_albedo(obj::SrcMapSceneObject)
    d = obj.data  # type unstable — d is Any
    return d * 0.5f0
end

function srcmap_compute_lighting(obj::SrcMapSceneObject, light_dir::Float32)
    albedo = srcmap_get_albedo(obj)
    return albedo * max(0.0f0, light_dir)
end

function srcmap_render_pixel(obj::SrcMapSceneObject, uv::Float32)
    light = sin(uv * 3.14159f0)
    return srcmap_compute_lighting(obj, light)
end

function srcmap_deep_scene!(out::LavaDeviceArray{Float32,1},
                            objects::LavaDeviceArray{SrcMapSceneObject,1})
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds out[i] = srcmap_render_pixel(objects[i], Float32(i) * 0.01f0)
    return nothing
end

# Pattern C: error() in validation function 3 levels deep (the classic "I'll just
# add a bounds check with a nice error message" mistake)
function srcmap_validate_range(x::Float32)
    if x < 0.0f0 || x > 1.0f0
        error("value out of range: $x")  # String alloc + throw
    end
    return x
end

function srcmap_apply_tonemap(x::Float32, exposure::Float32)
    y = x * exposure
    return srcmap_validate_range(y / (y + 1.0f0))
end

function srcmap_deep_error!(out::LavaDeviceArray{Float32,1},
                            input::LavaDeviceArray{Float32,1},
                            exposure::Float32)
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds out[i] = srcmap_apply_tonemap(input[i], exposure)
    return nothing
end

# Pattern D: Accidentally using a CPU-only function from a helper
# (e.g., calling an IO function for "logging")
function srcmap_normalize_vec(x::Float32, y::Float32, z::Float32)
    len = sqrt(x*x + y*y + z*z)
    return (x/len, y/len, z/len)
end

function srcmap_process_normal(nx::Float32, ny::Float32, nz::Float32)
    n = srcmap_normalize_vec(nx, ny, nz)
    # Accidentally left debug logging in
    @info "normal: $n"  # Allocates + IO
    return n
end

function srcmap_deep_logging!(out::LavaDeviceArray{Float32,1},
                              normals::LavaDeviceArray{NTuple{3,Float32},1})
    i = Lava.lava_global_invocation_id_x() + UInt32(1)
    @inbounds begin
        n = normals[i]
        result = srcmap_process_normal(n[1], n[2], n[3])
        out[i] = result[1] + result[2] + result[3]
    end
    return nothing
end

@testset "source maps and compile errors" begin

# ═══════════════════════════════════════════════════════════════════════
# Source map is populated for a simple kernel
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: simple kernel has mapped instructions" begin
    r = lava_compile(srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    @test r isa CompilationResult
    @test hasfield(CompilationResult, :source_map)
    @test r.source_map isa Dict{UInt32, Tuple{String, Int}}
    # A simple add kernel should produce at least 10 mapped instructions
    @test length(r.source_map) >= 10
end

# ═══════════════════════════════════════════════════════════════════════
# Source map points to user code, not Base internals
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: points to user code (inlined_at chain)" begin
    r = lava_compile(srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    # Collect all unique source files
    files = Set{String}()
    for (_, (file, _)) in r.source_map
        push!(files, file)
    end

    # Should NOT point to Base internals like pointer.jl, float.jl
    # (the inlined_at chain walking should find user code)
    for f in files
        @test !endswith(f, "pointer.jl")
        @test !endswith(f, "float.jl")
        @test !endswith(f, "int.jl")
        @test !endswith(f, "boot.jl")
    end

    # All entries should have non-zero line numbers
    for (id, (file, line)) in r.source_map
        @test line > 0
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Source map covers multiple source lines
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: complex kernel maps to multiple source lines" begin
    r = lava_compile(srcmap_complex!,
        Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, Float32, Float32})

    # Group by line number
    lines_seen = Set{Int}()
    for (_, (_, line)) in r.source_map
        push!(lines_seen, line)
    end

    # A complex kernel with branches should map to at least 4 different lines
    @test length(lines_seen) >= 4

    # Should have more mapped instructions than the simple kernel
    @test length(r.source_map) >= 20
end

# ═══════════════════════════════════════════════════════════════════════
# Source map works for struct kernels
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: struct kernel" begin
    r = lava_compile(srcmap_struct!,
        Tuple{LavaDeviceArray{SrcMapVec3,1}, LavaDeviceArray{SrcMapVec3,1}, Float32})

    @test length(r.source_map) >= 15

    # Verify entries have valid data
    for (id, (file, line)) in r.source_map
        @test !isempty(file)
        @test line > 0
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Source map works for integer kernels
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: integer kernel" begin
    r = lava_compile(srcmap_int!,
        Tuple{LavaDeviceArray{Int32,1}, Int32})

    @test length(r.source_map) >= 5
end

# ═══════════════════════════════════════════════════════════════════════
# Source map for multi-array kernel
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: multi-array kernel" begin
    r = lava_compile(srcmap_multi!,
        Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}})

    @test length(r.source_map) >= 10
end

# ═══════════════════════════════════════════════════════════════════════
# Source map with file-based kernel (real file paths)
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: file-based kernel has real file paths" begin
    # Write a kernel to a real file
    test_file = tempname() * "_srcmap_test.jl"
    write(test_file, """
    module SrcMapFileKernel
    using Lava: LavaDeviceArray, lava_global_invocation_id_x

    function file_kernel!(A::LavaDeviceArray{Float32,1}, B::LavaDeviceArray{Float32,1})
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds begin
            a = A[i]
            b = B[i]
            A[i] = a + b
        end
        return nothing
    end

    end
    """)

    try
        # `include` returns the module the file defines.
        filemod = include(test_file)
        r = lava_compile(filemod.file_kernel!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}})

        # Check that source map contains real file paths
        has_real_path = false
        for (_, (file, _)) in r.source_map
            if occursin("_srcmap_test.jl", file)
                has_real_path = true
                break
            end
        end
        @test has_real_path

        # Verify the mapped lines make sense (lines 5-11 of the test file)
        lines_in_file = Set{Int}()
        for (_, (file, line)) in r.source_map
            if occursin("_srcmap_test.jl", file)
                push!(lines_in_file, line)
            end
        end
        @test !isempty(lines_in_file)
        # Lines should be in the range of our function definition (roughly 5-12)
        @test minimum(lines_in_file) >= 4
        @test maximum(lines_in_file) <= 15
    finally
        rm(test_file; force=true)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# SPIR-V disassembly can be annotated with source map
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: SPIR-V disassembly annotations" begin
    r = lava_compile(srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    # Parse SPIR-V disassembly for IDs and check they can be annotated
    annotated_count = 0
    for line in split(r.spirv_disasm, '\n')
        m = match(r"^\s*%(\d+)\b", line)
        m === nothing && continue
        id = parse(UInt32, m.captures[1])
        if haskey(r.source_map, id)
            annotated_count += 1
        end
    end
    # At least half of the SPIR-V result IDs should be annotatable
    @test annotated_count >= 5
end

# ═══════════════════════════════════════════════════════════════════════
# Source map IDs are valid SPIR-V result IDs
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: IDs are valid SPIR-V result IDs" begin
    r = lava_compile(srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    # Parse all result IDs from SPIR-V disassembly
    valid_ids = Set{UInt32}()
    for line in split(r.spirv_disasm, '\n')
        m = match(r"^\s*%(\d+)\s*=", line)
        m !== nothing && push!(valid_ids, parse(UInt32, m.captures[1]))
    end

    # Most source map IDs should be valid SPIR-V result IDs.
    # Some IDs may be for non-result instructions (OpStore, OpBranch, etc.)
    # which don't appear as %id = in disassembly.
    matched = count(id -> id in valid_ids, keys(r.source_map))
    total = length(r.source_map)
    @test total > 0
    @test matched / total >= 0.5  # at least half should be result IDs
end

# ═══════════════════════════════════════════════════════════════════════
# shorten_path works correctly
# ═══════════════════════════════════════════════════════════════════════

@testset "shorten_path" begin
    @test shorten_path("/home/sim/programmieren/VulkanDev/dev/Lava/src/foo.jl") == "Lava/src/foo.jl"
    @test shorten_path("/home/sim/.julia/packages/GPUCompiler/abc/src/bar.jl") == "GPUCompiler/abc/src/bar.jl"
    @test endswith(shorten_path("/usr/share/julia/stdlib/v1.12/Test/src/Test.jl"), "Test.jl")
    @test shorten_path("relative/path.jl") == "path.jl"
end

# ═══════════════════════════════════════════════════════════════════════
# LavaCompilationError for heap allocation
# ═══════════════════════════════════════════════════════════════════════

@testset "Compilation error: heap allocation" begin
    err = try
        lava_compile(srcmap_bad_alloc!, Tuple{LavaDeviceArray{Float32,1}})
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    @test err.operation == "kernel compilation"
    @test occursin("srcmap_bad_alloc!", err.message)
    @test occursin("allocat", lowercase(err.suggestion)) ||
          occursin("heap", lowercase(err.suggestion))
    @test !isempty(err.raw_error)
end

# ═══════════════════════════════════════════════════════════════════════
# LavaCompilationError for type instability
# ═══════════════════════════════════════════════════════════════════════

@testset "Compilation error: type instability" begin
    err = try
        lava_compile(srcmap_bad_unstable!, Tuple{LavaDeviceArray{Any,1}})
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    @test occursin("srcmap_bad_unstable!", err.message)
    @test occursin("instab", lowercase(err.suggestion)) ||
          occursin("dispatch", lowercase(err.suggestion)) ||
          occursin("inferr", lowercase(err.suggestion))
end

# ═══════════════════════════════════════════════════════════════════════
# LavaCompilationError for global variable access
# ═══════════════════════════════════════════════════════════════════════

@testset "Compilation error: global variable access" begin
    err = try
        lava_compile(srcmap_bad_global!, Tuple{LavaDeviceArray{Float32,1}})
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    @test occursin("srcmap_bad_global!", err.message)
    # Should suggest one of: global, const, type instability
    suggestion_lower = lowercase(err.suggestion)
    @test occursin("global", suggestion_lower) ||
          occursin("const", suggestion_lower) ||
          occursin("instab", suggestion_lower) ||
          occursin("dispatch", suggestion_lower)
end

# ═══════════════════════════════════════════════════════════════════════
# LavaCompilationError has proper showerror
# ═══════════════════════════════════════════════════════════════════════

@testset "LavaCompilationError: showerror formatting" begin
    err = try
        lava_compile(srcmap_bad_alloc!, Tuple{LavaDeviceArray{Float32,1}})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError

    # Check that showerror produces readable output
    output = sprint(showerror, err)
    @test occursin("LavaCompilationError", output)
    @test occursin("kernel compilation", output)
    @test occursin("Suggestion:", output)
    @test length(output) > 100  # should have substantial content
end

# ═══════════════════════════════════════════════════════════════════════
# The launch path also produces LavaCompilationError
# ═══════════════════════════════════════════════════════════════════════

# A launch does not call `lava_compile`: the runtime builds the job with
# `lava_kernel_job` and compiles it through `compile_or_lookup`, which reaches
# `lava_compile_gpu_from_job` and wraps the error from the job's signature
# rather than from a function and a tuple type. In Mantle's suite this was a
# KernelAbstractions kernel launched on the Vulkan backend; this is the same
# call without the device.
#
# Two things this path got wrong and the device test did not look at. The
# message named the kernel `DataType` ("Cannot compile DataType(var\"#k\", ...)"):
# it was given the signature type and took the name of ITS type. And the error
# was printed in the frozen world, where a kernel defined after Lava loaded has
# no binding yet, so Julia 1.12 printed "Detected access to binding ... in a
# world prior to its definition world" on stderr; `--depwarn=error` makes that
# read an error, which would replace this one. `srcmap_launch_bad_alloc!` is a
# global defined after Lava loaded, as every user kernel is, and compiled by no
# other testset, so the warning would print here.
@testset "Compilation error: launch path (compile_or_lookup)" begin
    job = Lava.lava_kernel_job(srcmap_launch_bad_alloc!, Tuple{LavaDeviceArray{Float32,1}};
                               workgroup_size = (64, 1, 1))
    err = @test_nowarn try
        Lava.compile_or_lookup(job)
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    # Named as `lava_compile` names it: the kernel, then its argument types.
    @test occursin("srcmap_launch_bad_alloc!(", err.message)
    @test occursin("LavaDeviceArray{Float32, 1})", err.message)
    @test !occursin("DataType", err.message)
    @test occursin("allocat", lowercase(err.suggestion)) ||
          occursin("heap", lowercase(err.suggestion))
end

# ═══════════════════════════════════════════════════════════════════════
# Deep call chain — string interpolation in physics helper (3 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: string interpolation in collision check" begin
    err = try
        lava_compile(srcmap_deep_physics!,
            Tuple{LavaDeviceArray{SrcMapParticle,1}, Float32})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    @test !isempty(err.call_chains)

    # Call chain should show the path: kernel → physics_step → check_collision
    @test occursin("srcmap_deep_physics!", err.call_chains)
    @test occursin("srcmap_physics_step", err.call_chains)
    @test occursin("srcmap_check_collision", err.call_chains)

    # Should identify heap allocation as the problem
    @test occursin("heap", lowercase(err.call_chains)) ||
          occursin("alloc", lowercase(err.call_chains)) ||
          occursin("string", lowercase(err.call_chains))

    # The suggestion should also mention allocation
    @test occursin("allocat", lowercase(err.suggestion)) ||
          occursin("heap", lowercase(err.suggestion))

    # The update_velocity helper is fine — should NOT appear as a problem source
    # (it may appear in the chain path but not as the deepest problematic function)
end

# ═══════════════════════════════════════════════════════════════════════
# Deep call chain — abstract field causing dynamic dispatch (4 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: abstract field in scene renderer" begin
    err = try
        lava_compile(srcmap_deep_scene!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{SrcMapSceneObject,1}})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    @test !isempty(err.call_chains)

    # Call chain should trace through: kernel → render_pixel → compute_lighting → get_albedo
    @test occursin("srcmap_deep_scene!", err.call_chains)
    @test occursin("srcmap_get_albedo", err.call_chains)

    # Should identify type instability / dynamic dispatch
    @test occursin("dispatch", lowercase(err.call_chains)) ||
          occursin("instab", lowercase(err.call_chains))

    @test occursin("instab", lowercase(err.suggestion)) ||
          occursin("dispatch", lowercase(err.suggestion))
end

# ═══════════════════════════════════════════════════════════════════════
# Deep call chain — error() in validation function (3 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: error() in validation helper" begin
    err = try
        lava_compile(srcmap_deep_error!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, Float32})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    @test !isempty(err.call_chains)

    # Should show: kernel → apply_tonemap → validate_range
    @test occursin("srcmap_deep_error!", err.call_chains)
    @test occursin("srcmap_apply_tonemap", err.call_chains)
    @test occursin("srcmap_validate_range", err.call_chains)

    # Should identify allocation (from string interpolation in error())
    @test occursin("alloc", lowercase(err.call_chains)) ||
          occursin("heap", lowercase(err.call_chains))
end

# ═══════════════════════════════════════════════════════════════════════
# Deep call chain — @info logging left in GPU code (3 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: @info logging in helper" begin
    err = try
        lava_compile(srcmap_deep_logging!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{NTuple{3,Float32},1}})
        nothing
    catch e
        e
    end

    # This should fail — @info does IO + allocation
    @test err !== nothing
    # Could be LavaCompilationError or another error type depending on
    # where in the pipeline it fails (GPUCompiler vs emitter)
    if err isa LavaCompilationError
        @test !isempty(err.call_chains) || !isempty(err.raw_error)
        @test occursin("srcmap_process_normal", err.call_chains) ||
              occursin("srcmap_process_normal", err.raw_error)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Call chain deduplication — many reasons, few unique chains
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: deduplication reduces noise" begin
    # The physics kernel triggers ~16 GPUCompiler reasons but only ~2 unique user chains
    err = try
        lava_compile(srcmap_deep_physics!,
            Tuple{LavaDeviceArray{SrcMapParticle,1}, Float32})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError

    # Raw error has many "Reason:" lines (GPUCompiler is verbose)
    raw_reasons = count("Reason:", err.raw_error)
    @test raw_reasons >= 4  # typically 10-16 reasons for string interpolation

    # But call_chains should be compact — far fewer unique chains than raw reasons
    chain_lines = count("Problem:", err.call_chains)
    @test chain_lines >= 1
    @test chain_lines <= raw_reasons  # strictly fewer (deduplication works)

    # Output should be much shorter than raw error
    @test length(err.call_chains) < length(err.raw_error)
end

# ═══════════════════════════════════════════════════════════════════════
# showerror formatting with call chains
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: showerror shows call chains before raw error" begin
    err = try
        lava_compile(srcmap_deep_scene!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{SrcMapSceneObject,1}})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    output = sprint(showerror, err)

    # Call chains should appear BEFORE raw error in the output
    chain_pos = findfirst("Call chain", output)
    suggestion_pos = findfirst("Suggestion:", output)
    raw_pos = findfirst("Raw error", output)

    @test chain_pos !== nothing
    @test suggestion_pos !== nothing
    @test raw_pos !== nothing

    # Order: call chains → suggestion → raw error
    @test first(chain_pos) < first(suggestion_pos)
    @test first(suggestion_pos) < first(raw_pos)
end

end  # @testset "source maps and compile errors"
