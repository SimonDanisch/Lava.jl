# Tier 1: Real kernel compilation tests using actual Hikari/Raycore types
#
# These test that our SPIR-V emitter handles production-grade nested structs,
# complex control flow, and atomic patterns. No GPU dispatch — compilation only.
#
# Requires Hikari + Raycore — skipped if not available (e.g. CI without [sources]).

using Test

# Guard: skip entire file if Hikari/Raycore aren't loadable
const _HAS_HIKARI_RAYCORE = try
    Base.require(Main, :Hikari)
    Base.require(Main, :Raycore)
    true
catch
    false
end

if !_HAS_HIKARI_RAYCORE

@testset "Real Kernel Compilation" begin
    @test_broken false  # mark as known-skipped
end

else  # Hikari + Raycore available

if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc
using Hikari
using Raycore
using GeometryBasics
using StaticArrays

@testset "Real Kernel Compilation" begin

    # ── Test 1: VPRayWorkItem kernel (14 fields) ──
    @testset "VPRayWorkItem — deep struct read/write" begin
        function ray_work_kernel(items, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                item = items[i]
                # Access nested fields: ray, wavelengths, spectral radiance
                ox = item.ray.o[1]
                dx = item.ray.d[1]
                w1 = item.lambda.lambda[1]
                b1 = item.beta.data[1]
                r1 = item.r_u.data[1]
                result = ox * dx + w1 * b1 + r1 + item.eta_scale
                output[i] = result
            end
            return nothing
        end

        WI = Hikari.VPRayWorkItem
        r = Lava.lava_compile(ray_work_kernel,
                              Tuple{Lava.LavaDeviceArray{WI, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 2: VPMediumSampleWorkItem kernel (29 fields — largest work item) ──
    @testset "VPMediumSampleWorkItem — 29-field struct" begin
        function medium_sample_kernel(items, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                item = items[i]
                result = if item.has_surface_hit
                    # Access deep nested hit geometry
                    item.hit_pi[1] + item.hit_n[2] + item.hit_dpdu[3] +
                    item.hit_uv[1] + item.hit_triangle_area
                else
                    item.ray.o[1] + item.t_max + item.eta_scale
                end
                output[i] = result
            end
            return nothing
        end

        WI = Hikari.VPMediumSampleWorkItem
        r = Lava.lava_compile(medium_sample_kernel,
                              Tuple{Lava.LavaDeviceArray{WI, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 3: VPMaterialEvalWorkItem kernel (27 fields with SVector) ──
    @testset "VPMaterialEvalWorkItem — SVector + SetKey" begin
        function material_eval_kernel(items, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                item = items[i]
                # Dot product of shading normal and outgoing direction
                dot_val = item.ns[1] * item.wo[1] +
                          item.ns[2] * item.wo[2] +
                          item.ns[3] * item.wo[3]
                # Access SVector barycentrics
                bary_sum = item.bary[1] + item.bary[2] + item.bary[3]
                output[i] = dot_val + bary_sum + item.eta_scale
            end
            return nothing
        end

        WI = Hikari.VPMaterialEvalWorkItem
        r = Lava.lava_compile(material_eval_kernel,
                              Tuple{Lava.LavaDeviceArray{WI, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 4: Atomic WorkQueue push pattern ──
    @testset "atomic WorkQueue push — Device scope" begin
        function atomic_push_kernel(counter, items, values)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                val = values[i]
                if val > 0.0f0
                    idx = Lava.Atomix.@atomic counter[1] += Int32(1)
                    items[idx + Int32(1)] = val
                end
            end
            return nothing
        end

        r = Lava.lava_compile(atomic_push_kernel,
                              Tuple{Lava.LavaDeviceArray{Int32, 1},
                                    Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        # Verify Device scope atomics (not CrossDevice)
        check(r.spirv_disasm, "OpAtomicIAdd")
        check_not(r.spirv_disasm, "CrossDevice")
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 5: BVH traversal-like kernel (while-loop with break, Mat4f) ──
    @testset "BVH traversal — structured CF stress" begin
        function bvh_traverse_kernel(nodes, rays, hits)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                ray_ox = rays[i]
                # Simulate BVH stack walk with while loop + break
                stack_top = Int32(0)
                best_t = Float32(1.0e10)
                node_idx = Int32(1)
                while node_idx > Int32(0)
                    node_val = nodes[node_idx]
                    # AABB test: simple min/max
                    t_enter = max(node_val - ray_ox, 0.0f0)
                    t_exit = min(node_val + ray_ox, best_t)
                    if t_enter < t_exit
                        if node_val < 0.0f0
                            # Leaf: update best hit
                            best_t = t_enter
                            break
                        else
                            # Internal: descend
                            node_idx = unsafe_trunc(Int32, node_val)
                        end
                    else
                        # Pop stack (simplified)
                        stack_top -= Int32(1)
                        node_idx = stack_top
                    end
                end
                hits[i] = best_t
            end
            return nothing
        end

        r = Lava.lava_compile(bvh_traverse_kernel,
                              Tuple{Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        check(r.spirv_disasm, "OpLoopMerge")
        check(r.spirv_disasm, "OpSelectionMerge")
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 6: Multi-field output kernel (write large structs through BDA) ──
    @testset "large struct write — PSB alignment" begin
        function struct_write_kernel(input, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                val = input[i]
                # Create a VPShadowRayWorkItem and write it
                ray = Raycore.Ray(Point3f(val, 0.0f0, 0.0f0),
                                  Vec3f(0.0f0, 0.0f0, 1.0f0),
                                  0.0f0, 1000.0f0, 0.0f0)
                lambda = Hikari.SampledWavelengths{4}(
                    NTuple{4, Float32}((400.0f0, 500.0f0, 600.0f0, 700.0f0)),
                    NTuple{4, Float32}((0.25f0, 0.25f0, 0.25f0, 0.25f0)))
                ld = Hikari.SampledSpectrum{4}(NTuple{4, Float32}((val, val, val, val)))
                item = Hikari.VPShadowRayWorkItem(
                    ray, 100.0f0, lambda, ld, ld, ld,
                    Int32(i), Raycore.SetKey(UInt32(0), UInt32(0)))
                output[i] = item
            end
            return nothing
        end

        WI = Hikari.VPShadowRayWorkItem
        r = Lava.lava_compile(struct_write_kernel,
                              Tuple{Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{WI, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

end

end  # if _HAS_HIKARI_RAYCORE
