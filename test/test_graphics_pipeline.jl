# Tier 3: Graphics Pipeline GPU Execution Tests
#
# Tests the full graphics pipeline on GPU: compilation, draw, readback.
# Uses offscreen framebuffers (no window needed — works on lavapipe/CI).

using Test
using Lava
using Vulkan
using GeometryBasics

# Helper: create offscreen framebuffer, draw, flush, readback
function draw_and_readback(pipeline, vertex_count;
        width=16, height=16, args=(), frag_args=(),
        clear_color=(0f0, 0f0, 0f0, 1f0),
        color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT,
        depth=false, instances=1)
    fb = LavaFramebuffer(width, height; depth, color_format)
    target = OffscreenTarget(fb)
    ctx = Lava.vk_context()
    bq = ctx.default_bq
    draw!(bq, pipeline, target, vertex_count;
        args, frag_args, instances, clear_color)
    Lava.vk_flush!()
    return readback_framebuffer(fb)
end

@testset "Graphics Pipeline" begin

    # ── Basic vertex + fragment ──

    @testset "solid color triangle" begin
        function solid_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            return nothing
        end
        function solid_frag()
            Lava.gfx_output(0, Vec4f(0.0f0, 1.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=solid_vert, fragment=solid_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3)
        @test all(p -> p[2] ≈ 1f0, pixels)  # all green
        @test all(p -> p[4] ≈ 1f0, pixels)  # alpha = 1
    end

    @testset "clear color works" begin
        # Draw zero vertices — only clear should be visible
        function noop_vert()
            Lava.set_position!(Vec4f(0f0, 0f0, 0f0, 1f0))
            return nothing
        end
        function noop_frag()
            Lava.gfx_output(0, Vec4f(0f0, 0f0, 0f0, 0f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=noop_vert, fragment=noop_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 0; clear_color=(0.25f0, 0.5f0, 0.75f0, 1f0))
        @test all(p -> p[1] ≈ 0.25f0 && p[2] ≈ 0.5f0 && p[3] ≈ 0.75f0, pixels)
    end

    # ── BDA arguments ──

    @testset "vertex BDA args" begin
        function bda_vert(positions::Lava.LavaDeviceArray{Vec3f, 1})
            vid = Lava.vertex_index()
            @inbounds p = positions[vid]
            Lava.set_position!(Vec4f(p[1], p[2], p[3], 1.0f0))
            Lava.gfx_output(0, Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        function bda_frag()
            c = Lava.gfx_input(Vec4f, 0)
            Lava.gfx_output(0, c)
            return nothing
        end
        pip = GraphicsPipeline(; vertex=bda_vert, fragment=bda_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        # Fullscreen triangle
        positions = LavaArray(Vec3f[Vec3f(-1,-1,0), Vec3f(3,-1,0), Vec3f(-1,3,0)])
        pixels = draw_and_readback(pip, 3; args=(positions,))
        @test all(p -> p[1] ≈ 1f0, pixels)  # all red
    end

    @testset "vertex-to-fragment varying" begin
        function vary_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            # Pass UV as varying
            u = (x + 1f0) * 0.5f0
            v = (y + 1f0) * 0.5f0
            Lava.gfx_output(0, Vec2f(u, v))
            return nothing
        end
        function vary_frag()
            uv = Lava.gfx_input(Vec2f, 0)
            Lava.gfx_output(0, Vec4f(uv[1], uv[2], 0f0, 1f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=vary_vert, fragment=vary_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3; width=8, height=8)
        # Center pixel should have UV ≈ (0.5, 0.5)
        center = pixels[4, 4]
        @test center[1] ≈ 0.5f0 atol=0.15  # u
        @test center[2] ≈ 0.5f0 atol=0.15  # v
        # Corner (0,0) should have UV near (0, 0)
        @test pixels[1,1][1] < 0.2f0  # u near 0
        @test pixels[1,1][2] < 0.2f0  # v near 0
    end

    @testset "fragment uses frag_coord" begin
        function fc_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            return nothing
        end
        function fc_frag()
            fx = Lava.frag_coord_x()
            fy = Lava.frag_coord_y()
            # Normalize to [0,1] range using 16x16 framebuffer
            Lava.gfx_output(0, Vec4f(fx / 16f0, fy / 16f0, 0f0, 1f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=fc_vert, fragment=fc_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3)
        # Pixel at (8,8) should have frag_coord ≈ (8.5, 8.5) → normalized ≈ (0.53, 0.53)
        p = pixels[8, 8]
        @test 0.4f0 < p[1] < 0.7f0
        @test 0.4f0 < p[2] < 0.7f0
    end

    # ── Blend modes ──

    @testset "alpha blend" begin
        function ab_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            return nothing
        end
        function ab_frag()
            # Semi-transparent red
            Lava.gfx_output(0, Vec4f(1f0, 0f0, 0f0, 0.5f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=ab_vert, fragment=ab_frag,
            blend=AlphaBlend(), cull=NoCull(), depth=DepthOff())
        # Clear to white, draw semi-transparent red
        pixels = draw_and_readback(pip, 3; clear_color=(1f0, 1f0, 1f0, 1f0))
        # Result should be blended: red = 0.5*1 + 0.5*1 = 1, green = 0.5*0 + 0.5*1 = 0.5
        p = pixels[8, 8]
        @test p[1] ≈ 1f0 atol=0.05  # red
        @test p[2] ≈ 0.5f0 atol=0.05  # green (blended)
    end

    # ── Depth test ──

    @testset "depth test" begin
        # Draw two fullscreen triangles: first at z=0.3 (blue), second at z=0.7 (red).
        # With depth < test, closer (z=0.3) blue should win.
        function depth_vert(color::Vec4f, z_val::Float32)
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, z_val, 1.0f0))
            Lava.gfx_output(0, color)
            return nothing
        end
        function depth_frag()
            c = Lava.gfx_input(Vec4f, 0)
            Lava.gfx_output(0, c)
            return nothing
        end
        pip = GraphicsPipeline(; vertex=depth_vert, fragment=depth_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthLess())

        fb = LavaFramebuffer(8, 8; depth=true, color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT)
        target = OffscreenTarget(fb)
        ctx = Lava.vk_context()
        bq = ctx.default_bq

        # Draw far red at z=0.7
        draw!(bq, pip, target, 3;
            args=(Vec4f(1f0, 0f0, 0f0, 1f0), 0.7f0),
            clear_color=(0f0, 0f0, 0f0, 1f0))
        # Draw close blue at z=0.3 (no clear — load previous)
        draw!(bq, pip, target, 3;
            args=(Vec4f(0f0, 0f0, 1f0, 1f0), 0.3f0),
            clear_color=nothing)
        Lava.vk_flush!()

        pixels = readback_framebuffer(fb)
        # Blue should win (closer)
        p = pixels[4, 4]
        @test p[3] ≈ 1f0 atol=0.05  # blue channel
        @test p[1] ≈ 0f0 atol=0.05  # red channel
    end

    # ── Multiple outputs / instances ──

    @testset "instanced drawing" begin
        function inst_vert()
            vid = Lava.vertex_index() - Int32(1)
            iid = Lava.instance_index() - Int32(1)
            # Shift each instance right by 0.5 NDC
            base_x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            base_y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            x = base_x + Float32(iid) * 0.5f0
            Lava.set_position!(Vec4f(x, base_y, 0.5f0, 1.0f0))
            # Color by instance: 0=red, 1=green
            r = iid == Int32(0) ? 1f0 : 0f0
            g = iid == Int32(1) ? 1f0 : 0f0
            Lava.gfx_output(0, Vec4f(r, g, 0f0, 1f0))
            return nothing
        end
        function inst_frag()
            c = Lava.gfx_input(Vec4f, 0)
            Lava.gfx_output(0, c)
            return nothing
        end
        pip = GraphicsPipeline(; vertex=inst_vert, fragment=inst_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3; instances=2, width=32, height=32)
        # Should have some non-black pixels from both instances
        has_red = any(p -> p[1] > 0.5f0, pixels)
        has_green = any(p -> p[2] > 0.5f0, pixels)
        @test has_red
        @test has_green
    end
end
