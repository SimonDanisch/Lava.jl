# Mesh Lighting Example
#
# Renders a GeometryBasics mesh with Lambertian directional lighting in a
# window with an orbiting camera. Demonstrates:
#
# - BDA vertex pulling (positions + normals from LavaArrays)
# - MVP transform via column-major matrix stored in a LavaArray
# - Vertex-to-fragment data passing via gfx_output/gfx_input
# - RenderWindow with swapchain, acquire/present loop
# - Per-frame uniform updates (orbiting camera)
#
# No GLSL, no descriptor sets -- just Julia functions as shaders.

using Lava, GeometryBasics, LinearAlgebra
import GLFW

# =============================================================================
# Mesh preparation: flatten indexed mesh to per-vertex arrays
# =============================================================================

function flatten_mesh(mesh)
    faces_list = GeometryBasics.faces(mesh)
    verts = GeometryBasics.coordinates(mesh)
    norms = GeometryBasics.normals(mesh)
    n_tris = length(faces_list)
    n_verts = n_tris * 3
    positions = Vector{Vec3f}(undef, n_verts)
    normals = Vector{Vec3f}(undef, n_verts)
    for (i, face) in enumerate(faces_list)
        for (j, vi) in enumerate(face)
            idx = (i - 1) * 3 + j
            positions[idx] = Vec3f(verts[vi])
            normals[idx] = Vec3f(norms[vi])
        end
    end
    return positions, normals
end

# =============================================================================
# Camera
# =============================================================================

function perspective_mvp(;
    eye = Vec3f(2.5, 2.0, 3.0),
    target = Vec3f(0),
    up = Vec3f(0, 1, 0),
    fov = 60f0,
    aspect = 800f0 / 600f0,
    near = 0.1f0,
    far = 100f0,
)
    f = normalize(target - eye)
    r = normalize(cross(f, up))
    u = cross(r, f)
    view = Mat4f(
        r[1], u[1], -f[1], 0,
        r[2], u[2], -f[2], 0,
        r[3], u[3], -f[3], 0,
        -dot(r, eye), -dot(u, eye), dot(f, eye), 1,
    )
    t = tan(fov * Float32(pi) / 360f0)
    proj = Mat4f(
        1 / (aspect * t), 0, 0, 0,
        0, -1 / t, 0, 0,
        0, 0, far / (near - far), -1,
        0, 0, (near * far) / (near - far), 0,
    )
    return proj * view
end

function update_mvp!(gpu_mvp::LavaArray{Vec4f, 1}, m::Mat4f)
    cols = [Vec4f(m[1, j], m[2, j], m[3, j], m[4, j]) for j in 1:4]
    copyto!(gpu_mvp, cols)
end

# =============================================================================
# Shaders
# =============================================================================

function mesh_vertex(
    positions::Lava.LavaDeviceArray{Vec3f, 1},
    normals::Lava.LavaDeviceArray{Vec3f, 1},
    mvp_cols::Lava.LavaDeviceArray{Vec4f, 1},
)
    idx = vertex_index()
    @inbounds pos = positions[idx]
    @inbounds n = normals[idx]

    @inbounds c0 = mvp_cols[1]
    @inbounds c1 = mvp_cols[2]
    @inbounds c2 = mvp_cols[3]
    @inbounds c3 = mvp_cols[4]

    clip_x = c0[1] * pos[1] + c1[1] * pos[2] + c2[1] * pos[3] + c3[1]
    clip_y = c0[2] * pos[1] + c1[2] * pos[2] + c2[2] * pos[3] + c3[2]
    clip_z = c0[3] * pos[1] + c1[3] * pos[2] + c2[3] * pos[3] + c3[3]
    clip_w = c0[4] * pos[1] + c1[4] * pos[2] + c2[4] * pos[3] + c3[4]

    set_position!(Vec4f(clip_x, clip_y, clip_z, clip_w))
    gfx_output(0, n)
    return nothing
end

function mesh_fragment()
    normal = gfx_input(Vec3f, 0)

    len = sqrt(normal[1] * normal[1] + normal[2] * normal[2] + normal[3] * normal[3])
    len = max(len, 1.0f-6)
    nx = normal[1] / len
    ny = normal[2] / len
    nz = normal[3] / len

    # Directional light
    lx = 0.4f0; ly = 0.7f0; lz = 0.5f0
    ll = sqrt(lx * lx + ly * ly + lz * lz)
    lx /= ll; ly /= ll; lz /= ll

    ndotl = nx * lx + ny * ly + nz * lz
    diffuse = max(ndotl, 0.0f0)
    intensity = 0.15f0 + diffuse * 0.85f0

    gfx_output(0, Vec4f(intensity * 0.9f0, intensity * 0.7f0, intensity * 0.5f0, 1.0f0))
    return nothing
end

# =============================================================================
# Setup
# =============================================================================

WIDTH, HEIGHT = 800, 600

# Mesh
mesh = normal_mesh(Tesselation(Sphere(Point3f(0), 1.0f0), 64))
positions, normals = flatten_mesh(mesh)
n_verts = length(positions)

gpu_positions = LavaArray(positions)
gpu_normals = LavaArray(normals)
gpu_mvp = LavaArray(Vec4f[Vec4f(0) for _ in 1:4])

# Pipeline
# DepthOff because window rendering doesn't have a depth attachment yet.
# CullBack is enough for convex meshes like spheres.
pipeline = Rasterizer(
    vertex = mesh_vertex,
    fragment = mesh_fragment,
    cull = CullBack(),
    depth = DepthOff(),
)

TT_VERT = Tuple{
    Lava.LavaDeviceArray{Vec3f, 1},
    Lava.LavaDeviceArray{Vec3f, 1},
    Lava.LavaDeviceArray{Vec4f, 1},
}

# Window
win = RenderWindow(WIDTH, HEIGHT; title="Lava - Mesh Lighting", vsync=true)

# =============================================================================
# Render loop
# =============================================================================

function render_loop(win, pipeline, n_verts, gpu_positions, gpu_normals, gpu_mvp)
    t0 = time()
    frame = 0

    while isopen(win)
        GLFW.PollEvents()

        # Orbiting camera
        t = Float32(time() - t0)
        radius = 3.5f0
        speed = 0.5f0
        eye = Vec3f(
            radius * cos(t * speed),
            1.0f0 + 0.5f0 * sin(t * speed * 0.7f0),
            radius * sin(t * speed),
        )
        w, h = size(win)
        aspect = Float32(w) / Float32(max(h, 1))
        mvp = perspective_mvp(; eye, aspect, fov=60f0)
        update_mvp!(gpu_mvp, mvp)

        # Acquire swapchain image
        acquire_next_image!(win)

        # Draw
        draw!(pipeline, WindowTarget(win), n_verts;
            args = (gpu_positions, gpu_normals, gpu_mvp),
            tt_vertex = TT_VERT,
            tt_fragment = Tuple{},
            clear_color = (0.1f0, 0.1f0, 0.15f0, 1.0f0),
        )

        # Submit + present
        present_frame!(win)

        frame += 1
        if frame % 300 == 0
            elapsed = time() - t0
            fps = frame / elapsed
            println("Frame $frame  $(round(fps, digits=1)) FPS")
        end
    end

    close(win)
    println("Done. $frame frames rendered.")
end

render_loop(win, pipeline, n_verts, gpu_positions, gpu_normals, gpu_mvp)
