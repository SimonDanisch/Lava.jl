using Lava, GeometryBasics, LinearAlgebra
import GLFW

# =============================================================================
# Mesh preparation: flatten indexed mesh to per-vertex arrays
# =============================================================================
function flatten_mesh(mesh)
    f = GeometryBasics.faces(mesh)
    verts = GeometryBasics.coordinates(mesh)
    norms = GeometryBasics.normals(mesh)
    positions = Vec3f[Vec3f(verts[face[j]]) for face in f for j in 1:3]
    normals   = Vec3f[Vec3f(norms[face[j]]) for face in f for j in 1:3]
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

# =============================================================================
# Shaders
# =============================================================================

function mesh_vertex(
        positions::AbstractVector{Vec3f},
        normals::AbstractVector{Vec3f},
        mvp::Mat4f,
    )
    idx = vertex_index()
    @inbounds pos = positions[idx]
    @inbounds n = normals[idx]
    clip = mvp * Vec4f(pos[1], pos[2], pos[3], 1.0f0)
    return (position=clip, normal=n)
end

function mesh_fragment(inputs)
    n = normalize(inputs.normal)
    light_dir = normalize(Vec3f(0.4f0, 0.7f0, 0.5f0))
    diffuse = max(dot(n, light_dir), 0.0f0)
    intensity = 0.15f0 + diffuse * 0.85f0
    return Vec4f(intensity * 0.9f0, intensity * 0.7f0, intensity * 0.5f0, 1.0f0)
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

# Pipeline
# DepthOff because window rendering doesn't have a depth attachment yet.
# CullBack is enough for convex meshes like spheres.
pipeline = Rasterizer(
    vertex = mesh_vertex,
    fragment = mesh_fragment,
    varyings = (normal = Vec3f,),
    cull = CullBack(),
    depth = DepthOff(),
)

# Window
win = RenderWindow(WIDTH, HEIGHT; title="Lava - Mesh Lighting", vsync=true)

# =============================================================================
# Render loop
# =============================================================================

function render_loop(win, pipeline, n_verts, gpu_positions, gpu_normals)
    bq = vk_context().default_bq
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

        # Acquire swapchain image
        acquire_next_image!(win)

        # Draw
        draw!(bq, pipeline, WindowTarget(win), n_verts;
            args = (gpu_positions, gpu_normals, mvp),
            clear_color = (0.1f0, 0.1f0, 0.15f0, 1.0f0),
        )

        # Submit + present
        present_frame!(bq, win)

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

render_loop(win, pipeline, n_verts, gpu_positions, gpu_normals)
