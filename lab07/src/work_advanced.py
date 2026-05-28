import taichi as ti

ti.init(arch=ti.gpu, offline_cache=False)

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800

N = 20
NUM_PARTICLES = N * N
MAX_SPRINGS = N * N * 8
STRUCTURAL_SPRINGS = N * (N - 1) * 2

MASS = 1.0
INV_MASS = 1.0 / MASS
DEFAULT_DT = 5e-4
DEFAULT_STIFFNESS = 10000.0
DEFAULT_DAMPING = 1.0
DEFAULT_MAX_VELOCITY = 50.0
DEFAULT_SUBSTEPS = 40
IMPLICIT_ITERATIONS = 3
EPSILON = 1e-6

SHEAR_SCALE = 0.7
BENDING_SCALE = 0.35

SPHERE_CENTER = (0.0, 0.35, 0.0)
SPHERE_RADIUS = 0.28
DEFAULT_COLLISION_MARGIN = 0.05
TANGENTIAL_DAMPING = 0.82

EXPLICIT = 0
SEMI_IMPLICIT = 1
IMPLICIT = 2

gravity = ti.Vector([0.0, -9.8, 0.0])

x = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
is_fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)

x_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
v_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
f_next = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

spring_indices = ti.field(dtype=ti.i32, shape=MAX_SPRINGS * 2)
structural_spring_indices = ti.field(dtype=ti.i32, shape=STRUCTURAL_SPRINGS * 2)
spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_lengths = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
spring_scale = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
num_springs = ti.field(dtype=ti.i32, shape=())
num_structural_springs = ti.field(dtype=ti.i32, shape=())

sphere_center = ti.Vector.field(3, dtype=ti.f32, shape=1)


@ti.func
def vec3(xv: ti.f32, yv: ti.f32, zv: ti.f32):
    return ti.Vector([xv, yv, zv])


@ti.func
def add_spring(idx_a: ti.i32, idx_b: ti.i32, stiffness_scale: ti.f32, is_structural: ti.i32):
    spring = ti.atomic_add(num_springs[None], 1)
    spring_pairs[spring] = ti.Vector([idx_a, idx_b])
    spring_lengths[spring] = (x[idx_a] - x[idx_b]).norm()
    spring_scale[spring] = stiffness_scale

    if is_structural == 1:
        visible = ti.atomic_add(num_structural_springs[None], 1)
        structural_spring_indices[visible * 2] = idx_a
        structural_spring_indices[visible * 2 + 1] = idx_b


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = vec3(i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5)
        v[idx] = vec3(0.0, 0.0, 0.0)
        f[idx] = vec3(0.0, 0.0, 0.0)
        f_next[idx] = vec3(0.0, 0.0, 0.0)

        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

    sphere_center[0] = vec3(SPHERE_CENTER[0], SPHERE_CENTER[1], SPHERE_CENTER[2])


@ti.kernel
def init_springs(enable_shear: ti.i32, enable_bending: ti.i32):
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        if i < N - 1:
            add_spring(idx, (i + 1) * N + j, 1.0, 1)
        if j < N - 1:
            add_spring(idx, i * N + j + 1, 1.0, 1)

        if enable_shear == 1:
            if i < N - 1 and j < N - 1:
                add_spring(idx, (i + 1) * N + j + 1, SHEAR_SCALE, 0)
            if i < N - 1 and j > 0:
                add_spring(idx, (i + 1) * N + j - 1, SHEAR_SCALE, 0)

        if enable_bending == 1:
            if i < N - 2:
                add_spring(idx, (i + 2) * N + j, BENDING_SCALE, 0)
            if j < N - 2:
                add_spring(idx, i * N + j + 2, BENDING_SCALE, 0)


@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]


def init_cloth(enable_shear=True, enable_bending=True):
    num_springs[None] = 0
    num_structural_springs[None] = 0
    init_positions()
    init_springs(1 if enable_shear else 0, 1 if enable_bending else 0)
    init_spring_indices()
    return int(num_springs[None]), int(num_structural_springs[None])


@ti.func
def atomic_add_force(force: ti.template(), idx: ti.i32, value):
    for axis in ti.static(range(3)):
        ti.atomic_add(force[idx][axis], value[axis])


@ti.func
def compute_forces_on(
    pos: ti.template(),
    vel: ti.template(),
    force: ti.template(),
    spring_ks: ti.f32,
    damping_kd: ti.f32,
):
    for i in range(NUM_PARTICLES):
        force[i] = gravity * MASS - damping_kd * vel[i]

    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        delta = pos[idx_a] - pos[idx_b]
        dist = delta.norm()

        if dist > EPSILON:
            direction = delta / dist
            stiffness = spring_ks * spring_scale[i]
            spring_force = -stiffness * (dist - spring_lengths[i]) * direction
            atomic_add_force(force, idx_a, spring_force)
            atomic_add_force(force, idx_b, -spring_force)


@ti.func
def clamp_velocity(vel: ti.template(), idx: ti.i32, max_velocity: ti.f32):
    speed = vel[idx].norm()
    if speed > max_velocity and speed > EPSILON:
        vel[idx] = vel[idx] / speed * max_velocity


@ti.func
def damp_surface_velocity(vel: ti.template(), idx: ti.i32, normal):
    normal_speed = vel[idx].dot(normal)
    if normal_speed < 0.0:
        vel[idx] -= normal_speed * normal

    normal_part = vel[idx].dot(normal) * normal
    tangent_part = vel[idx] - normal_part
    vel[idx] = normal_part + tangent_part * TANGENTIAL_DAMPING


@ti.func
def solve_sphere_collision(
    pos: ti.template(),
    vel: ti.template(),
    idx: ti.i32,
    previous_position,
    enabled: ti.i32,
    collision_margin: ti.f32,
):
    if enabled == 1:
        center = sphere_center[0]
        collision_radius = SPHERE_RADIUS + collision_margin
        previous_offset = previous_position - center
        offset = pos[idx] - center
        dist = offset.norm()

        normal = vec3(0.0, 1.0, 0.0)
        if dist > EPSILON:
            normal = offset / dist

        hit = 0
        if dist < collision_radius:
            hit = 1
        else:
            motion = pos[idx] - previous_position
            motion_len2 = motion.dot(motion)

            if motion_len2 > EPSILON:
                closest_t = -previous_offset.dot(motion) / motion_len2
                closest_t = ti.max(0.0, ti.min(1.0, closest_t))
                closest_offset = previous_offset + closest_t * motion
                closest_dist = closest_offset.norm()

                if closest_dist < collision_radius:
                    hit = 1
                    if closest_dist > EPSILON:
                        normal = closest_offset / closest_dist
                    else:
                        previous_dist = previous_offset.norm()
                        if previous_dist > EPSILON:
                            normal = previous_offset / previous_dist

        if hit == 1:
            pos[idx] = center + normal * collision_radius
            damp_surface_velocity(vel, idx, normal)


@ti.kernel
def step_explicit(
    dt: ti.f32,
    spring_ks: ti.f32,
    damping_kd: ti.f32,
    max_velocity: ti.f32,
    collision_enabled: ti.i32,
    collision_margin: ti.f32,
):
    compute_forces_on(x, v, f, spring_ks, damping_kd)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            previous_position = x[i]
            x[i] += v[i] * dt
            v[i] += f[i] * INV_MASS * dt
            clamp_velocity(v, i, max_velocity)
            solve_sphere_collision(x, v, i, previous_position, collision_enabled, collision_margin)


@ti.kernel
def step_semi_implicit(
    dt: ti.f32,
    spring_ks: ti.f32,
    damping_kd: ti.f32,
    max_velocity: ti.f32,
    collision_enabled: ti.i32,
    collision_margin: ti.f32,
):
    compute_forces_on(x, v, f, spring_ks, damping_kd)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            previous_position = x[i]
            v[i] += f[i] * INV_MASS * dt
            clamp_velocity(v, i, max_velocity)
            x[i] += v[i] * dt
            solve_sphere_collision(x, v, i, previous_position, collision_enabled, collision_margin)


@ti.kernel
def step_implicit_iter(
    dt: ti.f32,
    spring_ks: ti.f32,
    damping_kd: ti.f32,
    max_velocity: ti.f32,
    collision_enabled: ti.i32,
    collision_margin: ti.f32,
):
    for i in range(NUM_PARTICLES):
        v_next[i] = v[i]
        x_next[i] = x[i]

    for _ in ti.static(range(IMPLICIT_ITERATIONS)):
        compute_forces_on(x_next, v_next, f_next, spring_ks, damping_kd)

        for i in range(NUM_PARTICLES):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + f_next[i] * INV_MASS * dt
                clamp_velocity(v_next, i, max_velocity)
                x_next[i] = x[i] + v_next[i] * dt
                solve_sphere_collision(x_next, v_next, i, x[i], collision_enabled, collision_margin)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            v[i] = v_next[i]
            x[i] = x_next[i]


def method_name(method):
    if method == EXPLICIT:
        return "Explicit Euler"
    if method == SEMI_IMPLICIT:
        return "Semi-Implicit Euler"
    return "Implicit Euler"


def step(method, dt, spring_ks, damping_kd, max_velocity, collision_enabled, collision_margin):
    collision_flag = 1 if collision_enabled else 0
    if method == EXPLICIT:
        step_explicit(dt, spring_ks, damping_kd, max_velocity, collision_flag, collision_margin)
    elif method == SEMI_IMPLICIT:
        step_semi_implicit(dt, spring_ks, damping_kd, max_velocity, collision_flag, collision_margin)
    else:
        step_implicit_iter(dt, spring_ks, damping_kd, max_velocity, collision_flag, collision_margin)


def main():
    use_shear = True
    use_bending = True
    collision_enabled = True
    render_all_springs = False
    spring_count, structural_spring_count = init_cloth(use_shear, use_bending)

    window = ti.ui.Window(
        "Lab07 Optional: Mass-Spring Cloth",
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.55, 2.1)
    camera.lookat(0.0, 0.25, 0.0)

    current_method = SEMI_IMPLICIT
    paused = False
    dt = DEFAULT_DT
    spring_ks = DEFAULT_STIFFNESS
    damping_kd = DEFAULT_DAMPING
    max_velocity = DEFAULT_MAX_VELOCITY
    collision_margin = DEFAULT_COLLISION_MARGIN
    substeps = DEFAULT_SUBSTEPS

    while window.running:
        window.GUI.begin("Optional Controls", 0.02, 0.02, 0.42, 0.55)
        window.GUI.text(f"Integration Method: {method_name(current_method)}")
        window.GUI.text(f"Springs: {spring_count}")

        prefix_0 = "[*] " if current_method == EXPLICIT else "[ ] "
        prefix_1 = "[*] " if current_method == SEMI_IMPLICIT else "[ ] "
        prefix_2 = "[*] " if current_method == IMPLICIT else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler"):
            current_method = EXPLICIT
            spring_count, structural_spring_count = init_cloth(use_shear, use_bending)
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
            current_method = SEMI_IMPLICIT
            spring_count, structural_spring_count = init_cloth(use_shear, use_bending)
        if window.GUI.button(prefix_2 + "Implicit Euler"):
            current_method = IMPLICIT
            spring_count, structural_spring_count = init_cloth(use_shear, use_bending)

        next_use_shear = window.GUI.checkbox("Shear springs", use_shear)
        next_use_bending = window.GUI.checkbox("Bending springs", use_bending)
        next_collision_enabled = window.GUI.checkbox("Sphere collision", collision_enabled)
        render_all_springs = window.GUI.checkbox("Render all springs", render_all_springs)
        if next_use_shear != use_shear or next_use_bending != use_bending:
            use_shear = next_use_shear
            use_bending = next_use_bending
            spring_count, structural_spring_count = init_cloth(use_shear, use_bending)
        collision_enabled = next_collision_enabled

        if window.GUI.button("Resume Simulation" if paused else "Pause Simulation"):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            spring_count, structural_spring_count = init_cloth(use_shear, use_bending)

        dt = window.GUI.slider_float("dt", dt, 1e-4, 1e-2)
        substeps = window.GUI.slider_int("substeps", substeps, 1, 80)
        spring_ks = window.GUI.slider_float("k_s", spring_ks, 1000.0, 20000.0)
        damping_kd = window.GUI.slider_float("k_d", damping_kd, 0.0, 8.0)
        max_velocity = window.GUI.slider_float("max velocity", max_velocity, 5.0, 80.0)
        collision_margin = window.GUI.slider_float("collision margin", collision_margin, 0.0, 0.08)
        window.GUI.end()

        if not paused:
            for _ in range(substeps):
                step(
                    current_method,
                    dt,
                    spring_ks,
                    damping_kd,
                    max_velocity,
                    collision_enabled,
                    collision_margin,
                )

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-0.8, 0.8, 0.8), color=(0.35, 0.42, 0.55))

        if collision_enabled:
            scene.particles(sphere_center, radius=SPHERE_RADIUS, color=(0.95, 0.38, 0.22))

        scene.particles(x, radius=0.012, color=(0.2, 0.6, 1.0))
        if render_all_springs:
            scene.lines(
                x,
                indices=spring_indices,
                width=1.0,
                color=(0.82, 0.84, 0.80),
                index_count=spring_count * 2,
            )
        else:
            scene.lines(
                x,
                indices=structural_spring_indices,
                width=1.3,
                color=(0.82, 0.84, 0.80),
                index_count=structural_spring_count * 2,
            )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
