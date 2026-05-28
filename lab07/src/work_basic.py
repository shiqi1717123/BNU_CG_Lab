import taichi as ti

ti.init(arch=ti.gpu, offline_cache=False)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

N = 20
NUM_PARTICLES = N * N
MAX_SPRINGS = N * N * 4
STRUCTURAL_SPRINGS = N * (N - 1) * 2
SPRING_INDEX_COUNT = STRUCTURAL_SPRINGS * 2

MASS = 1.0
INV_MASS = 1.0 / MASS
DEFAULT_DT = 5e-4
DEFAULT_STIFFNESS = 10000.0
DEFAULT_DAMPING = 1.0
DEFAULT_MAX_VELOCITY = 50.0
DEFAULT_SUBSTEPS = 40
IMPLICIT_ITERATIONS = 3
EPSILON = 1e-6

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
spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_lengths = ti.field(dtype=ti.f32, shape=MAX_SPRINGS)
num_springs = ti.field(dtype=ti.i32, shape=())


@ti.func
def vec3(xv: ti.f32, yv: ti.f32, zv: ti.f32):
    return ti.Vector([xv, yv, zv])


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


@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        if i < N - 1:
            idx_right = (i + 1) * N + j
            spring = ti.atomic_add(num_springs[None], 1)
            spring_pairs[spring] = ti.Vector([idx, idx_right])
            spring_lengths[spring] = (x[idx] - x[idx_right]).norm()

        if j < N - 1:
            idx_down = i * N + j + 1
            spring = ti.atomic_add(num_springs[None], 1)
            spring_pairs[spring] = ti.Vector([idx, idx_down])
            spring_lengths[spring] = (x[idx] - x[idx_down]).norm()


@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]


def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()


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
            spring_force = -spring_ks * (dist - spring_lengths[i]) * direction
            atomic_add_force(force, idx_a, spring_force)
            atomic_add_force(force, idx_b, -spring_force)


@ti.func
def clamp_velocity(vel: ti.template(), idx: ti.i32, max_velocity: ti.f32):
    speed = vel[idx].norm()
    if speed > max_velocity and speed > EPSILON:
        vel[idx] = vel[idx] / speed * max_velocity


@ti.kernel
def step_explicit(dt: ti.f32, spring_ks: ti.f32, damping_kd: ti.f32, max_velocity: ti.f32):
    compute_forces_on(x, v, f, spring_ks, damping_kd)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += f[i] * INV_MASS * dt
            clamp_velocity(v, i, max_velocity)


@ti.kernel
def step_semi_implicit(dt: ti.f32, spring_ks: ti.f32, damping_kd: ti.f32, max_velocity: ti.f32):
    compute_forces_on(x, v, f, spring_ks, damping_kd)

    for i in range(NUM_PARTICLES):
        if is_fixed[i] == 0:
            v[i] += f[i] * INV_MASS * dt
            clamp_velocity(v, i, max_velocity)
            x[i] += v[i] * dt


@ti.kernel
def step_implicit_iter(dt: ti.f32, spring_ks: ti.f32, damping_kd: ti.f32, max_velocity: ti.f32):
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


def step(method, dt, spring_ks, damping_kd, max_velocity):
    if method == EXPLICIT:
        step_explicit(dt, spring_ks, damping_kd, max_velocity)
    elif method == SEMI_IMPLICIT:
        step_semi_implicit(dt, spring_ks, damping_kd, max_velocity)
    else:
        step_implicit_iter(dt, spring_ks, damping_kd, max_velocity)


def main():
    init_cloth()

    window = ti.ui.Window(
        "Lab07 Basic: Mass-Spring System",
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = SEMI_IMPLICIT
    paused = False
    dt = DEFAULT_DT
    spring_ks = DEFAULT_STIFFNESS
    damping_kd = DEFAULT_DAMPING
    max_velocity = DEFAULT_MAX_VELOCITY
    substeps = DEFAULT_SUBSTEPS

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.40, 0.45)
        window.GUI.text(f"Integration Method: {method_name(current_method)}")

        prefix_0 = "[*] " if current_method == EXPLICIT else "[ ] "
        prefix_1 = "[*] " if current_method == SEMI_IMPLICIT else "[ ] "
        prefix_2 = "[*] " if current_method == IMPLICIT else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler"):
            current_method = EXPLICIT
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler"):
            current_method = SEMI_IMPLICIT
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler"):
            current_method = IMPLICIT
            init_cloth()

        if window.GUI.button("Resume Simulation" if paused else "Pause Simulation"):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()

        dt = window.GUI.slider_float("dt", dt, 1e-4, 1e-2)
        substeps = window.GUI.slider_int("substeps", substeps, 1, 80)
        spring_ks = window.GUI.slider_float("k_s", spring_ks, 1000.0, 20000.0)
        damping_kd = window.GUI.slider_float("k_d", damping_kd, 0.0, 8.0)
        max_velocity = window.GUI.slider_float("max velocity", max_velocity, 5.0, 80.0)
        window.GUI.end()

        if not paused:
            for _ in range(substeps):
                step(current_method, dt, spring_ks, damping_kd, max_velocity)

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(
            x,
            indices=spring_indices,
            width=1.5,
            color=(0.8, 0.8, 0.8),
            index_count=SPRING_INDEX_COUNT,
        )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
