import math

import taichi as ti

ti.init(arch=ti.cpu)

NUM_VERTICES = 3
TRIANGLE_VERTICES = (
    (2.0, 0.0, -2.0),
    (0.0, 2.0, -2.0),
    (-2.0, 0.0, -2.0),
)
EDGE_COLORS = (0xFF0000, 0x00FF00, 0x0000FF)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)


@ti.func
def get_model_matrix(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])


@ti.func
def get_projection_matrix(
    eye_fov: ti.f32,
    aspect_ratio: ti.f32,
    zNear: ti.f32,
    zFar: ti.f32,
):
    n = -zNear
    f = -zFar

    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    persp_to_ortho = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0],
    ])
    ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    ortho_translate = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    return ortho_scale @ ortho_translate @ persp_to_ortho


@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = proj @ view @ model

    for i in range(NUM_VERTICES):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]

        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0


def init_triangle_vertices():
    for i, vertex in enumerate(TRIANGLE_VERTICES):
        vertices[i] = vertex


def main():
    init_triangle_vertices()

    gui = ti.GUI("3D Triangle Transformation (Taichi)", res=(700, 700))
    angle = 0.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "a":
                angle += 10.0
            elif gui.event.key == "d":
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False

        compute_transform(angle)
        gui.clear(0x112240)

        gui.line(screen_coords[0], screen_coords[1], radius=2, color=EDGE_COLORS[0])
        gui.line(screen_coords[1], screen_coords[2], radius=2, color=EDGE_COLORS[1])
        gui.line(screen_coords[2], screen_coords[0], radius=2, color=EDGE_COLORS[2])

        gui.show()


if __name__ == "__main__":
    main()
