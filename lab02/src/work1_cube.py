import math

import taichi as ti

ti.init(arch=ti.cpu)

NUM_VERTICES = 8
CUBE_VERTICES = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)
CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
EDGE_COLORS = (
    0xF94144, 0xF3722C, 0xF8961E, 0xF9844A,
    0x90BE6D, 0x43AA8B, 0x4D908E, 0x577590,
    0x277DA1, 0x7B2CBF, 0xC77DFF, 0x4895EF,
)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)


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


@ti.func
def get_cube_model_matrix(angle: ti.f32):
    y_rad = angle * math.pi / 180.0
    x_rad = angle * 0.6 * math.pi / 180.0

    cy = ti.cos(y_rad)
    sy = ti.sin(y_rad)
    cx = ti.cos(x_rad)
    sx = ti.sin(x_rad)

    rotate_y = ti.Matrix([
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    rotate_x = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    return rotate_y @ rotate_x


@ti.kernel
def compute_transform(angle: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_cube_model_matrix(angle)
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


def init_cube_vertices():
    for i, vertex in enumerate(CUBE_VERTICES):
        vertices[i] = vertex


def main():
    init_cube_vertices()

    gui = ti.GUI("3D Cube Transformation (Optional)", res=(700, 700))
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
        gui.clear(0x0B132B)

        for edge_id, (start, end) in enumerate(CUBE_EDGES):
            gui.line(
                screen_coords[start],
                screen_coords[end],
                radius=2,
                color=EDGE_COLORS[edge_id],
            )

        gui.show()


if __name__ == "__main__":
    main()
