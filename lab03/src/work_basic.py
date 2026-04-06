import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
NUM_SEGMENTS = 1000
NUM_CURVE_POINTS = NUM_SEGMENTS + 1
MAX_CONTROL_POINTS = 100
MAX_POLYGON_LINE_VERTICES = 2 * (MAX_CONTROL_POINTS - 1)

BACKGROUND_COLOR = (1.0, 1.0, 1.0)
CURVE_COLOR = (0.10, 0.72, 0.22)
CONTROL_POINT_COLOR = (0.90, 0.18, 0.18)
CONTROL_POLYGON_COLOR = (0.55, 0.55, 0.55)
CONTROL_POINT_RADIUS = 0.008
CONTROL_POLYGON_WIDTH = 0.002
OFFSCREEN_COORD = -10.0

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WINDOW_WIDTH, WINDOW_HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_CURVE_POINTS)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
polygon_line_points = ti.Vector.field(
    2,
    dtype=ti.f32,
    shape=MAX_POLYGON_LINE_VERTICES,
)


@ti.kernel
def clear_pixels(r: ti.f32, g: ti.f32, b: ti.f32):
    for x, y in pixels:
        pixels[x, y] = ti.Vector([r, g, b])


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        x = ti.cast(curve_points_field[i][0] * (WINDOW_WIDTH - 1), ti.i32)
        y = ti.cast(curve_points_field[i][1] * (WINDOW_HEIGHT - 1), ti.i32)

        if 0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT:
            pixels[x, y] = ti.Vector([CURVE_COLOR[0], CURVE_COLOR[1], CURVE_COLOR[2]])


def de_casteljau(points, t):
    working = np.array(points, dtype=np.float32, copy=True)
    while len(working) > 1:
        working = (1.0 - t) * working[:-1] + t * working[1:]
    return working[0]


def sample_curve(control_points, out_curve_points):
    control_points_np = np.array(control_points, dtype=np.float32)
    for i in range(NUM_CURVE_POINTS):
        t = i / NUM_SEGMENTS
        out_curve_points[i] = de_casteljau(control_points_np, t)


def update_gui_pools(control_points, gui_points_np, polygon_line_points_np):
    gui_points_np.fill(OFFSCREEN_COORD)
    polygon_line_points_np.fill(OFFSCREEN_COORD)

    control_count = len(control_points)
    if control_count == 0:
        gui_points.from_numpy(gui_points_np)
        polygon_line_points.from_numpy(polygon_line_points_np)
        return

    control_points_np = np.array(control_points, dtype=np.float32)
    gui_points_np[:control_count] = control_points_np

    if control_count >= 2:
        segment_count = control_count - 1
        polygon_line_points_np[0 : 2 * segment_count : 2] = control_points_np[:-1]
        polygon_line_points_np[1 : 2 * segment_count : 2] = control_points_np[1:]

    gui_points.from_numpy(gui_points_np)
    polygon_line_points.from_numpy(polygon_line_points_np)


def main():
    window = ti.ui.Window(
        "Lab03 Basic: Bezier Curve",
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()

    control_points = []
    curve_points_np = np.zeros((NUM_CURVE_POINTS, 2), dtype=np.float32)
    gui_points_np = np.full(
        (MAX_CONTROL_POINTS, 2),
        OFFSCREEN_COORD,
        dtype=np.float32,
    )
    polygon_line_points_np = np.full(
        (MAX_POLYGON_LINE_VERTICES, 2),
        OFFSCREEN_COORD,
        dtype=np.float32,
    )

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.LMB and len(control_points) < MAX_CONTROL_POINTS:
                x, y = window.get_cursor_pos()
                control_points.append((float(x), float(y)))
            elif event.key in ("c", "C"):
                control_points.clear()

        clear_pixels(*BACKGROUND_COLOR)
        update_gui_pools(control_points, gui_points_np, polygon_line_points_np)

        if len(control_points) >= 2:
            sample_curve(control_points, curve_points_np)
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_CURVE_POINTS)

        canvas.set_image(pixels)

        if len(control_points) >= 2:
            canvas.lines(
                polygon_line_points,
                width=CONTROL_POLYGON_WIDTH,
                color=CONTROL_POLYGON_COLOR,
            )

        if control_points:
            canvas.circles(
                gui_points,
                radius=CONTROL_POINT_RADIUS,
                color=CONTROL_POINT_COLOR,
            )

        window.show()


if __name__ == "__main__":
    main()
