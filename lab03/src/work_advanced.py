import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
BEZIER_NUM_SEGMENTS = 1000
BSPLINE_SAMPLES_PER_SEGMENT = 64
MAX_CONTROL_POINTS = 100
MAX_POLYGON_LINE_VERTICES = 2 * (MAX_CONTROL_POINTS - 1)
MAX_CURVE_POINTS = max(
    BEZIER_NUM_SEGMENTS + 1,
    (MAX_CONTROL_POINTS - 3) * BSPLINE_SAMPLES_PER_SEGMENT + 1,
)

BACKGROUND_COLOR = (1.0, 1.0, 1.0)
CURVE_COLOR = (0.10, 0.72, 0.22)
CONTROL_POINT_COLOR = (0.90, 0.18, 0.18)
CONTROL_POLYGON_COLOR = (0.55, 0.55, 0.55)
CONTROL_POINT_RADIUS = 0.008
CONTROL_POLYGON_WIDTH = 0.002
OFFSCREEN_COORD = -10.0
AA_SIGMA = 0.55
CURVE_MODE_BEZIER = 0
CURVE_MODE_BSPLINE = 1

BSPLINE_BASIS = np.array(
    [
        [-1.0, 3.0, -3.0, 1.0],
        [3.0, -6.0, 3.0, 0.0],
        [-3.0, 0.0, 3.0, 0.0],
        [1.0, 4.0, 1.0, 0.0],
    ],
    dtype=np.float32,
) / 6.0

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WINDOW_WIDTH, WINDOW_HEIGHT))
coverage = ti.field(dtype=ti.f32, shape=(WINDOW_WIDTH, WINDOW_HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
polygon_line_points = ti.Vector.field(
    2,
    dtype=ti.f32,
    shape=MAX_POLYGON_LINE_VERTICES,
)


@ti.kernel
def clear_buffers(r: ti.f32, g: ti.f32, b: ti.f32):
    for x, y in pixels:
        pixels[x, y] = ti.Vector([r, g, b])
        coverage[x, y] = 0.0


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        x = ti.cast(curve_points_field[i][0] * (WINDOW_WIDTH - 1), ti.i32)
        y = ti.cast(curve_points_field[i][1] * (WINDOW_HEIGHT - 1), ti.i32)

        if 0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT:
            pixels[x, y] = ti.Vector([CURVE_COLOR[0], CURVE_COLOR[1], CURVE_COLOR[2]])


@ti.kernel
def draw_curve_aa_kernel(n: ti.i32):
    for i in range(n):
        fx = curve_points_field[i][0] * (WINDOW_WIDTH - 1)
        fy = curve_points_field[i][1] * (WINDOW_HEIGHT - 1)
        base_x = ti.floor(fx, ti.i32)
        base_y = ti.floor(fy, ti.i32)

        for ox, oy in ti.ndrange((-1, 2), (-1, 2)):
            px = base_x + ox
            py = base_y + oy

            if 0 <= px < WINDOW_WIDTH and 0 <= py < WINDOW_HEIGHT:
                dx = fx - ti.cast(px, ti.f32)
                dy = fy - ti.cast(py, ti.f32)
                distance2 = dx * dx + dy * dy
                weight = ti.exp(-distance2 / (2.0 * AA_SIGMA * AA_SIGMA))
                ti.atomic_max(coverage[px, py], weight)


@ti.kernel
def compose_curve_from_coverage():
    for x, y in pixels:
        alpha = ti.min(coverage[x, y], 1.0)
        pixels[x, y] = (
            ti.Vector([BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2]]) * (1.0 - alpha)
            + ti.Vector([CURVE_COLOR[0], CURVE_COLOR[1], CURVE_COLOR[2]]) * alpha
        )


def de_casteljau(points, t):
    working = np.array(points, dtype=np.float32, copy=True)
    while len(working) > 1:
        working = (1.0 - t) * working[:-1] + t * working[1:]
    return working[0]


def sample_bezier_curve(control_points, out_curve_points):
    control_points_np = np.array(control_points, dtype=np.float32)
    for i in range(BEZIER_NUM_SEGMENTS + 1):
        t = i / BEZIER_NUM_SEGMENTS
        out_curve_points[i] = de_casteljau(control_points_np, t)
    return BEZIER_NUM_SEGMENTS + 1


def sample_bspline_curve(control_points, out_curve_points):
    control_points_np = np.array(control_points, dtype=np.float32)
    segment_count = len(control_points) - 3
    out_index = 0

    for seg in range(segment_count):
        geometry = control_points_np[seg : seg + 4]
        for step in range(BSPLINE_SAMPLES_PER_SEGMENT):
            u = step / BSPLINE_SAMPLES_PER_SEGMENT
            power = np.array([u**3, u**2, u, 1.0], dtype=np.float32)
            out_curve_points[out_index] = power @ BSPLINE_BASIS @ geometry
            out_index += 1

    power = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    out_curve_points[out_index] = power @ BSPLINE_BASIS @ control_points_np[-4:]
    return out_index + 1


def rebuild_curve_points(control_points, curve_mode, out_curve_points):
    if curve_mode == CURVE_MODE_BEZIER:
        if len(control_points) < 2:
            return 0
        return sample_bezier_curve(control_points, out_curve_points)

    if len(control_points) < 4:
        return 0
    return sample_bspline_curve(control_points, out_curve_points)


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


def mode_name(curve_mode):
    if curve_mode == CURVE_MODE_BEZIER:
        return "Bezier"
    return "Uniform Cubic B-Spline"


def print_controls():
    print("Controls:")
    print("  Left Mouse Button: add a control point")
    print("  C: clear all control points")
    print("  B: switch between Bezier and B-spline")
    print("  A: toggle anti-aliasing")


def main():
    window = ti.ui.Window(
        "Lab03 Advanced: Bezier / B-Spline",
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()

    control_points = []
    curve_points_np = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)
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

    curve_mode = CURVE_MODE_BEZIER
    use_antialias = True
    curve_point_count = 0
    curve_dirty = True

    print_controls()
    print(f"Current mode: {mode_name(curve_mode)}, anti-aliasing: ON")

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.LMB and len(control_points) < MAX_CONTROL_POINTS:
                x, y = window.get_cursor_pos()
                control_points.append((float(x), float(y)))
                curve_dirty = True
            elif event.key in ("c", "C"):
                control_points.clear()
                curve_dirty = True
            elif event.key in ("b", "B"):
                if curve_mode == CURVE_MODE_BEZIER:
                    curve_mode = CURVE_MODE_BSPLINE
                else:
                    curve_mode = CURVE_MODE_BEZIER
                curve_dirty = True
                print(f"Current mode: {mode_name(curve_mode)}, anti-aliasing: {'ON' if use_antialias else 'OFF'}")
            elif event.key in ("a", "A"):
                use_antialias = not use_antialias
                print(f"Current mode: {mode_name(curve_mode)}, anti-aliasing: {'ON' if use_antialias else 'OFF'}")

        if curve_dirty:
            update_gui_pools(control_points, gui_points_np, polygon_line_points_np)
            curve_point_count = rebuild_curve_points(control_points, curve_mode, curve_points_np)
            curve_points_field.from_numpy(curve_points_np)
            curve_dirty = False

        clear_buffers(*BACKGROUND_COLOR)

        if curve_point_count > 0:
            if use_antialias:
                draw_curve_aa_kernel(curve_point_count)
                compose_curve_from_coverage()
            else:
                draw_curve_kernel(curve_point_count)

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
