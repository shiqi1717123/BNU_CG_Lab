import taichi as ti

from src import work_basic as basic

MAX_SAMPLES = 8
DEFAULT_SAMPLES = 4


@ti.func
def stable_random01(x: ti.i32, y: ti.i32, sample_id: ti.i32, salt: ti.i32):
    seed = ti.cast(
        x * 1973 + y * 9277 + sample_id * 26699 + salt * 31847,
        ti.f32,
    )
    value = ti.sin(seed) * 43758.5453
    return value - ti.floor(value)


@ti.func
def make_camera_ray_sample(
    px: ti.i32,
    py: ti.i32,
    display_aspect: ti.f32,
    jitter_x: ti.f32,
    jitter_y: ti.f32,
):
    origin = basic.camera_position()
    target = basic.vec3(
        basic.CAMERA_TARGET[0],
        basic.CAMERA_TARGET[1],
        basic.CAMERA_TARGET[2],
    )
    world_up = basic.vec3(0.0, 1.0, 0.0)

    forward = (target - origin).normalized()
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()

    ndc_x = (ti.cast(px, ti.f32) + jitter_x) / ti.cast(basic.WINDOW_WIDTH, ti.f32)
    ndc_y = (ti.cast(py, ti.f32) + jitter_y) / ti.cast(basic.WINDOW_HEIGHT, ti.f32)
    screen_x = (2.0 * ndc_x - 1.0) * display_aspect
    screen_y = 2.0 * ndc_y - 1.0
    tan_half_fov = ti.tan(basic.FOV_DEGREES * 3.14159265359 / 360.0)

    return (
        forward
        + right * screen_x * tan_half_fov
        + up * screen_y * tan_half_fov
    ).normalized()


@ti.func
def trace_ray(ray_origin, ray_dir, light_pos, max_bounces: ti.i32):
    throughput = basic.vec3(1.0, 1.0, 1.0)
    final_color = basic.vec3(0.0, 0.0, 0.0)
    active = 1

    for bounce in range(basic.MAX_BOUNCES):
        if active == 1 and bounce < max_bounces:
            hit_t, hit_kind, material_id = basic.closest_hit(ray_origin, ray_dir)

            if hit_kind == basic.HIT_NONE:
                final_color += throughput * basic.sky_color(ray_dir)
                active = 0
            else:
                point = ray_origin + hit_t * ray_dir
                normal = basic.surface_normal(point, hit_kind)

                if material_id == basic.MAT_MIRROR:
                    mirror_tint = basic.surface_color(point, hit_kind)
                    ray_dir = (
                        ray_dir - 2.0 * ray_dir.dot(normal) * normal
                    ).normalized()
                    ray_origin = point + normal * basic.EPSILON
                    throughput *= mirror_tint * basic.MIRROR_REFLECTANCE
                else:
                    base_color = basic.surface_color(point, hit_kind)
                    shaded = basic.phong_lighting(
                        point,
                        normal,
                        base_color,
                        hit_kind,
                        ray_dir,
                        light_pos,
                    )
                    final_color += throughput * shaded
                    active = 0

    return final_color


@ti.kernel
def render_kernel(
    light_x: ti.f32,
    light_y: ti.f32,
    light_z: ti.f32,
    max_bounces: ti.i32,
    samples_per_pixel: ti.i32,
    display_aspect: ti.f32,
):
    light_pos = basic.vec3(light_x, light_y, light_z)

    for x, y in basic.pixels:
        accumulated_color = basic.vec3(0.0, 0.0, 0.0)
        valid_samples = ti.max(1, samples_per_pixel)

        for sample_id in range(MAX_SAMPLES):
            if sample_id < valid_samples:
                jitter_x = 0.5
                jitter_y = 0.5

                if valid_samples > 1:
                    jitter_x = stable_random01(x, y, sample_id, 0)
                    jitter_y = stable_random01(x, y, sample_id, 1)

                ray_origin = basic.camera_position()
                ray_dir = make_camera_ray_sample(
                    x,
                    y,
                    display_aspect,
                    jitter_x,
                    jitter_y,
                )
                accumulated_color += trace_ray(
                    ray_origin,
                    ray_dir,
                    light_pos,
                    max_bounces,
                )

        basic.pixels[x, y] = basic.clamp01(
            accumulated_color / ti.cast(valid_samples, ti.f32)
        )


def main():
    window = ti.ui.Window(
        "Lab05 Optional 2: MSAA",
        (basic.WINDOW_WIDTH, basic.WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = basic.DEFAULT_LIGHT_POS[0]
    light_y = basic.DEFAULT_LIGHT_POS[1]
    light_z = basic.DEFAULT_LIGHT_POS[2]
    max_bounces = basic.DEFAULT_BOUNCES
    samples_per_pixel = DEFAULT_SAMPLES

    while window.running:
        window_width, window_height = window.get_window_shape()
        display_aspect = window_width / max(1, window_height)

        render_kernel(
            light_x,
            light_y,
            light_z,
            max_bounces,
            samples_per_pixel,
            display_aspect,
        )
        canvas.set_image(basic.pixels)

        with gui.sub_window("Ray Tracing Controls", 0.02, 0.02, 0.34, 0.31) as panel:
            panel.text("MSAA Anti-Aliasing")
            light_x = panel.slider_float("Light X", light_x, -5.0, 5.0)
            light_y = panel.slider_float("Light Y", light_y, 0.2, 6.0)
            light_z = panel.slider_float("Light Z", light_z, -3.0, 6.0)
            max_bounces = panel.slider_int(
                "Max Bounces",
                max_bounces,
                1,
                basic.MAX_BOUNCES,
            )
            samples_per_pixel = panel.slider_int(
                "Samples Per Pixel",
                samples_per_pixel,
                1,
                MAX_SAMPLES,
            )

        window.show()


if __name__ == "__main__":
    main()
