import taichi as ti

from src import work_basic as basic

SHADOW_BIAS = 1e-3
HIT_GROUND = 4
GROUND_Y = -1.45
GROUND_X_LIMIT = 4.0
GROUND_Z_MIN = -4.0
GROUND_Z_MAX = 3.0
GROUND_COLOR = (0.12, 0.28, 0.30)


@ti.func
def intersect_ground(ray_origin, ray_dir):
    nearest_t = basic.INF
    if ti.abs(ray_dir[1]) > basic.EPSILON:
        t = (GROUND_Y - ray_origin[1]) / ray_dir[1]
        if t > basic.EPSILON:
            point = ray_origin + t * ray_dir
            if (
                -GROUND_X_LIMIT <= point[0] <= GROUND_X_LIMIT
                and GROUND_Z_MIN <= point[2] <= GROUND_Z_MAX
            ):
                nearest_t = t
    return nearest_t


@ti.func
def closest_hit(ray_origin, ray_dir, include_ground: ti.i32):
    closest_t, hit_kind = basic.closest_hit(ray_origin, ray_dir)

    if include_ground == 1:
        ground_t = intersect_ground(ray_origin, ray_dir)
        if ground_t < closest_t:
            closest_t = ground_t
            hit_kind = HIT_GROUND

    return closest_t, hit_kind


@ti.func
def surface_normal(point, hit_kind: ti.i32):
    normal = basic.surface_normal(point, hit_kind)
    if hit_kind == HIT_GROUND:
        normal = basic.vec3(0.0, 1.0, 0.0)
    return normal


@ti.func
def object_color(hit_kind: ti.i32):
    color = basic.object_color(hit_kind)
    if hit_kind == HIT_GROUND:
        color = basic.vec3(GROUND_COLOR[0], GROUND_COLOR[1], GROUND_COLOR[2])
    return color


@ti.func
def is_shadowed(point, normal):
    light_pos = basic.vec3(
        basic.LIGHT_POS[0],
        basic.LIGHT_POS[1],
        basic.LIGHT_POS[2],
    )
    light_vec = light_pos - point
    light_distance = light_vec.norm()
    light_dir = light_vec / light_distance
    shadow_origin = point + normal * SHADOW_BIAS
    hit_t, hit_kind = closest_hit(shadow_origin, light_dir, 0)

    blocked = 0
    if hit_kind != basic.HIT_NONE and hit_t < light_distance - SHADOW_BIAS:
        blocked = 1
    return blocked


@ti.func
def local_lighting(
    point,
    normal,
    base_color,
    ka: ti.f32,
    kd: ti.f32,
    ks: ti.f32,
    shininess: ti.f32,
    use_blinn: ti.i32,
    use_shadow: ti.i32,
):
    light_pos = basic.vec3(
        basic.LIGHT_POS[0],
        basic.LIGHT_POS[1],
        basic.LIGHT_POS[2],
    )
    camera_pos = basic.vec3(
        basic.CAMERA_POS[0],
        basic.CAMERA_POS[1],
        basic.CAMERA_POS[2],
    )
    light_color = basic.vec3(
        basic.LIGHT_COLOR[0],
        basic.LIGHT_COLOR[1],
        basic.LIGHT_COLOR[2],
    )

    light_dir = (light_pos - point).normalized()
    view_dir = (camera_pos - point).normalized()
    ndotl = ti.max(0.0, normal.dot(light_dir))

    ambient = ka * base_color * light_color
    diffuse = kd * ndotl * base_color * light_color

    shadowed = 0
    if use_shadow == 1 and ndotl > 0.0:
        shadowed = is_shadowed(point, normal)

    specular = basic.vec3(0.0, 0.0, 0.0)
    if ndotl > 0.0 and shadowed == 0:
        spec_angle = 0.0
        if use_blinn == 1:
            half_dir = (light_dir + view_dir).normalized()
            spec_angle = ti.max(0.0, normal.dot(half_dir))
        else:
            reflect_dir = (2.0 * normal.dot(light_dir) * normal - light_dir).normalized()
            spec_angle = ti.max(0.0, reflect_dir.dot(view_dir))
        specular = ks * ti.pow(spec_angle, shininess) * light_color

    color = ambient
    if shadowed == 0:
        color += diffuse + specular

    return basic.clamp01(color)


@ti.kernel
def render_advanced_kernel(
    ka: ti.f32,
    kd: ti.f32,
    ks: ti.f32,
    shininess: ti.f32,
    display_aspect: ti.f32,
    use_blinn: ti.i32,
    use_shadow: ti.i32,
):
    ray_origin = basic.vec3(
        basic.CAMERA_POS[0],
        basic.CAMERA_POS[1],
        basic.CAMERA_POS[2],
    )

    for x, y in basic.pixels:
        ray_dir = basic.make_camera_ray(x, y, display_aspect)
        hit_t, hit_kind = closest_hit(ray_origin, ray_dir, 1)

        color = basic.background_color(y)
        if hit_kind != basic.HIT_NONE:
            point = ray_origin + hit_t * ray_dir
            normal = surface_normal(point, hit_kind)
            base_color = object_color(hit_kind)
            color = local_lighting(
                point,
                normal,
                base_color,
                ka,
                kd,
                ks,
                shininess,
                use_blinn,
                use_shadow,
            )

        basic.pixels[x, y] = color


def main():
    window = ti.ui.Window(
        "Lab04 Optional: Blinn-Phong and Hard Shadow",
        (basic.WINDOW_WIDTH, basic.WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    ka = basic.DEFAULT_KA
    kd = basic.DEFAULT_KD
    ks = basic.DEFAULT_KS
    shininess = basic.DEFAULT_SHININESS
    use_blinn = True
    use_shadow = True

    while window.running:
        window_width, window_height = window.get_window_shape()
        display_aspect = window_width / max(1, window_height)

        render_advanced_kernel(
            ka,
            kd,
            ks,
            shininess,
            display_aspect,
            int(use_blinn),
            int(use_shadow),
        )
        canvas.set_image(basic.pixels)

        with gui.sub_window("Optional Controls", 0.02, 0.02, 0.39, 0.32) as panel:
            panel.text("Specular and Shadow")
            ka = panel.slider_float("Ka (Ambient)", ka, 0.0, 1.0)
            kd = panel.slider_float("Kd (Diffuse)", kd, 0.0, 1.0)
            ks = panel.slider_float("Ks (Specular)", ks, 0.0, 1.0)
            shininess = panel.slider_float("Shininess", shininess, 1.0, 128.0)
            use_blinn = panel.checkbox("Use Blinn-Phong", use_blinn)
            use_shadow = panel.checkbox("Use Hard Shadow", use_shadow)

        window.show()


if __name__ == "__main__":
    main()
