import math

import taichi as ti

ti.init(arch=ti.gpu, offline_cache=False)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FOV_DEGREES = 70.0
EPSILON = 1e-4
INF = 1e8

CAMERA_POS = (0.0, 0.0, 5.0)
LIGHT_POS = (2.0, 3.0, 4.0)
LIGHT_COLOR = (1.0, 1.0, 1.0)
BACKGROUND_TOP = (0.02, 0.20, 0.24)
BACKGROUND_BOTTOM = (0.01, 0.08, 0.11)

SPHERE_CENTER = (-1.2, -0.2, 0.0)
SPHERE_RADIUS = 1.2
SPHERE_COLOR = (0.8, 0.1, 0.1)

CONE_APEX = (1.2, 1.2, 0.0)
CONE_BASE_Y = -1.4
CONE_BASE_RADIUS = 1.2
CONE_HEIGHT = CONE_APEX[1] - CONE_BASE_Y
CONE_SLOPE = CONE_BASE_RADIUS / CONE_HEIGHT
CONE_COLOR = (0.6, 0.2, 0.8)

DEFAULT_KA = 0.2
DEFAULT_KD = 0.7
DEFAULT_KS = 0.5
DEFAULT_SHININESS = 32.0

HIT_NONE = 0
HIT_SPHERE = 1
HIT_CONE_SIDE = 2
HIT_CONE_BASE = 3

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WINDOW_WIDTH, WINDOW_HEIGHT))


@ti.func
def vec3(x: ti.f32, y: ti.f32, z: ti.f32):
    return ti.Vector([x, y, z])


@ti.func
def clamp01(color):
    return ti.Vector(
        [
            ti.min(1.0, ti.max(0.0, color[0])),
            ti.min(1.0, ti.max(0.0, color[1])),
            ti.min(1.0, ti.max(0.0, color[2])),
        ]
    )


@ti.func
def background_color(screen_y: ti.i32):
    t = ti.cast(screen_y, ti.f32) / ti.cast(WINDOW_HEIGHT - 1, ti.f32)
    top = vec3(BACKGROUND_TOP[0], BACKGROUND_TOP[1], BACKGROUND_TOP[2])
    bottom = vec3(BACKGROUND_BOTTOM[0], BACKGROUND_BOTTOM[1], BACKGROUND_BOTTOM[2])
    return bottom * (1.0 - t) + top * t


@ti.func
def make_camera_ray(px: ti.i32, py: ti.i32, display_aspect: ti.f32):
    ndc_x = (ti.cast(px, ti.f32) + 0.5) / ti.cast(WINDOW_WIDTH, ti.f32)
    ndc_y = (ti.cast(py, ti.f32) + 0.5) / ti.cast(WINDOW_HEIGHT, ti.f32)
    screen_x = (2.0 * ndc_x - 1.0) * display_aspect
    screen_y = 2.0 * ndc_y - 1.0
    tan_half_fov = ti.tan(FOV_DEGREES * math.pi / 360.0)
    return vec3(screen_x * tan_half_fov, screen_y * tan_half_fov, -1.0).normalized()


@ti.func
def intersect_sphere(ray_origin, ray_dir):
    center = vec3(SPHERE_CENTER[0], SPHERE_CENTER[1], SPHERE_CENTER[2])
    oc = ray_origin - center
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS
    discriminant = b * b - 4.0 * a * c
    nearest_t = INF

    if discriminant >= 0.0:
        sqrt_d = ti.sqrt(discriminant)
        inv_2a = 0.5 / a
        t0 = (-b - sqrt_d) * inv_2a
        t1 = (-b + sqrt_d) * inv_2a
        if t0 > EPSILON:
            nearest_t = t0
        elif t1 > EPSILON:
            nearest_t = t1

    return nearest_t


@ti.func
def cone_side_candidate(ray_origin, ray_dir, t: ti.f32):
    valid = 0
    if t > EPSILON:
        point = ray_origin + t * ray_dir
        local_y = point[1] - CONE_APEX[1]
        if -CONE_HEIGHT <= local_y <= 0.0:
            valid = 1
    return valid


@ti.func
def intersect_cone_side(ray_origin, ray_dir):
    apex = vec3(CONE_APEX[0], CONE_APEX[1], CONE_APEX[2])
    local_origin = ray_origin - apex
    k2 = CONE_SLOPE * CONE_SLOPE

    a = ray_dir[0] * ray_dir[0] + ray_dir[2] * ray_dir[2] - k2 * ray_dir[1] * ray_dir[1]
    b = 2.0 * (
        local_origin[0] * ray_dir[0]
        + local_origin[2] * ray_dir[2]
        - k2 * local_origin[1] * ray_dir[1]
    )
    c = (
        local_origin[0] * local_origin[0]
        + local_origin[2] * local_origin[2]
        - k2 * local_origin[1] * local_origin[1]
    )
    nearest_t = INF

    if ti.abs(a) < EPSILON:
        if ti.abs(b) > EPSILON:
            t = -c / b
            if cone_side_candidate(ray_origin, ray_dir, t) == 1:
                nearest_t = t
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            sqrt_d = ti.sqrt(discriminant)
            inv_2a = 0.5 / a
            t0 = (-b - sqrt_d) * inv_2a
            t1 = (-b + sqrt_d) * inv_2a
            if cone_side_candidate(ray_origin, ray_dir, t0) == 1:
                nearest_t = t0
            if cone_side_candidate(ray_origin, ray_dir, t1) == 1 and t1 < nearest_t:
                nearest_t = t1

    return nearest_t


@ti.func
def intersect_cone_base(ray_origin, ray_dir):
    nearest_t = INF
    if ti.abs(ray_dir[1]) > EPSILON:
        t = (CONE_BASE_Y - ray_origin[1]) / ray_dir[1]
        if t > EPSILON:
            point = ray_origin + t * ray_dir
            dx = point[0] - CONE_APEX[0]
            dz = point[2] - CONE_APEX[2]
            if dx * dx + dz * dz <= CONE_BASE_RADIUS * CONE_BASE_RADIUS:
                nearest_t = t
    return nearest_t


@ti.func
def closest_hit(ray_origin, ray_dir):
    closest_t = INF
    hit_kind = HIT_NONE

    sphere_t = intersect_sphere(ray_origin, ray_dir)
    if sphere_t < closest_t:
        closest_t = sphere_t
        hit_kind = HIT_SPHERE

    cone_side_t = intersect_cone_side(ray_origin, ray_dir)
    if cone_side_t < closest_t:
        closest_t = cone_side_t
        hit_kind = HIT_CONE_SIDE

    cone_base_t = intersect_cone_base(ray_origin, ray_dir)
    if cone_base_t < closest_t:
        closest_t = cone_base_t
        hit_kind = HIT_CONE_BASE

    return closest_t, hit_kind


@ti.func
def surface_normal(point, hit_kind: ti.i32):
    normal = vec3(0.0, 1.0, 0.0)

    if hit_kind == HIT_SPHERE:
        center = vec3(SPHERE_CENTER[0], SPHERE_CENTER[1], SPHERE_CENTER[2])
        normal = (point - center).normalized()
    elif hit_kind == HIT_CONE_SIDE:
        local = point - vec3(CONE_APEX[0], CONE_APEX[1], CONE_APEX[2])
        k2 = CONE_SLOPE * CONE_SLOPE
        normal = vec3(local[0], -k2 * local[1], local[2]).normalized()
    elif hit_kind == HIT_CONE_BASE:
        normal = vec3(0.0, -1.0, 0.0)

    return normal


@ti.func
def object_color(hit_kind: ti.i32):
    color = vec3(1.0, 1.0, 1.0)

    if hit_kind == HIT_SPHERE:
        color = vec3(SPHERE_COLOR[0], SPHERE_COLOR[1], SPHERE_COLOR[2])
    elif hit_kind == HIT_CONE_SIDE or hit_kind == HIT_CONE_BASE:
        color = vec3(CONE_COLOR[0], CONE_COLOR[1], CONE_COLOR[2])

    return color


@ti.func
def phong_shader(point, normal, base_color, ka: ti.f32, kd: ti.f32, ks: ti.f32, shininess: ti.f32):
    light_pos = vec3(LIGHT_POS[0], LIGHT_POS[1], LIGHT_POS[2])
    camera_pos = vec3(CAMERA_POS[0], CAMERA_POS[1], CAMERA_POS[2])
    light_color = vec3(LIGHT_COLOR[0], LIGHT_COLOR[1], LIGHT_COLOR[2])

    light_dir = (light_pos - point).normalized()
    view_dir = (camera_pos - point).normalized()

    ndotl = ti.max(0.0, normal.dot(light_dir))
    ambient = ka * base_color * light_color
    diffuse = kd * ndotl * base_color * light_color

    specular = vec3(0.0, 0.0, 0.0)
    if ndotl > 0.0:
        reflect_dir = (2.0 * normal.dot(light_dir) * normal - light_dir).normalized()
        spec_angle = ti.max(0.0, reflect_dir.dot(view_dir))
        specular = ks * ti.pow(spec_angle, shininess) * light_color

    return clamp01(ambient + diffuse + specular)


@ti.kernel
def render_kernel(
    ka: ti.f32,
    kd: ti.f32,
    ks: ti.f32,
    shininess: ti.f32,
    display_aspect: ti.f32,
):
    ray_origin = vec3(CAMERA_POS[0], CAMERA_POS[1], CAMERA_POS[2])

    for x, y in pixels:
        ray_dir = make_camera_ray(x, y, display_aspect)
        hit_t, hit_kind = closest_hit(ray_origin, ray_dir)

        color = background_color(y)
        if hit_kind != HIT_NONE:
            point = ray_origin + hit_t * ray_dir
            normal = surface_normal(point, hit_kind)
            base_color = object_color(hit_kind)
            color = phong_shader(point, normal, base_color, ka, kd, ks, shininess)

        pixels[x, y] = color


def main():
    window = ti.ui.Window(
        "Lab04 Basic: Phong Lighting",
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    ka = DEFAULT_KA
    kd = DEFAULT_KD
    ks = DEFAULT_KS
    shininess = DEFAULT_SHININESS

    while window.running:
        window_width, window_height = window.get_window_shape()
        display_aspect = window_width / max(1, window_height)

        render_kernel(ka, kd, ks, shininess, display_aspect)
        canvas.set_image(pixels)

        with gui.sub_window("Phong Controls", 0.02, 0.02, 0.36, 0.24) as panel:
            panel.text("Local Phong Lighting")
            ka = panel.slider_float("Ka (Ambient)", ka, 0.0, 1.0)
            kd = panel.slider_float("Kd (Diffuse)", kd, 0.0, 1.0)
            ks = panel.slider_float("Ks (Specular)", ks, 0.0, 1.0)
            shininess = panel.slider_float("Shininess", shininess, 1.0, 128.0)

        window.show()


if __name__ == "__main__":
    main()
