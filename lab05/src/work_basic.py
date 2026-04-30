import math

import taichi as ti

ti.init(arch=ti.gpu, offline_cache=False)

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 640
FOV_DEGREES = 55.0
EPSILON = 1e-4
INF = 1e8

MAX_BOUNCES = 5
DEFAULT_BOUNCES = 3
MIRROR_REFLECTANCE = 0.8

CAMERA_POS = (0.0, 0.45, 5.4)
CAMERA_TARGET = (0.0, -0.15, 0.0)

DEFAULT_LIGHT_POS = (-3.0, 4.0, 3.0)
LIGHT_COLOR = (1.0, 1.0, 1.0)

BACKGROUND_TOP = (0.55, 0.72, 0.88)
BACKGROUND_BOTTOM = (0.06, 0.08, 0.10)

GROUND_Y = -1.0
GROUND_WHITE = (0.86, 0.86, 0.82)
GROUND_BLACK = (0.10, 0.11, 0.12)

RED_SPHERE_CENTER = (-1.5, 0.0, 0.0)
MIRROR_SPHERE_CENTER = (1.5, 0.0, 0.0)
SPHERE_RADIUS = 1.0
RED_COLOR = (0.85, 0.08, 0.06)
SILVER_TINT = (0.82, 0.82, 0.78)

AMBIENT_STRENGTH = 0.12
DIFFUSE_STRENGTH = 0.88
SPECULAR_STRENGTH = 0.18
GROUND_SPECULAR_STRENGTH = 0.03
SHININESS = 48.0

HIT_NONE = 0
HIT_GROUND = 1
HIT_RED_DIFFUSE = 2
HIT_SILVER_MIRROR = 3

MAT_NONE = 0
MAT_DIFFUSE = 1
MAT_MIRROR = 2

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
def sky_color(ray_dir):
    t = 0.5 * (ray_dir[1] + 1.0)
    top = vec3(BACKGROUND_TOP[0], BACKGROUND_TOP[1], BACKGROUND_TOP[2])
    bottom = vec3(BACKGROUND_BOTTOM[0], BACKGROUND_BOTTOM[1], BACKGROUND_BOTTOM[2])
    return bottom * (1.0 - t) + top * t


@ti.func
def camera_position():
    return vec3(CAMERA_POS[0], CAMERA_POS[1], CAMERA_POS[2])


@ti.func
def make_camera_ray(px: ti.i32, py: ti.i32, display_aspect: ti.f32):
    origin = camera_position()
    target = vec3(CAMERA_TARGET[0], CAMERA_TARGET[1], CAMERA_TARGET[2])
    world_up = vec3(0.0, 1.0, 0.0)

    forward = (target - origin).normalized()
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()

    ndc_x = (ti.cast(px, ti.f32) + 0.5) / ti.cast(WINDOW_WIDTH, ti.f32)
    ndc_y = (ti.cast(py, ti.f32) + 0.5) / ti.cast(WINDOW_HEIGHT, ti.f32)
    screen_x = (2.0 * ndc_x - 1.0) * display_aspect
    screen_y = 2.0 * ndc_y - 1.0
    tan_half_fov = ti.tan(FOV_DEGREES * math.pi / 360.0)

    return (
        forward
        + right * screen_x * tan_half_fov
        + up * screen_y * tan_half_fov
    ).normalized()


@ti.func
def intersect_sphere(ray_origin, ray_dir, center):
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
def intersect_ground(ray_origin, ray_dir):
    nearest_t = INF
    if ti.abs(ray_dir[1]) > EPSILON:
        t = (GROUND_Y - ray_origin[1]) / ray_dir[1]
        if t > EPSILON:
            nearest_t = t
    return nearest_t


@ti.func
def hit_material(hit_kind: ti.i32):
    material = MAT_NONE
    if hit_kind == HIT_GROUND or hit_kind == HIT_RED_DIFFUSE:
        material = MAT_DIFFUSE
    elif hit_kind == HIT_SILVER_MIRROR:
        material = MAT_MIRROR
    return material


@ti.func
def closest_hit(ray_origin, ray_dir):
    closest_t = INF
    hit_kind = HIT_NONE

    ground_t = intersect_ground(ray_origin, ray_dir)
    if ground_t < closest_t:
        closest_t = ground_t
        hit_kind = HIT_GROUND

    red_center = vec3(
        RED_SPHERE_CENTER[0],
        RED_SPHERE_CENTER[1],
        RED_SPHERE_CENTER[2],
    )
    red_t = intersect_sphere(ray_origin, ray_dir, red_center)
    if red_t < closest_t:
        closest_t = red_t
        hit_kind = HIT_RED_DIFFUSE

    mirror_center = vec3(
        MIRROR_SPHERE_CENTER[0],
        MIRROR_SPHERE_CENTER[1],
        MIRROR_SPHERE_CENTER[2],
    )
    mirror_t = intersect_sphere(ray_origin, ray_dir, mirror_center)
    if mirror_t < closest_t:
        closest_t = mirror_t
        hit_kind = HIT_SILVER_MIRROR

    return closest_t, hit_kind, hit_material(hit_kind)


@ti.func
def surface_normal(point, hit_kind: ti.i32):
    normal = vec3(0.0, 1.0, 0.0)

    if hit_kind == HIT_RED_DIFFUSE:
        center = vec3(
            RED_SPHERE_CENTER[0],
            RED_SPHERE_CENTER[1],
            RED_SPHERE_CENTER[2],
        )
        normal = (point - center).normalized()
    elif hit_kind == HIT_SILVER_MIRROR:
        center = vec3(
            MIRROR_SPHERE_CENTER[0],
            MIRROR_SPHERE_CENTER[1],
            MIRROR_SPHERE_CENTER[2],
        )
        normal = (point - center).normalized()

    return normal


@ti.func
def checker_color(point):
    tile = ti.cast(ti.floor(point[0]) + ti.floor(point[2]), ti.i32)
    color = vec3(GROUND_WHITE[0], GROUND_WHITE[1], GROUND_WHITE[2])
    if tile % 2 != 0:
        color = vec3(GROUND_BLACK[0], GROUND_BLACK[1], GROUND_BLACK[2])
    return color


@ti.func
def surface_color(point, hit_kind: ti.i32):
    color = vec3(1.0, 1.0, 1.0)

    if hit_kind == HIT_GROUND:
        color = checker_color(point)
    elif hit_kind == HIT_RED_DIFFUSE:
        color = vec3(RED_COLOR[0], RED_COLOR[1], RED_COLOR[2])
    elif hit_kind == HIT_SILVER_MIRROR:
        color = vec3(SILVER_TINT[0], SILVER_TINT[1], SILVER_TINT[2])

    return color


@ti.func
def is_shadowed(point, normal, light_pos):
    light_vec = light_pos - point
    light_distance = light_vec.norm()
    light_dir = light_vec / light_distance
    shadow_origin = point + normal * EPSILON

    hit_t, hit_kind, _ = closest_hit(shadow_origin, light_dir)
    blocked = 0
    if hit_kind != HIT_NONE and hit_t < light_distance - EPSILON:
        blocked = 1

    return blocked


@ti.func
def phong_lighting(point, normal, base_color, hit_kind: ti.i32, incoming_dir, light_pos):
    light_color = vec3(LIGHT_COLOR[0], LIGHT_COLOR[1], LIGHT_COLOR[2])
    light_dir = (light_pos - point).normalized()
    view_dir = (-incoming_dir).normalized()
    ndotl = ti.max(0.0, normal.dot(light_dir))

    ambient = AMBIENT_STRENGTH * base_color * light_color
    color = ambient

    shadowed = 0
    if ndotl > 0.0:
        shadowed = is_shadowed(point, normal, light_pos)

    if shadowed == 0:
        diffuse = DIFFUSE_STRENGTH * ndotl * base_color * light_color
        reflect_dir = (2.0 * normal.dot(light_dir) * normal - light_dir).normalized()
        spec_angle = ti.max(0.0, reflect_dir.dot(view_dir))

        specular_strength = SPECULAR_STRENGTH
        if hit_kind == HIT_GROUND:
            specular_strength = GROUND_SPECULAR_STRENGTH

        specular = specular_strength * ti.pow(spec_angle, SHININESS) * light_color
        color += diffuse + specular

    return clamp01(color)


@ti.kernel
def render_kernel(
    light_x: ti.f32,
    light_y: ti.f32,
    light_z: ti.f32,
    max_bounces: ti.i32,
    display_aspect: ti.f32,
):
    light_pos = vec3(light_x, light_y, light_z)

    for x, y in pixels:
        ray_origin = camera_position()
        ray_dir = make_camera_ray(x, y, display_aspect)

        throughput = vec3(1.0, 1.0, 1.0)
        final_color = vec3(0.0, 0.0, 0.0)
        active = 1

        for bounce in range(MAX_BOUNCES):
            if active == 1 and bounce < max_bounces:
                hit_t, hit_kind, material_id = closest_hit(ray_origin, ray_dir)

                if hit_kind == HIT_NONE:
                    final_color += throughput * sky_color(ray_dir)
                    active = 0
                else:
                    point = ray_origin + hit_t * ray_dir
                    normal = surface_normal(point, hit_kind)

                    if material_id == MAT_MIRROR:
                        mirror_tint = surface_color(point, hit_kind)
                        ray_dir = (
                            ray_dir - 2.0 * ray_dir.dot(normal) * normal
                        ).normalized()
                        ray_origin = point + normal * EPSILON
                        throughput *= mirror_tint * MIRROR_REFLECTANCE
                    else:
                        base_color = surface_color(point, hit_kind)
                        shaded = phong_lighting(
                            point,
                            normal,
                            base_color,
                            hit_kind,
                            ray_dir,
                            light_pos,
                        )
                        final_color += throughput * shaded
                        active = 0

        pixels[x, y] = clamp01(final_color)


def main():
    window = ti.ui.Window(
        "Lab05 Basic: Whitted Ray Tracing",
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = DEFAULT_LIGHT_POS[0]
    light_y = DEFAULT_LIGHT_POS[1]
    light_z = DEFAULT_LIGHT_POS[2]
    max_bounces = DEFAULT_BOUNCES

    while window.running:
        window_width, window_height = window.get_window_shape()
        display_aspect = window_width / max(1, window_height)

        render_kernel(light_x, light_y, light_z, max_bounces, display_aspect)
        canvas.set_image(pixels)

        with gui.sub_window("Ray Tracing Controls", 0.02, 0.02, 0.34, 0.26) as panel:
            panel.text("Whitted-Style Ray Tracing")
            light_x = panel.slider_float("Light X", light_x, -5.0, 5.0)
            light_y = panel.slider_float("Light Y", light_y, 0.2, 6.0)
            light_z = panel.slider_float("Light Z", light_z, -3.0, 6.0)
            max_bounces = panel.slider_int("Max Bounces", max_bounces, 1, MAX_BOUNCES)

        window.show()


if __name__ == "__main__":
    main()
