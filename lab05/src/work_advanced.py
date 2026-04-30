import taichi as ti

from src import work_basic as basic

MAT_GLASS = 3

DEFAULT_GLASS_IOR = 1.5
AIR_IOR = 1.0
GLASS_TRANSMISSION = 0.96
GLASS_TINT = (0.92, 0.98, 1.0)
GLASS_REFLECTION_TINT = (0.86, 0.94, 1.0)


@ti.func
def hit_material(hit_kind: ti.i32):
    material = basic.MAT_NONE
    if hit_kind == basic.HIT_GROUND:
        material = basic.MAT_DIFFUSE
    elif hit_kind == basic.HIT_RED_DIFFUSE:
        material = MAT_GLASS
    elif hit_kind == basic.HIT_SILVER_MIRROR:
        material = basic.MAT_MIRROR
    return material


@ti.func
def closest_hit(ray_origin, ray_dir):
    closest_t = basic.INF
    hit_kind = basic.HIT_NONE

    ground_t = basic.intersect_ground(ray_origin, ray_dir)
    if ground_t < closest_t:
        closest_t = ground_t
        hit_kind = basic.HIT_GROUND

    glass_center = basic.vec3(
        basic.RED_SPHERE_CENTER[0],
        basic.RED_SPHERE_CENTER[1],
        basic.RED_SPHERE_CENTER[2],
    )
    glass_t = basic.intersect_sphere(ray_origin, ray_dir, glass_center)
    if glass_t < closest_t:
        closest_t = glass_t
        hit_kind = basic.HIT_RED_DIFFUSE

    mirror_center = basic.vec3(
        basic.MIRROR_SPHERE_CENTER[0],
        basic.MIRROR_SPHERE_CENTER[1],
        basic.MIRROR_SPHERE_CENTER[2],
    )
    mirror_t = basic.intersect_sphere(ray_origin, ray_dir, mirror_center)
    if mirror_t < closest_t:
        closest_t = mirror_t
        hit_kind = basic.HIT_SILVER_MIRROR

    return closest_t, hit_kind, hit_material(hit_kind)


@ti.func
def is_shadowed(point, normal, light_pos):
    light_vec = light_pos - point
    light_distance = light_vec.norm()
    light_dir = light_vec / light_distance
    shadow_origin = point + normal * basic.EPSILON

    blocked = 0
    active = 1
    travelled = 0.0

    for _ in range(basic.MAX_BOUNCES):
        if active == 1:
            hit_t, hit_kind, material_id = closest_hit(shadow_origin, light_dir)
            if hit_kind != basic.HIT_NONE and travelled + hit_t < light_distance:
                if material_id == MAT_GLASS:
                    step = hit_t + basic.EPSILON
                    shadow_origin += light_dir * step
                    travelled += step
                else:
                    blocked = 1
                    active = 0
            else:
                active = 0

    return blocked


@ti.func
def local_lighting(point, normal, base_color, hit_kind: ti.i32, incoming_dir, light_pos):
    light_color = basic.vec3(
        basic.LIGHT_COLOR[0],
        basic.LIGHT_COLOR[1],
        basic.LIGHT_COLOR[2],
    )
    light_dir = (light_pos - point).normalized()
    view_dir = (-incoming_dir).normalized()
    ndotl = ti.max(0.0, normal.dot(light_dir))

    ambient = basic.AMBIENT_STRENGTH * base_color * light_color
    color = ambient

    shadowed = 0
    if ndotl > 0.0:
        shadowed = is_shadowed(point, normal, light_pos)

    if shadowed == 0:
        diffuse = basic.DIFFUSE_STRENGTH * ndotl * base_color * light_color
        reflect_dir = (2.0 * normal.dot(light_dir) * normal - light_dir).normalized()
        spec_angle = ti.max(0.0, reflect_dir.dot(view_dir))

        specular_strength = basic.SPECULAR_STRENGTH
        if hit_kind == basic.HIT_GROUND:
            specular_strength = basic.GROUND_SPECULAR_STRENGTH

        specular = specular_strength * ti.pow(spec_angle, basic.SHININESS) * light_color
        color += diffuse + specular

    return basic.clamp01(color)


@ti.func
def refract_or_reflect(incoming_dir, normal, glass_ior: ti.f32):
    cos_i = ti.min(1.0, ti.max(-1.0, incoming_dir.dot(normal)))
    eta_i = AIR_IOR
    eta_t = glass_ior
    oriented_normal = normal

    if cos_i < 0.0:
        cos_i = -cos_i
    else:
        eta_i = glass_ior
        eta_t = AIR_IOR
        oriented_normal = -normal

    eta = eta_i / eta_t
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    new_dir = basic.vec3(0.0, 0.0, 0.0)
    total_internal_reflection = 0

    if sin2_t > 1.0:
        total_internal_reflection = 1
        new_dir = (
            incoming_dir
            - 2.0 * incoming_dir.dot(oriented_normal) * oriented_normal
        ).normalized()
    else:
        cos_t = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
        new_dir = (
            eta * incoming_dir
            + (eta * cos_i - cos_t) * oriented_normal
        ).normalized()

    return new_dir, total_internal_reflection


@ti.kernel
def render_kernel(
    light_x: ti.f32,
    light_y: ti.f32,
    light_z: ti.f32,
    max_bounces: ti.i32,
    glass_ior: ti.f32,
    display_aspect: ti.f32,
):
    light_pos = basic.vec3(light_x, light_y, light_z)

    for x, y in basic.pixels:
        ray_origin = basic.camera_position()
        ray_dir = basic.make_camera_ray(x, y, display_aspect)

        throughput = basic.vec3(1.0, 1.0, 1.0)
        final_color = basic.vec3(0.0, 0.0, 0.0)
        active = 1

        for bounce in range(basic.MAX_BOUNCES):
            if active == 1 and bounce < max_bounces:
                hit_t, hit_kind, material_id = closest_hit(ray_origin, ray_dir)

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
                    elif material_id == MAT_GLASS:
                        ray_dir, total_internal_reflection = refract_or_reflect(
                            ray_dir,
                            normal,
                            glass_ior,
                        )
                        ray_origin = point + ray_dir * basic.EPSILON

                        if total_internal_reflection == 1:
                            tint = basic.vec3(
                                GLASS_REFLECTION_TINT[0],
                                GLASS_REFLECTION_TINT[1],
                                GLASS_REFLECTION_TINT[2],
                            )
                            throughput *= tint * basic.MIRROR_REFLECTANCE
                        else:
                            tint = basic.vec3(
                                GLASS_TINT[0],
                                GLASS_TINT[1],
                                GLASS_TINT[2],
                            )
                            throughput *= tint * GLASS_TRANSMISSION
                    else:
                        base_color = basic.surface_color(point, hit_kind)
                        shaded = local_lighting(
                            point,
                            normal,
                            base_color,
                            hit_kind,
                            ray_dir,
                            light_pos,
                        )
                        final_color += throughput * shaded
                        active = 0

        basic.pixels[x, y] = basic.clamp01(final_color)


def main():
    window = ti.ui.Window(
        "Lab05 Optional 1: Glass Refraction",
        (basic.WINDOW_WIDTH, basic.WINDOW_HEIGHT),
        vsync=True,
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x = basic.DEFAULT_LIGHT_POS[0]
    light_y = basic.DEFAULT_LIGHT_POS[1]
    light_z = basic.DEFAULT_LIGHT_POS[2]
    max_bounces = basic.DEFAULT_BOUNCES
    glass_ior = DEFAULT_GLASS_IOR

    while window.running:
        window_width, window_height = window.get_window_shape()
        display_aspect = window_width / max(1, window_height)

        render_kernel(
            light_x,
            light_y,
            light_z,
            max_bounces,
            glass_ior,
            display_aspect,
        )
        canvas.set_image(basic.pixels)

        with gui.sub_window("Ray Tracing Controls", 0.02, 0.02, 0.34, 0.31) as panel:
            panel.text("Glass Refraction")
            light_x = panel.slider_float("Light X", light_x, -5.0, 5.0)
            light_y = panel.slider_float("Light Y", light_y, 0.2, 6.0)
            light_z = panel.slider_float("Light Z", light_z, -3.0, 6.0)
            max_bounces = panel.slider_int(
                "Max Bounces",
                max_bounces,
                1,
                basic.MAX_BOUNCES,
            )
            glass_ior = panel.slider_float("Glass IOR", glass_ior, 1.0, 2.4)

        window.show()


if __name__ == "__main__":
    main()
