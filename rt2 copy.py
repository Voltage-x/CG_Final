# In[1]:

from PIL import Image
from functools import reduce
import numpy as np
import time


class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)


rgb = vec3

(weight, height) = (1920, 1080)         # Screen size
max_depth = 1
light = vec3(5, 5, 5)        # Point light position
camera = vec3(0, 0, 1)     # Eye position
FARAWAY = 1.0e40            # an implausibly huge distance


def raytrace(ray_origin, ray_direction, objects, reflection=0):

    distances = [obj.intersect(ray_origin, ray_direction) for obj in objects]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (obj, d) in zip(objects, distances):
        color += obj.light(ray_origin, ray_direction, d, objects, reflection) * \
            (nearest != FARAWAY) * (d == nearest)
    return color


class Sphere:
    def __init__(self, center, radius, diffuse, mirror=0.5):
        self.center = center
        self.radius = radius
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.center)
        c = abs(self.center) + abs(O) - 2 * \
            self.center.dot(O) - (self.radius * self.radius)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)

        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, origin, direction, min_distance, scene, reflection):
        # intersection point
        intersection = (origin + direction * min_distance)
        normal_to_surface = (intersection - self.center) * \
            (1. / self.radius)        # normal
        # direction to light
        intersection_to_light = (light - intersection).norm()
        # direction to ray origin
        intersection_to_origin = (camera - intersection).norm()
        # M nudged to avoid itself
        nudged = intersection + normal_to_surface * .0001

        # Shadow: find if the point is shadowed or not.
        light_distances = [s.intersect(
            nudged, intersection_to_light) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.01, 0.01, 0.01)

        # Lambert shading (diffuse)
        lv = np.maximum(normal_to_surface.dot(intersection_to_light), 0)
        color += self.diffusecolor(intersection) * lv * seelight

        # Reflection
        if reflection < max_depth:
            rayD = (direction - normal_to_surface * 2 *
                    direction.dot(normal_to_surface)).norm()
            color += raytrace(nudged, rayD, scene,
                              reflection + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = normal_to_surface.dot(
            (intersection_to_light + intersection_to_origin).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


scene = [
    Sphere(vec3(-0.2, 0, -1), .7, rgb(1, 0, 0)),
    Sphere(vec3(0.1, -0.3, 0), .1, rgb(.7, 0, .7)),
    Sphere(vec3(-0.3, 0, 0), .15, rgb(0, 1, 0)),
    Sphere(vec3(0, -9000, 0), 9000-0.7, rgb(.7, .7, .7)),
]

ratio = float(weight) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
x = np.tile(np.linspace(screen[0], screen[2], weight), height)
y = np.repeat(np.linspace(screen[1], screen[3], height), weight)

t0 = time.time()
pixel = vec3(x, y, 0)
color = raytrace(camera, (pixel - camera).norm(), scene)
print("Took", time.time() - t0)

rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, weight))
                        ).astype(np.uint8), "L") for c in color.components()]
Image.merge("RGB", rgb).save("rt2.png")

# %%
