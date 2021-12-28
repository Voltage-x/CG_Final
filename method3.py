# In[1]:
from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers
import random


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


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

    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))

    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(
            cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


rgb = vec3

(weight, height) = (1920, 1080)         # Screen size
light = vec3(5, 5, 5)        # Point light position
camera = vec3(0, 0, 1)     # Eye position
FARAWAY = np.inf            # an implausibly huge distance
max_depth = 1


def raytrace(ray_origin, ray_direction, objects, reflection=0):

    distances = [obj.intersect(ray_origin, ray_direction) for obj in objects]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(objects, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = ray_origin.extract(hit)
            Dc = ray_direction.extract(hit)
            cc = s.light(Oc, Dc, dc, objects, reflection)
            color += cc.place(hit)
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

    def diffusecolor(self):
        return self.diffuse

    def light(self, origin, min_distance, direction, scene, reflection):
        # intersection point
        intersection = (origin + min_distance * direction)
        normal_to_surface = (intersection - self.center) * \
            (1. / self.radius)        # normal
        # direction to light
        intersection_to_light = (light - intersection).norm()
        # direction to ray origin
        intersection_to_original = (camera - intersection).norm()
        # M nudged to avoid itself
        nudged = intersection + normal_to_surface * .0001

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(
            nudged, intersection_to_light) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.001, 0.001, 0.001)

        # Lambert shading (diffuse)
        lv = np.maximum(normal_to_surface.dot(intersection_to_light), 0)
        color += self.diffusecolor() * lv * seelight

        # Reflection
        if reflection < max_depth:
            rayD = (min_distance - normal_to_surface * 2 *
                    min_distance.dot(normal_to_surface)).norm()
            color += raytrace(nudged, rayD, scene,
                              reflection + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = normal_to_surface.dot(
            (intersection_to_light + intersection_to_original).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


scene = [
    #Sphere(vec3(-0.2, 0, -1), .7, rgb(.7, 0, 0)),
    #Sphere(vec3(0.1, -0.3, 0), .1, rgb(.7, 0, .7)),
    #Sphere(vec3(-0.3, 0, 0), .15, rgb(0, .7, 0)),
    Sphere(vec3(0, -9000, 0), 9000-0.7, rgb(.7, .7, .7)),
]


def main(w, h, n):
    rgb = vec3
    v1 = -2.6
    v2 = 1.7
    v3 = .1
    for i in range(n):
        scene.append(Sphere(vec3(v1, v2, -2), v3,
                     rgb(random.random(), random.random(), random.random())))
        v1 += .3
        if (i+1) % 16 == 0 and i != 0:
            v1 = -2.6
            v2 -= .3
    weight = w
    height = h
    ratio = float(weight) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    x = np.tile(np.linspace(screen[0], screen[2], weight), height)
    y = np.repeat(np.linspace(screen[1], screen[3], height), weight)

    t0 = time.time()
    pixel = vec3(x, y, 0)
    color = raytrace(camera, (pixel - camera).norm(), scene)

    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, weight))
                            ).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", rgb).save("method3.png")
    return time.time() - t0
# %%
