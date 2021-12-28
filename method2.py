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

(weight, height) = (1920, 1080)  # resolution
max_depth = 1  # number of reflection
light = vec3(5, 5, 5)  # light position
camera = vec3(0, 0, 1)  # camera position
infDist = 1.0e40  # large Distance


def raytrace(ray_origin, ray_direction, objects, reflection=0):
    global rgb
    # check for intersections
    distances = [obj.intersect(ray_origin, ray_direction) for obj in objects]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    # reduce to one array and calculate color
    for (obj, d) in zip(objects, distances):
        color += obj.light(ray_origin, ray_direction, d, objects, reflection) * \
            (nearest != infDist) * (d == nearest)
    return color


class Sphere:
    def __init__(self, center, radius, diffuse, reflectionVar=0.5):
        self.center = center
        self.radius = radius
        self.diffuse = diffuse
        self.reflectionVar = reflectionVar

    def intersect(self, ray_origin, ray_direction):
        b = 2 * ray_direction.dot(ray_origin - self.center)
        c = abs(self.center) + abs(ray_origin) - 2 * \
            self.center.dot(ray_origin) - (self.radius * self.radius)
        delta = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, delta))
        t1 = (-b - sq) / 2
        t2 = (-b + sq) / 2
        h = np.where((t1 > 0) & (t1 < t2), t1, t2)

        pred = (delta > 0) & (h > 0)
        return np.where(pred, h, infDist)

    def renderDiffuse(self):
        return self.diffuse

    def light(self, origin, direction, min_distance, objects, reflection):
        # intersection point
        intersection = (origin + direction * min_distance)
        normalToSurface = (intersection - self.center) * \
            (1. / self.radius)        # normal

        intersectionToLight = (light - intersection).norm()
        intersectionToOrigin = (camera - intersection).norm()
        # offset
        offsetSelf = intersection + normalToSurface * 1.0e-4

        # if the point is shadowed.
        lightDistances = [obj.intersect(
            offsetSelf, intersectionToLight) for obj in objects]
        lightNearest = reduce(np.minimum, lightDistances)
        boolSeelight = lightDistances[objects.index(self)] == lightNearest

        # Ambient
        color = rgb(0.01, 0.01, 0.01)

        # Diffuse
        diffuseLevel = np.maximum(normalToSurface.dot(intersectionToLight), 0)
        color += self.renderDiffuse() * diffuseLevel * boolSeelight

        # Reflection
        if reflection < max_depth:
            rayD = (direction - normalToSurface * 2 *
                    direction.dot(normalToSurface)).norm()
            color += raytrace(offsetSelf, rayD, objects,
                              reflection + 1) * self.reflectionVar

        # Specular
        phong = normalToSurface.dot(
            (intersectionToLight + intersectionToOrigin).norm())
        color += rgb(1, 1, 1) * \
            np.power(np.clip(phong, 0, 1), 50) * boolSeelight
        return color


objects = [
    Sphere(vec3(-0.2, 0, -1), .7, rgb(1, 0, 0)),
    Sphere(vec3(0.1, -0.3, 0), .1, rgb(.7, 0, .7)),
    Sphere(vec3(-0.3, 0, 0), .15, rgb(0, 1, 0)),
    Sphere(vec3(0, -9000, 0), 9000-0.7, rgb(.7, .7, .7)),
]


def main(w, h):
    weight = w
    height = h
    ratio = float(weight) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    x = np.tile(np.linspace(screen[0], screen[2], weight), height)
    y = np.repeat(np.linspace(screen[1], screen[3], height), weight)

    t0 = time.time()
    pixel = vec3(x, y, 0)
    color = raytrace(camera, (pixel - camera).norm(), objects)

    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, weight))
                            ).astype(np.uint8), "L") for c in color.components()]
    Image.merge("RGB", rgb).save("method2.png")
    return time.time() - t0


main(400, 300)
# %%
