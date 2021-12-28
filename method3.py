# In[1]:
from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers
import random


def extractHitPixels(cond, x):
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

    def normalizationVec3(self):
        magnitude = np.sqrt(abs(self))
        return self * (1.0 / np.where(magnitude == 0, 1, magnitude))

    def rgbValues(self):
        return (self.x, self.y, self.z)

    def extractHitPixels(self, cond):
        return vec3(extractHitPixels(cond, self.x),
                    extractHitPixels(cond, self.y),
                    extractHitPixels(cond, self.z))

    def copyValue(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(
            cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r


rgb = vec3

(weight, height) = (1920, 1080)  # resolution
light = vec3(5, 5, 5)  # light position
camera = vec3(0, 0, 1)  # camera position
infDist = 1.0e40  # large Distance
max_depth = 1  # number of reflection


def raytrace(rayOrigin, rayDirection, objects, reflection=0):
    global rgb

    # check for intersections
    distances = [obj.intersect(rayOrigin, rayDirection) for obj in objects]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)

    # reduce to one array of each object and calculate color
    for (obj, d) in zip(objects, distances):
        hit = (nearest != infDist) & (d == nearest)
        if np.any(hit):
            directionHit = extractHitPixels(hit, d)
            originalHit = rayOrigin.extractHitPixels(hit)
            distanceHit = rayDirection.extractHitPixels(hit)
            calculateHit = obj.light(
                originalHit, distanceHit, directionHit, objects, reflection)
            color += calculateHit.copyValue(hit)
    return color


class Sphere:
    def __init__(self, center, radius, diffuse, reflectionVar=0.5):
        self.center = center
        self.radius = radius
        self.diffuse = diffuse
        self.reflectionVar = reflectionVar

    def intersect(self, rayOrigin, rayDirection):
        b = 2 * rayDirection.dot(rayOrigin - self.center)
        c = abs(self.center) + abs(rayOrigin) - 2 * \
            self.center.dot(rayOrigin) - (self.radius * self.radius)
        delta = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, delta))
        t1 = (-b - sq) / 2
        t2 = (-b + sq) / 2
        h = np.where((t1 > 0) & (t1 < t2), t1, t2)

        pred = (delta > 0) & (h > 0)
        return np.where(pred, h, infDist)

    def renderDiffuse(self):
        return self.diffuse

    def light(self, origin, direction, minDistance, objects, reflection):
        # intersection point
        intersection = (origin + direction * minDistance)
        normalToSurface = (intersection - self.center) * \
            (1. / self.radius)

        intersectionToLight = (light - intersection).normalizationVec3()
        intersectionToOrigin = (camera - intersection).normalizationVec3()

        # offset
        offsetSelf = intersection + normalToSurface * 1.0e-4

        # if the point is shadowed.
        lightDistances = [obj.intersect(
            offsetSelf, intersectionToLight) for obj in objects]
        lightNearest = reduce(np.minimum, lightDistances)
        boolSeelight = lightDistances[objects.index(self)] == lightNearest

        # ambient
        color = rgb(0.001, 0.001, 0.001)

        # diffuse
        diffuseLevel = np.maximum(normalToSurface.dot(intersectionToLight), 0)
        color += self.renderDiffuse() * diffuseLevel * boolSeelight

        # reflection times
        if reflection < max_depth:
            newRayDirection = (direction - normalToSurface * 2 *
                               direction.dot(normalToSurface)).normalizationVec3()
            color += raytrace(offsetSelf, newRayDirection, objects,
                              reflection + 1) * self.reflectionVar

        # specular
        phongShading = normalToSurface.dot(
            (intersectionToLight + intersectionToOrigin).normalizationVec3())
        color += rgb(1, 1, 1) * \
            np.power(np.clip(phongShading, 0, 1), 50) * boolSeelight
        return color


objects = [
    Sphere(vec3(-0.2, 0, -1), .7, rgb(.7, 0, 0)),
    Sphere(vec3(0.1, -0.3, 0), .1, rgb(.7, 0, .7)),
    Sphere(vec3(-0.3, 0, 0), .15, rgb(0, .7, 0)),
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

    # screen is on origin
    pixel = vec3(x, y, 0)
    color = raytrace(camera, (pixel - camera).normalizationVec3(), objects)

    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, weight))
                            ).astype(np.uint8), "L") for c in color.rgbValues()]
    Image.merge("RGB", rgb).save("method3.png")
    return time.time() - t0
# %%
