# In[1]:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numbers
from functools import reduce


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

FARAWAY = 1.0e39
L = vec3(5, 5, -10)        # Point light position
E = vec3(0, 0.35, -1)     # Eye position


class Sphere:
    def __init__(self, center, r, diffuse, mirror=0.5):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, reflection):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if reflection < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += sphere_intersect(nudged, rayD,
                                      scene, reflection + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker


def normalize(x):
    x /= np.linalg.norm(x)
    return x


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


# = intersect(rayO, rayD, obj)
def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction, reflection=0):
    distances = [s.intersect(ray_origin, ray_direction) for s in objects]
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


width = 300
height = 200

max_depth = 3

camera = vec3(0, 0.35, -1)
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

light = {'position': np.array([5, 5, 5]), 'ambient': np.array(
    [1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1])}

objects = [
    Sphere(vec3(.75, .1, 1), .6, rgb(0, 0, 1)),
    Sphere(vec3(-.75, .1, 2.25), .6, rgb(.5, .223, .5)),
    Sphere(vec3(-2.75, .1, 3.5), .6, rgb(1, .572, .184)),
    CheckeredSphere(vec3(0, -99999.5, 0), 99999, rgb(.75, .75, .75), 0.25),
]

image = np.zeros((height, width, 3))
x = np.tile(np.linspace(screen[0], screen[2], width), height)
y = np.repeat(np.linspace(screen[1], screen[3], height), width)
# screen is on origin
pixel = vec3(x, y, 0)  # Q
origin = camera  # O
direction = (pixel - origin).norm()  # (Q-E).norm

color = np.zeros((3))  # col
reflection = 1

# check for intersections
color = nearest_intersected_object(objects, origin, direction)

rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))
                        ).astype(np.uint8), "L") for c in color.components()]
Image.merge("RGB", rgb).save("test.png")

# %%
