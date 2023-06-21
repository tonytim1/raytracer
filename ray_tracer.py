import argparse
import random
from typing import Optional

from PIL import Image
import numpy as np

from Intersection import Intersection
from Ray import Ray
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from utils import normalize, get_perpendicular_plane, EPSILON


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    assert camera is not None
    return camera, scene_settings, objects


def save_image(image_array, output_image):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(output_image)


class Scene:
    def __init__(self, camera, scene_settings, objects, width, height):
        self.camera = camera
        # we set here because camera doesn't get (width, height) in init, and they are needed for height
        self.camera.set_screen_height(width, height)
        self.scene_settings = scene_settings
        self.materials = [i for i in objects if isinstance(i, Material)]
        self.spheres = [i for i in objects if isinstance(i, Sphere)]
        self.planes = [i for i in objects if isinstance(i, InfinitePlane)]
        self.cubes = [i for i in objects if isinstance(i, Cube)]
        self.lights = [i for i in objects if isinstance(i, Light)]
        self.width = width
        self.height = height

        # camera.towards == self.towards, duplicated until we decide the best place for it
        self.towards = normalize(camera.look_at - camera.position)
        self.right = normalize(np.cross(camera.up_vector, self.towards))
        self.up = normalize(np.cross(self.right, self.towards))

        self.screen_center = self.camera.position + self.camera.screen_distance * self.towards  # P_c

        self.left_bottom = (
            self.screen_center
            - self.camera.screen_width/2 * self.right   # why not self.right ?
            - self.camera.screen_height/2 * self.up  # why not self.up ?
        )  # P_0

        pixel_height = self.camera.screen_height / height  # Ry
        pixel_width = self.camera.screen_width / width  # Rx
        self.Vy = pixel_height * self.up  # vertical
        self.Vx = pixel_width * self.right  # horizontal

    def construct_ray_through_pixel(self, i, j) -> Ray:
        P = self.left_bottom + i * self.Vy + j * self.Vx
        ray = Ray(self.camera.position, normalize(P - self.camera.position))
        return ray

    def find_intersection(self, ray: Ray, start_surface=None) -> Optional[Intersection]:
        min_t = None
        min_surface = None

        for surface in self.spheres + self.planes + self.cubes:
            if surface is start_surface:  # in case of reflection/transparency we dont want to pick same obj again
                continue

            t = surface.intersect(ray)
            if t is not None:
                if min_t is None or t < min_t:
                    min_t, min_surface = t, surface

        if min_t is not None:
            return Intersection(min_t, min_surface, ray)

        return None

    def get_color(self, hit: Optional[Intersection], rec_depth=0) -> list[float]:
        if hit is None or rec_depth == self.scene_settings.max_recursions:
            return self.scene_settings.background_color

        diffuse = np.zeros(3, dtype=float)
        specular = np.zeros(3, dtype=float)
        reflection = 0
        transparency = self.scene_settings.background_color

        material = self.materials[hit.surface.material_index - 1]
        N = hit.get_normal()  # N

        for light in self.lights:
            L = normalize(hit.intersect_pos - light.position)  # L
            NL = np.maximum(np.dot(N, -L), 0)  # NL
            R = normalize(L + 2 * NL * N)
            V = -hit.ray.V
            RV = np.maximum(np.dot(R, V), 0)

            light_intensity = self.get_light_intensity(hit, light, L)  # I_L

            diffuse += light.color * light_intensity * NL
            specular += light.color * light_intensity * (RV ** material.shininess) * light.specular_intensity

        diffuse *= material.diffuse_color
        specular *= material.specular_color

        # reflection
        if material.reflection_color.sum() > 0:
            reflection_vector = normalize(hit.ray.V - 2 * N * np.dot(hit.ray.V, N))
            reflection_ray = Ray(hit.intersect_pos, reflection_vector)
            reflection_hit = self.find_intersection(reflection_ray, hit.surface)  # we ignore current surface
            reflection = self.get_color(reflection_hit, rec_depth + 1) * material.reflection_color

        # transparency
        if material.transparency > 0:
            transparency_ray = Ray(hit.intersect_pos, hit.ray.V)
            transparency_hit = self.find_intersection(transparency_ray, hit.surface)  # we ignore current surface
            transparency = self.get_color(transparency_hit, rec_depth + 1) * material.transparency

        return transparency * material.transparency + (diffuse + specular) * (1 - material.transparency) + reflection

    def get_light_intensity(self, hit, light, light_ray):
        N = self.scene_settings.root_number_shadow_rays

        # Find a plane perpendicular to the ray
        P_x = get_perpendicular_plane(light_ray)
        if P_x is None:
            return 1 - light.shadow_intensity
        P_y = normalize(np.cross(P_x, light_ray))

        # Define rectangle on that plane
        left_bottom = light.position - (light.radius / 2) * P_x - (light.radius / 2) * P_y
        cell_size = light.radius / N
        P_x *= cell_size
        P_y *= cell_size

        # Shoot a ray from the center of each cell
        shadow_count = 0
        for i in range(N):
            for j in range(N):
                shadow_pos = left_bottom + P_x * (i + random.random()) + P_y * (j + random.random())
                shadow_vector = normalize(hit.intersect_pos - shadow_pos)
                shadow_ray = Ray(shadow_pos, shadow_vector)
                shadow_hit = self.find_intersection(shadow_ray)
                if np.linalg.norm(shadow_hit.intersect_pos - hit.intersect_pos) < EPSILON:
                #if shadow_hit.surface is hit.surface:
                    shadow_count += 1

        return 1 - light.shadow_intensity + light.shadow_intensity * (shadow_count / (N ** 2))

    def ray_tracing(self):
        image_array = np.zeros((self.height, self.width, 3), dtype=float)

        for i in range(self.width):
            print(i)
            for j in range(self.height):
                ray = self.construct_ray_through_pixel(i, j)
                hit = self.find_intersection(ray)
                image_array[i, j] = self.get_color(hit)

        return image_array


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    scene = Scene(camera, scene_settings, objects, args.width, args.height)
    image_array = scene.ray_tracing()

    # Save the output image
    image_array = np.clip(image_array, 0, 1) * 255
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
