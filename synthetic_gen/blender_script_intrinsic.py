"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
import pickle

import bpy
import numpy as np
from mathutils import Matrix, Vector

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

FOCAL_LENGTHS: list = [
    # 13, 
    24, 
    28, 
    35, 
    48, 
    70,
    # 77, 
    # 120
]


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""

    scene = bpy.context.scene
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    camera.data.lens = random.choice(FOCAL_LENGTHS)

    return camera

def set_camera_at_position(
    x: float, y: float, z: float, focal_length: float = 35
) -> bpy.types.Object:
    """Sets the camera at the given position.

    Args:
        x (float): x-coordinate of the camera.
        y (float): y-coordinate of the camera.
        z (float): z-coordinate of the camera.
        focal_length (float, optional): Focal length of the camera. Defaults to 35.

    Returns:
        bpy.types.Object: The camera object.
    """
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    camera.data.lens = focal_length
    return camera

def generate_camera_trajectory(
    num_frames: int,
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 1.8,
    minz: float = 1.1,
    only_northern_hemisphere: bool = False,
) -> List[Tuple[float, float, float]]:
    """Generates a camera trajectory inside of a spherical shell.

    Args:
        num_frames (int): Number of frames in the camera trajectory.
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        List[Tuple[float, float, float]]: List of (x, y, z) points in the spherical
        shell.
    """
    camera_positions = []
    for _ in range(num_frames):
        x, y, z = _sample_spherical(
            radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
        )
        if only_northern_hemisphere:
            z = abs(z)
        camera_positions.append((x, y, z))
    # sort by x, y rotation
    xy_rotations = [np.arctan2(x, y) for x, y, z in camera_positions]
    camera_positions = [
        pos
        for _, pos in sorted(zip(xy_rotations, camera_positions), key=lambda pair: pair[0])
    ]
    return camera_positions


def calculate_near_far(camera: bpy.types.Object, bbox_min: Vector, bbox_max: Vector) -> Tuple[float, float]:
    corners = [
        Vector((bbox_min.x, bbox_min.y, bbox_min.z)),
        Vector((bbox_min.x, bbox_min.y, bbox_max.z)),
        Vector((bbox_min.x, bbox_max.y, bbox_min.z)),
        Vector((bbox_min.x, bbox_max.y, bbox_max.z)),
        Vector((bbox_max.x, bbox_min.y, bbox_min.z)),
        Vector((bbox_max.x, bbox_min.y, bbox_max.z)),
        Vector((bbox_max.x, bbox_max.y, bbox_min.z)),
        Vector((bbox_max.x, bbox_max.y, bbox_max.z)),
    ]
    camera_location = camera.matrix_world.translation
    distances = [(camera_location - corner).length for corner in corners]
    return min(distances), max(distances)

def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        file_path = os.path.abspath(object_path)
        inner_path = "Collection"
        object_name = os.path.basename(file_path).split(".")[0]
        out = import_function(filepath=os.path.join(file_path, inner_path, object_name),
                        directory=os.path.join(file_path, inner_path),
                        filename=object_name)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT

# def get_K_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
#     """Returns the intrinsic camera matrix from the given camera.
    
#     Args:
#         cam (bpy.types.Object): The camera object.

#     Returns:
#         Matrix: The intrinsic camera matrix from the given camera.
#     """

#     camera_data = cam.data
    
#     # Extract camera parameters
#     f_in_mm = camera_data.lens  # Focal length in mm
#     sensor_width_in_mm = camera_data.sensor_width  # Sensor width in mm
#     sensor_height_in_mm = camera_data.sensor_height  # Sensor height in mm
#     resolution_x_in_px = bpy.context.scene.render.resolution_x  # Image resolution width
#     resolution_y_in_px = bpy.context.scene.render.resolution_y  # Image resolution height
#     scale = bpy.context.scene.render.resolution_percentage / 100.0  # Resolution scale factor
#     pixel_aspect_ratio = bpy.context.scene.render.pixel_aspect_x / bpy.context.scene.render.pixel_aspect_y
    
#     # Adjust sensor size for pixel aspect ratio
#     sensor_width_in_px = resolution_x_in_px * scale
#     sensor_height_in_px = resolution_y_in_px * scale / pixel_aspect_ratio
    
#     # Compute focal length in pixels
#     fx = f_in_mm * (sensor_width_in_px / sensor_width_in_mm)
#     fy = f_in_mm * (sensor_height_in_px / sensor_height_in_mm)
    
#     # Principal point (assuming the principal point is at the center of the image)
#     cx = sensor_width_in_px / 2.0
#     cy = sensor_height_in_px / 2.0
    
#     # Create the intrinsic camera matrix
#     K = Matrix(((fx, 0, cx),
#                 (0, fy, cy),
#                 (0, 0, 1)))
    
#     return K


# def get_K_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
#     """Returns the intrinsic camera matrix from the given camera.
    
#     Args:
#         cam (bpy.types.Object): The camera object.

#     Returns:
#         mathutils.Matrix: The intrinsic camera matrix from the given camera.
#     """
#     if cam.type != 'CAMERA':
#         raise ValueError("The object must be a camera")
    
#     camera_data = cam.data
    
#     # Extract camera parameters
#     f_in_mm = camera_data.lens  # Focal length in mm
#     sensor_width_in_mm = camera_data.sensor_width  # Sensor width in mm
#     sensor_height_in_mm = camera_data.sensor_height  # Sensor height in mm
    
#     # Get render resolution from the scene
#     scene = bpy.context.scene
#     resolution_x_in_px = scene.render.resolution_x  # Image resolution width
#     resolution_y_in_px = scene.render.resolution_y  # Image resolution height
#     scale = scene.render.resolution_percentage / 100.0  # Resolution scale factor
#     pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    
#     # Adjust sensor size for pixel aspect ratio
#     sensor_width_in_px = resolution_x_in_px * scale
#     sensor_height_in_px = resolution_y_in_px * scale / pixel_aspect_ratio
    
#     # Compute focal length in pixels
#     fx = f_in_mm * (sensor_width_in_px / sensor_width_in_mm)
#     fy = f_in_mm * (sensor_height_in_px / sensor_height_in_mm)
    
#     # Principal point (assuming the principal point is at the center of the image)
#     cx = sensor_width_in_px / 2.0
#     cy = sensor_height_in_px / 2.0
    
#     # Create the intrinsic camera matrix
#     K = Matrix(((fx, 0, cx),
#                 (0, fy, cy),
#                 (0, 0, 1)))
    
#     return K


import bpy
from mathutils import Matrix

def get_K_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """
    Returns the intrinsic camera matrix from the given camera.

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The intrinsic camera matrix from the given camera.
    """
    # Get the camera data
    cam_data = cam.data

    # Get the camera parameters
    f_in_mm = cam_data.lens
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y

    # Calculate the pixel aspect ratio
    pixel_aspect_ratio = bpy.context.scene.render.pixel_aspect_x / bpy.context.scene.render.pixel_aspect_y

    # Calculate the focal length in pixels
    f_x = f_in_mm / cam_data.sensor_width * resolution_x
    f_y = f_in_mm / cam_data.sensor_height * resolution_y

    # Calculate the principal point offset
    c_x = resolution_x / 2
    c_y = resolution_y / 2

    # Construct the intrinsic camera matrix
    K = Matrix(((f_x, 0, c_x),
                (0, f_y * pixel_aspect_ratio, c_y),
                (0, 0, 1)))
    
    # print(resolution_x, resolution_y, f_x, f_y, c_x, c_y, pixel_aspect_ratio)

    return K


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    follow_trajectory: bool,
    output_dir: str,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # load the object
    if object_file.endswith(".blend"):
        
        bpy.ops.wm.open_mainfile(filepath=object_file)
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
        if "Cube" in bpy.data.objects:
            bpy.data.objects["Cube"].hide_render = True
            bpy.data.objects["Cube"].hide_viewport = True
        # load_object(object_file)
        
    else:
        reset_scene()
        delete_invisible_objects()
        load_object(object_file)

    for scene in bpy.data.scenes:
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 128
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 3
        scene.cycles.transmission_bounces = 3
        scene.cycles.filter_width = 0.01
        scene.cycles.use_denoising = True
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.resolution_x = 512
        scene.render.resolution_y = 384
        scene.render.resolution_percentage = 100


    # Set up cameras
    scene = bpy.context.scene
    cam = scene.objects["Camera"]
    # cam.data.lens = 35
    cam.data.sensor_width = 32
    cam.data.sensor_height = 24

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        missing_textures = delete_missing_textures()
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    # normalize the scene
    normalize_scene()

    # randomize the lighting
    randomize_lighting()

    bbox_min, bbox_max = scene_bbox()

    # render the images
    if follow_trajectory:
        camera_positions = generate_camera_trajectory(num_renders)
        for i, pos in enumerate(camera_positions):
            render_path = os.path.join(output_dir, f"{i:03d}.png")
            scene.render.filepath = render_path
            focal = random.choice(FOCAL_LENGTHS)
            camera = set_camera_at_position(pos[0], pos[1], pos[2], focal)

            # render the image
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)

            rt_matrix = get_3x4_RT_matrix_from_blender(camera)
            rt_matrix_path = os.path.join(output_dir, f"{i:03d}_rt.npy")
            np.save(rt_matrix_path, rt_matrix)
            K_matrix = get_K_matrix_from_blender(camera)
            K_matrix_path = os.path.join(output_dir, f"{i:03d}_k.npy")
            np.save(K_matrix_path, K_matrix)

            near, far = calculate_near_far(camera, bbox_min, bbox_max)

            image_meta = {
                "focal_length": focal,
                "near": near,
                "far": far,
            }
            image_meta_path = os.path.join(output_dir, f"{i:03d}.pkl")

            # save metadata
            with open(image_meta_path, "wb") as f:
                pickle.dump(image_meta, f)

    else:
        for i in range(num_renders):
            # set camera
            camera = randomize_camera(
                only_northern_hemisphere=only_northern_hemisphere,
            )

            # render the image
            render_path = os.path.join(output_dir, f"{i:03d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

            # save camera RT matrix
            rt_matrix = get_3x4_RT_matrix_from_blender(camera)
            rt_matrix_path = os.path.join(output_dir, f"{i:03d}_rt.npy")
            np.save(rt_matrix_path, rt_matrix)

            # save camera intrinsic matrix
            K_matrix = get_K_matrix_from_blender(camera)
            K_matrix_path = os.path.join(output_dir, f"{i:03d}_k.npy")
            np.save(K_matrix_path, K_matrix)

            # save metadata
            near, far = calculate_near_far(camera, bbox_min, bbox_max)
            image_meta = {
                "focal_length": camera.data.lens,
                "near": near,
                "far": far,
            }
            image_meta_path = os.path.join(output_dir, f"{i:03d}.pkl")

            with open(image_meta_path, "wb") as f:
                pickle.dump(image_meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object.",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=12,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--follow_trajectory",
        action="store_true",
        help="Follow a trajectory around the object.",
        default=False,
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 512
    render.resolution_y = 384
    render.resolution_percentage = 100

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # Render the images
    render_object(
        object_file=args.object_path,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        follow_trajectory=args.follow_trajectory,
        output_dir=args.output_dir,
    )