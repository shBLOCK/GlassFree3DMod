import pupil_apriltags as apriltags
import numpy as np
import cv2
from dataclasses import dataclass
from spatium import *
import vedo
from math import radians
from scipy import linalg
import queue
from queue import Queue
import socket
import datetime as dt
import json
from threading import Thread

def homogeneous_vec(vec: np.ndarray):
    # vec: (3, N)
    dim = vec.shape[0]
    div = vec[-1, :].reshape((1, vec.shape[1]))
    div = np.repeat(div, dim, axis=0)
    return vec / div

@dataclass
class Camera:
    """Parameters of a camera"""
    resolution: Vec2i
    fov_x: float | None
    fov_y: float | None

def get_camera_scaling(camera: Camera) -> tuple[float, float]:
    x_scaling = 0.0
    y_scaling = 0.0
    if camera.fov_x == None and camera.fov_y == None:
        raise f"You must specify either fov_x or fov_y!"
    elif camera.fov_x == None:
        y_scaling = np.tan(camera.fov_y / 2)
        x_scaling = y_scaling * camera.resolution.x / camera.resolution.y
    elif camera.fov_y == None:
        x_scaling = np.tan(camera.fov_x / 2)
        y_scaling = x_scaling * camera.resolution.y / camera.resolution.x
    else:
        x_scaling = np.tan(camera.fov_x / 2)
        y_scaling = np.tan(camera.fov_y / 2)
    return (x_scaling, y_scaling)

def get_camera_matrix(camera: Camera) -> np.ndarray:
    """Get the conversion matrix between the camera frame (x, y, z) to the image frame (unit: pixel)"""
    x_scaling, y_scaling = get_camera_scaling(camera)
    uv_to_image = np.array([
        [camera.resolution.x/2, 0.0, camera.resolution.x/2],
        [0.0, camera.resolution.y/2, camera.resolution.y/2],
        [0.0, 0.0, 1.0],
    ])
    world_to_uv = np.array([[1/x_scaling, 0.0, 0.0], [0.0, 1/y_scaling, 0.0], [0.0, 0.0, 1.0]])
    return uv_to_image @ world_to_uv

def get_inv_camera_matrix(camera: Camera) -> np.ndarray:
    """Get the conversion matrix between image frame (unit: pixel) to camera 3D frame (x, y, z)"""
    x_scaling, y_scaling = get_camera_scaling(camera)
    image_to_uv = np.array([
        [2/camera.resolution.x, 0.0, -1.0],
        [0.0, 2/camera.resolution.y, -1.0],
        [0.0, 0.0, 1.0],
    ])
    uv_to_world = np.array([[x_scaling, 0.0, 0.0], [0.0, y_scaling, 0.0], [0.0, 0.0, 1.0]])
    return uv_to_world @ image_to_uv

@dataclass
class TaggedObject:
    """An object that uses apriltags to locate."""
    name: str
    tags: dict[int, tuple[float, Transform3D]]  # records the mapping from tag IDs to their scale and relative transformation from the object's center

def get_simple_tag_object(name: str, id: int, scale: float):
    """Get a simple object with only one tag. Use the center of the tag as the object's center."""
    return TaggedObject(name=name, tags=dict([(id, (scale, Transform3D()))]))

def locate(id: int, tag: apriltags.Detection, camera: Camera, scale: float, plt: vedo.Plotter | None = None):
    # # `denoiser` records the mapping between apriltag id and their corresponding denoiser state.
    # # a denoiser state is defined by a denoiser object and an integer -- disconnected frame count.
    # # disconnected frame count records the number of frames since the last time an april tag is
    # # detected.
    # --- Deprecated (2025.7.6)

    inv_camera_mat = get_inv_camera_matrix(camera)
    point_cam_frame: np.ndarray
    # order: center, left (-x), right (+x), up (-y), down (+y), UL, UR, DL, DR
    # dimension: (3, 9)
    points = np.array([
        [0., -1.,  1.,  0.,  0., -1.,  1., -1.,  1.,],
        [0.,  0.,  0., -1.,  1., -1., -1.,  1.,  1.,],
        [1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,],
    ])

    # Use the detected positions to update the tag
    homography = tag.homography
    point_cam_frame = homogeneous_vec(inv_camera_mat @ homography @ points)    # dim: (3, N)
    # if plt != None:
    #     plt += vedo.Points(-homogeneous_vec(point_cam_frame).T, c="red")

    def get_edge_direction_vec(endpoints, midpoint) -> np.ndarray:
        nonlocal plt
        """
        Get the direction vector of an edge defined in apriltag space

        # Arguments
        - `endpoints` (3, 2): a stack of vectors in the order of starting point and ending point.
        - `midpoint` (3,): a vector pointing to the direction of midpoint.
        """
        start_vec = endpoints[:, 0].copy()
        end_vec = endpoints[:, 1].copy()
        _A = endpoints
        _y = midpoint
        solution = np.linalg.inv(_A.T @ _A) @ _A.T @ _y
        direction: np.ndarray = solution[0] * start_vec - midpoint * 0.5
        # if plt != None:
            # plt += vedo.Line([Vec3(), Vec3(*start_vec)], c="red")
            # plt += vedo.Line([Vec3(), Vec3(*end_vec)], c="green")
            # plt += vedo.Line([Vec3(), Vec3(*mid_vec)], c="blue")
            # plt += vedo.Line([Vec3(), direction], c="orange")
        return direction / np.linalg.norm(direction)

    dir_x = get_edge_direction_vec(point_cam_frame[:, [1, 2]], point_cam_frame[:, 0])
    dir_y = get_edge_direction_vec(point_cam_frame[:, [3, 4]], point_cam_frame[:, 0])
    dir_z = np.cross(dir_x, dir_y)
    dir_z /= np.linalg.norm(dir_z)
    if plt != None:
        plt += vedo.Line([np.zeros(3), dir_x], c="red")
        plt += vedo.Line([np.zeros(3), dir_y], c="green")
        plt += vedo.Line([np.zeros(3), dir_z], c="blue")
    
    # Take a section of the frustum with the plane perpendicular to the dir_z vector
    frustum_corners: np.ndarray = (point_cam_frame[:, 5:]).T  # shape: (4, 3)
    ts = frustum_corners @ dir_z
    ts = np.repeat(ts.reshape((-1, 1)), repeats=3, axis=1)
    points = frustum_corners / ts
    edge_lengths = np.array([
            np.linalg.norm(points[0, :] - points[1, :]),   # up
            np.linalg.norm(points[2, :] - points[3, :]),   # down
            np.linalg.norm(points[0, :] - points[2, :]),   # left
            np.linalg.norm(points[1, :] - points[3, :])    # right
    ])
    # print(edge_lengths)
    k = scale / edge_lengths.mean()
    points *= k
    center = np.sum(points, axis=0) * 0.25
    if plt != None:
        plt += vedo.Point(center, c="blue")
        for i in range(points.shape[0]):
            plt += vedo.Point(points[i, :], c="red")
    
    # # generate random points to check the variation
    # N = 100
    # for _ in range(N):
    #     new_point_cam_frame = point_cam_frame + np.random.randn(*point_cam_frame.shape) * 0.002
    #     new_dir_x = get_edge_direction_vec(new_point_cam_frame[:, [1, 2]], new_point_cam_frame[:, 0])
    #     new_dir_y = get_edge_direction_vec(new_point_cam_frame[:, [3, 4]], new_point_cam_frame[:, 0])
    #     new_dir_z = np.cross(new_dir_x, new_dir_y)
    #     new_dir_z /= np.linalg.norm(new_dir_z)
    #     if plt != None:
    #         plt += vedo.Points(-homogeneous_vec(new_point_cam_frame).T, c="yellow")
    #         plt += vedo.Line([np.zeros(3), new_dir_z], c="gray")
    return Transform3D(*dir_x, *dir_y, *dir_z, *center)

def locate_by_PnP(tags: dict[int, apriltags.Detection], object: TaggedObject, camera: Camera, plt: vedo.Plotter | None = None) -> Transform3D | None:
    APRILTAG_CORNER_ORDER = [
        Vec3(-0.5, 0.5, 0.0),
        Vec3(0.5, 0.5, 0.0),
        Vec3(0.5, -0.5, 0.0),
        Vec3(-0.5, -0.5, 0.0),
    ]
    camera_mat = get_camera_matrix(camera)
    distortion = np.zeros(5)
    object_points = []
    image_points = []
    for tag_id in object.tags.keys():
        if tag_id in tags:
            tag = tags[tag_id]
            scale, tag_in_obj = object.tags[tag_id]
            # The object requires the tag to locate
            for i in range(4):
                object_points.append(tag_in_obj(APRILTAG_CORNER_ORDER[i] * scale))
                image_points.append(tag.corners[i])
    object_points = np.array(object_points)
    # print(object_points)
    image_points = np.array(image_points)
    if len(object_points) == 0 or len(image_points) == 0:
        return None

    # call solvePnP
    ret, rotation_vec, translation_vec = cv2.solvePnP(object_points, image_points, camera_mat, distortion)
    if not ret:
        print(f"Cannot solve the PnP problem on the object")
        return
    rotation_mat, jacobian = cv2.Rodrigues(rotation_vec)
    return Transform3D(*rotation_mat.flatten("F"), *translation_vec.flatten())

def server_thread_main(queue: Queue[str], port: int = 30002):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))
        print(f"Server started at port {port}")
        while True:
            sock.listen()
            conn, addr = sock.accept()
            with conn:
                print(f"Accepted client: {addr}")
                while True:
                    try:
                        conn.sendall(queue.get().encode())
                    except OSError as e:
                        print(f"Failed to send packet: {e}")
                        break

def bgr_to_rgb(bgr: tuple) -> tuple:
    return (bgr[2], bgr[1], bgr[0])

def transform_to_json(t: Transform3D) -> dict:
    return {
        "rx": [*t.x],
        "ry": [*t.y],
        "rz": [*t.z],
        "t": [*t.origin],
    }

APRILTAG_EDGE_COLORS = list(map(bgr_to_rgb, [
    (231, 17, 19),
    (94, 204, 51),
    (33, 26, 222),
    (182, 170, 6),
]))

MAX_DISCONNECTED_CNT = 6

def main():
    plt = vedo.Plotter(size=(1280, 800), interactive=True)
    detector = apriltags.Detector(families="tag36h11", nthreads=1, quad_decimate=2.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.6, debug=0)
    cam = cv2.VideoCapture(0)
    camera_data = Camera(resolution=Vec2i(1920, 1080), fov_x=None, fov_y=radians(50))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_data.resolution.x)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_data.resolution.y)
    x_scaling, y_scaling = get_camera_scaling(camera_data)

    # TODO: load these objects from a file instead of hardcoding them in here
    screen_obj = TaggedObject(name="handhold screen", tags={
        0: (28, Transform3D.translating(Vec3(-42.4, -21.5, 0.0))),
        1: (28, Transform3D.translating(Vec3(42.4, -21.5, 0.0))),
        2: (28, Transform3D.translating(Vec3(-42.4, 21.5, 0.0))),
        3: (28, Transform3D.translating(Vec3(42.4, 21.5, 0.0))),
    })
    wand_obj = TaggedObject(name="wand", tags={
        120: (40, Transform3D.translating(Vec3(0., 0., -31.))),
        121: (40, Transform3D.rotating(Vec3(1., 0., 0.), np.pi/2).rotate_ip(Vec3(0., 0., 1.), -np.pi/2).translate_ip(Vec3(31., 0., 0.))),
        122: (40, Transform3D.rotating(Vec3(1., 0., 0.), np.pi/2).rotate_ip(Vec3(0., 0., 1.), np.pi).translate_ip(Vec3(0., -31.0, 0.))),
        123: (40, Transform3D.rotating(Vec3(1., 0., 0.), np.pi/2).rotate_ip(Vec3(0., 0., 1.), np.pi/2).translate_ip(Vec3(-31., 0., 0.))),
        124: (40, Transform3D.rotating(Vec3(1., 0., 0.), np.pi/2).translate_ip(Vec3(0., 31., 0.)))
    })
    object_to_detect = [screen_obj, wand_obj, get_simple_tag_object('ababa', 128, 4.0)]

    packet_queue: Queue[str] = Queue()
    server_thread = Thread(target=server_thread_main, args=(packet_queue,), name="Server thread", daemon=True)
    server_thread.start()

    # update function in the main thread
    def update(*_):
        nonlocal plt
        for obj in plt.objects:
            plt.remove(obj)
        ret, frame = cam.read()
        if not ret:
            print("An error occurred with the camera")
            plt.close()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags: list[apriltags.Detection] = detector.detect(gray)
        tags_dict: dict[int, apriltags.Detection] = dict([(tag.tag_id, tag) for tag in tags])
        for obj in object_to_detect:
            obj_transform = locate_by_PnP(tags_dict, obj, camera_data, plt)
            if obj_transform != None:
                plt += vedo.Point(obj_transform(Vec3(0., 0., 0.)), c="black")
                plt += vedo.Line(obj_transform(Vec3(0., 0., 0.)), obj_transform(Vec3(20., 0., 0.)), c="red")
                plt += vedo.Line(obj_transform(Vec3(0., 0., 0.)), obj_transform(Vec3(0., 20., 0.)), c="green")
                plt += vedo.Line(obj_transform(Vec3(0., 0., 0.)), obj_transform(Vec3(0., 0., 20.)), c="blue")
                for id, (scale, tag_transform) in obj.tags.items():
                    points = [
                        Vec3(-.5, .5, 0.), Vec3(.5, .5, 0.),
                        Vec3(.5, -.5, 0.), Vec3(-.5, -.5, 0.),
                        Vec3(-.5, .5, 0.)
                    ]
                    points = [obj_transform(tag_transform(point * scale)) for point in points]
                    plt += vedo.Line(points, c="red")
                    for point in points[:4]:
                        plt += vedo.Line([-point / point.z, point], c="black")
                    
                    # send the data to the server
                    timestamp = dt.datetime.now().timestamp()
                    packet = json.dumps({
                        "name": obj.name,
                        "transform": transform_to_json(obj_transform),
                        "time": timestamp,
                    }) + "\n"
                    try:
                        packet_queue.put_nowait(packet)
                    except queue.Full:
                        print("Packet queue full")
        
        # visualize
        plt += vedo.Axes(
            xrange=(-10, 10), yrange=(-10, 10), zrange=(-10, 0),
            xygrid=False,
            text_scale=0.5,
            z_inverted=True,
            xshift_along_y=0.5,
            xshift_along_z=1,
            yshift_along_x=0.5,
            yshift_along_z=1,
            zshift_along_x=0.5,
            zshift_along_y=0.5
        )
        img = vedo.Image(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
        img.scale(y_scaling * 2 / frame.shape[0])
        img.pos(-x_scaling, -y_scaling, -1)
        plt += img

        plt.render()

    plt.add_callback("timer", update, enable_picking=False)
    plt.timer_callback("create", dt=1)
    plt.parallel_projection()
    plt.show(viewup="y", title="MediaPipeEye3DPositioner")
    server_thread.join()

if __name__ == "__main__":
    main()