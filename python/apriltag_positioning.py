import pupil_apriltags as apriltags
import numpy as np
import cv2
from dataclasses import dataclass
from spatium import *
import vedo
from math import radians
from scipy import linalg

def homogeneous_vec(vec: np.ndarray):
    # vec: (3, N)
    dim = vec.shape[0]
    div = vec[-1, :].reshape((1, vec.shape[1]))
    div = np.repeat(div, dim, axis=0)
    return vec / div

@dataclass
class Camera:
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

def direction_vector(camera: Camera, points: np.ndarray):
    """
    Get the normalized 3d direction vector of a point on the image taken by a camera

    # Arguments
    - `camera`: The parameters of the camera
    - `points`: a numpy array of dimension (N, 2), where N is the number of points to be calculated.
    """
    n = points.shape[0]
    x_scaling, y_scaling = get_camera_scaling(camera)
    half_resolution = np.array([camera.resolution.x, camera.resolution.y], dtype=np.float64) * 0.5
    half_resolution = np.repeat(half_resolution.reshape((1, 2)), axis=0, repeats=n)
    uv = (points - half_resolution) / half_resolution
    uv[:, 1] = -uv[:, 1]
    scaled_pts = np.concatenate([uv, -np.ones((n, 1))], axis=1)  # set all z values to -1
    scaled_pts[:, 0] *= x_scaling
    scaled_pts[:, 1] *= y_scaling
    return scaled_pts / np.linalg.norm(scaled_pts, axis=1, keepdims=True)

def bgr_to_rgb(bgr: tuple) -> tuple:
    return (bgr[2], bgr[1], bgr[0])

def locate(tag: apriltags.Detection, camera: Camera, scale: float, plt: vedo.Plotter):
    homography = tag.homography
    # order: center, left (-x), right (+x), up (-y), down (+y), UL, UR, DL, DR
    points = np.array([
        [0., -1.,  1.,  0.,  0., -1.,  1., -1.,  1.,],
        [0.,  0.,  0., -1.,  1., -1., -1.,  1.,  1.,],
        [1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,],
    ])
    point_cam_frame = homogeneous_vec(homography @ points)[:2, :].T  # dim: (N, 2)
    def get_edge_direction_vec(vecs) -> np.ndarray:
        nonlocal plt
        """
        Get the direction vector of an edge defined in apriltag space

        # Arguments
        - `vecs` (N, 2): a stack of vectors in the order of starting point, ending point, and midpoint.
        """
        tmp: np.ndarray = direction_vector(camera, vecs)
        start_vec = tmp[0, :].copy()
        # end_vec = tmp[1, :].copy()
        mid_vec = tmp[2, :].copy()
        _A = tmp[:2, :].T
        _y = tmp[2, :] * 2
        solution = np.linalg.inv(_A.T @ _A) @ _A.T @ _y
        direction: np.ndarray = solution[0] * start_vec - mid_vec
        return direction / np.linalg.norm(direction)
    dir_x = get_edge_direction_vec(point_cam_frame[[1, 2, 0], :])
    dir_y = get_edge_direction_vec(point_cam_frame[[3, 4, 0], :])
    dir_z = np.cross(dir_x, dir_y)
    dir_z /=  np.linalg.norm(dir_z)
    # print(np.linalg.norm(dir_x), np.linalg.norm(dir_y), np.linalg.norm(dir_z))
    # plt += vedo.Line([np.array([0., 0., 0.]), dir_z], c="blue")
    
    # Take a section of the frustum with the plane perpendicular to the dir_z vector
    frustum_corners: np.ndarray = direction_vector(camera, point_cam_frame[5:, :])  # shape: (4, 3)
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
    for i in range(points.shape[0]):
        plt += vedo.Point(points[i, :], c="red")
    center = np.sum(points, axis=0) * 0.25
    plt += vedo.Point(center, c="blue")
    return Transform3D(*dir_x, *dir_y, *dir_z, *center)

APRILTAG_EDGE_COLORS = list(map(bgr_to_rgb, [
    (231, 17, 19),
    (94, 204, 51),
    (33, 26, 222),
    (182, 170, 6),
]))

def main():
    plt = vedo.Plotter(size=(1280, 800), interactive=True)
    detector = apriltags.Detector(families="tag36h11", nthreads=1, quad_decimate=2.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.6, debug=0)
    cam = cv2.VideoCapture(0)
    camera_data = Camera(resolution=Vec2i(1920, 1080), fov_x=None, fov_y=radians(50))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_data.resolution.x)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_data.resolution.y)
    x_scaling, y_scaling = get_camera_scaling(camera_data)
    
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
        # if len(tags) >= 1:
        #     print(f"{len(tags)} tags detected")
        for tag in tags:
            for i, corner in enumerate(tag.corners):
                cv2.circle(frame, tuple(corner.astype(int)), 4, (227, 68, 69), 2)
                cv2.line(frame, tuple(corner.astype(int)), tuple(tag.corners[(i+1) % 4].astype(int)), color=APRILTAG_EDGE_COLORS[i])
            cv2.putText(frame, f"tag:{tag.tag_id}", tuple(tag.center.astype(int)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=APRILTAG_EDGE_COLORS[0])
            # print(tag)
            transform = locate(tag, camera_data, 1.0, plt)
            plt += vedo.Point(transform(Vec3(0, 0, 1)), c="pink")
        
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
        img = vedo.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.scale(y_scaling * 2 / frame.shape[0])
        img.pos(-x_scaling, -y_scaling, -1)
        plt += img

        plt.render()

    plt.add_callback("timer", update, enable_picking=False)
    plt.timer_callback("create", dt=1)
    plt.parallel_projection()
    plt.show(viewup="y", title="MediaPipeEye3DPositioner")

if __name__ == "__main__":
    main()