import queue
import time
import collections
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread

import mediapipe as mp
import cv2
import numpy as np
from numpy import sin
import vedo
from spatium import *
from mediapipe.tasks.python.components.containers import landmark as landmark_module


def _landmark_to_vec3(landmark: landmark_module.NormalizedLandmark) -> Vec3:
    return Vec3(landmark.x, landmark.y, landmark.z)

def _fmt_vec3(v: Vec3):
    return f"({v.x: .1f}, {v.y: .1f}, {v.z: .1f})"

# def _draw_landmarks_on_image(rgb_image, landmarks):
#     from mediapipe.framework.formats import landmark_pb2
#
#     annotated_image = np.copy(rgb_image)
#
#     # Draw the face landmarks.
#     face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     face_landmarks_proto.landmark.extend([
#         landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
#     ])
#
#     mp.solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_tesselation_style())
#     mp.solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_contours_style())
#     mp.solutions.drawing_utils.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks_proto,
#         connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp.solutions.drawing_styles
#         .get_default_face_mesh_iris_connections_style())
#
#     return annotated_image

# noinspection PyProtectedMember
class MediaPipeEye3DPositioner:
    @dataclass
    class Result:
        _face_landmarks: list[landmark_module.NormalizedLandmark]
        _face_transform: Transform3D
        _frame_view: np.ndarray

        left_eye_3d: Vec3 = None
        right_eye_3d: Vec3 = None

    def __init__(
        self,
        camera: cv2.VideoCapture = None,
        fov_y: float = np.pi / 2,
        std_eye_distance: float = 6.0,
        visualize: bool = True,
        result_callback: Callable[[Result], None] = lambda _: None
    ):
        self.camera = camera or cv2.VideoCapture(0)
        self.fov_y = fov_y
        self.std_eye_distance = std_eye_distance
        self._visualize = visualize
        self.result_callback = result_callback

        self.running = True
        self._results_queue = queue.Queue()
        self._processor_thread = Thread(
            target=self._processor_thread_main,
            name="MediaPipeEye3DPositioningProcessor",
            daemon=True
        )
        self._camera_thread = Thread(
            target=self._camera_thread_main,
            name="MediaPipeEye3DPositioningCamera",
            daemon=True
        )
        if self._visualize:
            self._visualization_thread = Thread(
                target=self._visualization_thread_main,
                name="MediaPipeEye3DPositioningVisualization",
                daemon=True
            )

        self._last_3d_visualizer: Callable | None = None

        self._visualization_image = None
        self._visualization_trail_left = []
        self._visualization_trail_right = []

    def _process_result(self, result: Result):
        left_eye_uv = _landmark_to_vec3(result._face_landmarks[473]).xy
        right_eye_uv = _landmark_to_vec3(result._face_landmarks[468]).xy

        ar = result._frame_view.shape[1] / result._frame_view.shape[0]
        p_scale = Vec2(sin(self.fov_y / 2) * ar, -sin(self.fov_y / 2))
        left_eye_p_2d = (left_eye_uv - 0.5) * 2 * p_scale
        right_eye_p_2d = (right_eye_uv - 0.5) * 2 * p_scale
        left_eye_p = Vec3(left_eye_p_2d, -1)
        right_eye_p = Vec3(right_eye_p_2d, -1)

        # normal vector of the q plane (plane formed by origin, left_eye_p and right_eye_p)
        q_normal = (left_eye_p ^ right_eye_p).normalized

        head_right_dir = result._face_transform.x
        # head_right_dir, projected into the p plane
        head_right_dir_q = (head_right_dir - q_normal * (head_right_dir @ q_normal)).normalized

        head_right_dir_q_perpendicular = (q_normal ^ head_right_dir_q).normalized

        # left_eye_p & right_eye_p with standard depth (has projection length of one when projected onto head_right_dir_q_perpendicular)
        left_eye_p_std_depth = left_eye_p / (left_eye_p @ head_right_dir_q_perpendicular)
        right_eye_p_std_depth = right_eye_p / (right_eye_p @ head_right_dir_q_perpendicular)

        eyes_p_distance_at_std_depth = left_eye_p_std_depth | right_eye_p_std_depth
        eyes_depth_scale = self.std_eye_distance / eyes_p_distance_at_std_depth

        left_eye_3d = left_eye_p_std_depth * eyes_depth_scale
        right_eye_3d = right_eye_p_std_depth * eyes_depth_scale

        result.left_eye_3d = left_eye_3d
        result.right_eye_3d = right_eye_3d

        def visualize_3d(plt: vedo.Plotter):
            from vedo import Line, Image, Lines, Axes, Points, Text2D

            trail_points = 100
            self._visualization_trail_left.append(left_eye_3d)
            if len(self._visualization_trail_left) > trail_points:
                self._visualization_trail_left.pop(0)
            self._visualization_trail_right.append(right_eye_3d)
            if len(self._visualization_trail_right) > trail_points:
                self._visualization_trail_right.pop(0)

            # axes
            plt += Axes(
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

            image_half_size = Vec2(sin(self.fov_y / 2) * ar, sin(self.fov_y / 2))

            # image
            self._visualization_image = cv2.cvtColor(result._frame_view, cv2.COLOR_BGR2RGB, dst=self._visualization_image)
            image = Image(self._visualization_image)
            image.scale(image_half_size.y * 2 / self._visualization_image.shape[0])
            image.pos(*-image_half_size, z=-1)
            plt += image

            # eye rays
            plt += Line(Vec3(0), left_eye_p * 10, c="red")
            plt += Line(Vec3(0), right_eye_p * 10, c="green")
            plt += Points([left_eye_p], c="red")
            plt += Points([right_eye_p], c="green")

            plt += Line(Vec3(0), head_right_dir_q_perpendicular)
            plt += Points([left_eye_p_std_depth], c="blue")
            plt += Points([right_eye_p_std_depth], c="blue")
            plt += Line(left_eye_p_std_depth, head_right_dir_q_perpendicular)
            plt += Line(right_eye_p_std_depth, head_right_dir_q_perpendicular)

            plt += Points([left_eye_3d], r=8, c="red")
            plt += Points([right_eye_3d], r=8, c="green")
            plt += Line(Vec3(0), left_eye_3d, c="red")
            plt += Line(Vec3(0), right_eye_3d, c="green")
            plt += Line(left_eye_3d, right_eye_3d, c="blue")
            plt += Line(self._visualization_trail_left, c="red")
            plt += Line(self._visualization_trail_right, c="green")

            # camera frustum
            plt += Lines(
                start_pts=[Vec3(0)] * 4,
                end_pts=[Vec3(image_half_size * d, -1) for d in (Vec2(-1, -1), Vec2(1, -1), Vec2(1, 1), Vec2(-1, 1))],
                scale=10,
                alpha=0.5
            )

            plt += Text2D(f"Left:   {_fmt_vec3(left_eye_3d)}", c="red", font="VictorMono")
            plt += Text2D(f"\nRight:  {_fmt_vec3(right_eye_3d)}", c="green", font="VictorMono")
            plt += Text2D(f"\n\nCenter: {_fmt_vec3((left_eye_3d + right_eye_3d) / 2)}", c="blue", font="VictorMono")

        self._last_3d_visualizer = visualize_3d

    def _processor_thread_main(self):
        while True:
            result = self._results_queue.get()
            if not self.running:
                break
            self._process_result(result)
            self._last_result = result
            self.result_callback(result)

    def _camera_thread_main(self):
        def result_callback(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            if result.face_landmarks:
                self._results_queue.put(self.Result(
                    _face_landmarks=result.face_landmarks[0],
                    _face_transform=Transform3D(*result.facial_transformation_matrixes[0][:3].flatten("F")),
                    _frame_view=output_image.numpy_view()
                ))

        frame = None

        with mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                output_facial_transformation_matrixes=True,
                result_callback=result_callback
            )
        ) as landmarker:
            last_time_ms = 0
            while True:
                _, frame = self.camera.read(frame)
                time_ms = int(time.monotonic() * 1e3)
                if time_ms == last_time_ms:
                    continue
                landmarker.detect_async(
                    image=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
                    timestamp_ms=time_ms
                )
                # print(f"FPS: {1000 / (time_ms - last_time_ms)}")
                last_time_ms = time_ms

    def _visualization_thread_main(self):
        # vedo.settings.use_depth_peeling = True
        # vedo.settings.force_single_precision_points = False

        plt = vedo.Plotter(size=(1280, 800), interactive=True)

        def update(*_):
            if not self.running:
                plt.close()
                return

            #clear
            plt.clear(deep=True)

            visualizer = self._last_3d_visualizer
            if visualizer is not None:
                visualizer(plt)
            plt.render()

        plt.add_callback("timer", update, enable_picking=False)
        plt.timer_callback("create", dt=1)
        plt.parallel_projection()
        plt.show(viewup="y", title="MediaPipeEye3DPositioner")
        plt.close()
        self.running = False

    def start(self):
        self._processor_thread.start()
        if self._visualize:
            self._visualization_thread.start()
        self._camera_thread.start()

    def close(self):
        self.running = False
        self._processor_thread.join()
        if self._visualize:
            self._visualization_thread.join()
        self._camera_thread.join()

    # def _visualize_result(self, result: Result):
    #     frame = _draw_landmarks_on_image(result._frame_view, result._face_landmarks)
    #     if self.visualization_scale != 1.0:
    #         frame = cv2.resize(
    #             frame,
    #             (0, 0), fx=self.visualization_scale, fy=self.visualization_scale
    #         )
    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         return False
    #     return True
