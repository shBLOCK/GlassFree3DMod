
from dataclasses import dataclass
from threading import Thread
import time
import numpy as np
import cv2
import vedo
from spatium import *
from mediapipe.tasks.python.components.containers import landmark as landmark_module
from math import radians
from typing_extensions import Optional

@dataclass
class CameraData:
    id: int
    resolution: Vec2i
    fov_y: float
    position: Vec3
    orientation: Vec3

config: list[CameraData] = [
    # camera 1
    # CameraData(id=1, resolution=Vec2(1920, 1080), fov_y=radians(53), position=Vec3(0.0), orientation=Vec3(0.0)),
    # CameraData(id=2, resolution=Vec2(1920, 1080), fov_y=radians(53), position=Vec3(0.0), orientation=Vec3(0.0)),
    CameraData(id=0, resolution=Vec2(1280, 720), fov_y=radians(53), position=Vec3(0.0), orientation=Vec3(0.0)),
]

def calibrate_camera(chessboard_dim: Vec2i) -> list[Transform3D]:
    """
    Calibrate all cameras listed in the config array.

    Returns the list of transformation matrices of camera index 1~n with respect to camera index 0.

    # Arguments
    - `chessboard_dim`: dimensions (width, height) of the chessboard's intersections.
    """
    chessboard_dim_tuple = (chessboard_dim.x, chessboard_dim.y)
    exit_signal = False
    captures = [cv2.VideoCapture(cam_data.id) for cam_data in config]
    for i, capture in enumerate(captures):
        if not capture.isOpened():
            print(f"相机{id}无法打开")
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config[i].resolution.x)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config[i].resolution.y)
    def video_stream_main(index: int):
        id = config[index].id
        while not exit_signal:
            ret, frame = captures[index].read()
            if not ret:
                print(f"无法从相机{id}中获取视频流")
            cv2.imshow(f"frame-{id}", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyWindow(f"frame-{id}")
    video_stream_threads = [Thread(target=video_stream_main, args=(i,)) for i in range(len(captures))]
    for thread in video_stream_threads:
        thread.start()

    # objpoints[k][i, n, x]: kth capture, ith camera, nth point, (x=0, y=1, z=2)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    # captured_imgs[k][i, :]: kth capture, ith camera
    captured_imgs: list[np.ndarray] = []
    # camera_shapes: Optional[list] = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_dim.x * chessboard_dim.y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_dim.x, 0:chessboard_dim.y].T.reshape(-1, 2)
    # start taking photos
    while not exit_signal:
        print(f"objpoints - {len(objpoints)}, imgpoints - {len(imgpoints)}")
        print("请按回车键拍照，键入q再回车退出标定程序")
        line = input()
        if line == "q" or line == "Q":
            exit_signal = True
            break
        one_capture = [capture.read()[1] for capture in captures]
        grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in one_capture]
        # if camera_shapes == None:
        #     camera_shapes = [gray.shape[::-1] for gray in grays]

        cur_objpts = []
        cur_imgpts = []
        continue_signal = False
        for i, gray in enumerate(grays):
            ret, corner = cv2.findChessboardCorners(gray, chessboard_dim_tuple, None)
            if not ret:
                print(f"相机{config[i].id}拍摄的照片无法识别棋盘格")
                continue_signal = True
                break
            cur_objpts.append(objp)
            corners2 = cv2.cornerSubPix(gray, corner, (11, 11), (-1, -1), criteria)
            cur_imgpts.append(corners2)
        if continue_signal:
            continue
        # draw all chessboard images
        for i in range(len(one_capture)):
            id = config[i].id
            cv2.drawChessboardCorners(one_capture[i], chessboard_dim_tuple, cur_imgpts[i], True)
            cv2.imshow(f"chessboard-{id}", one_capture[i])
            cv2.waitKey(500)
        cur_objpts = np.array(cur_objpts)
        cur_imgpts = np.array(cur_imgpts)
        grays = np.array(grays)
        objpoints.append(cur_objpts)
        imgpoints.append(cur_imgpts)
        captured_imgs.append(grays)

    print(f"拍照完毕，各个相机共有{len(objpoints)}张照片")
    if len(objpoints) == 0:
        return
    print("开始进行标定")
    for thread in video_stream_threads:
        thread.join()
    cv2.destroyAllWindows()

    camera_params = []
    corrected_imgs: list[np.ndarray] = [np.zeros(captured_img.shape) for captured_img in captured_imgs]
    for i in range(len(captures)):
        id = config[i].id
        print("-" * 20)
        print(f"相机{id}标定数据：")
        objpoint = [objpoints[n][i, :, :] for n in range(len(objpoints))]
        imgpoint = [imgpoints[n][i, :, :] for n in range(len(imgpoints))]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoint, imgpoint, chessboard_dim_tuple, None, None)
        print("内参矩阵：")
        print(mtx)
        print("畸变参数：")
        print(dist)
        camera_params.append((mtx, dist, rvecs, tvecs))
        for j in range(len(captured_imgs)):
            undistorted_img = cv2.undistort(captured_imgs[j][i, :, :], mtx, dist)
            cv2.imshow(f"window-{j}", undistorted_img)
            cv2.waitKey()
        # TODO: 我也不知道要怎么算了啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
        # 救救孩子


def main():
    pass

if __name__ == "__main__":
    # main()
    calibrate_camera(chessboard_dim=Vec2i(11, 8))