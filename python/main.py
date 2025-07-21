import queue
import socket
import time
from math import radians
from threading import Thread
import json

import cv2

from media_pipe_eye_3d_positioning import MediaPipeEye3DPositioner
from spatium import *


packet_queue = queue.Queue(60)

def server_thread_main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 30001))
        print("Server started")
        while True:
            sock.listen()
            conn, addr = sock.accept()
            with conn:
                print(f"Accepted client: {addr}")
                while True:
                    try:
                        conn.sendall(packet_queue.get().encode())
                    except OSError as e:
                        print(f"Failed to send packet: {e}")
                        break

def vec3_json(v: Vec3) -> dict:
    return {"x": v.x, "y": v.y, "z": v.z}

def main():
    server_thread = Thread(target=server_thread_main, name="Server", daemon=True)
    server_thread.start()

    def result_callback(result: MediaPipeEye3DPositioner.Result):
        packet = json.dumps({
            "left_eye_3d": vec3_json(result.left_eye_3d),
            "right_eye_3d": vec3_json(result.right_eye_3d)
        }) + "\n"
        try:
            packet_queue.put_nowait(packet)
        except queue.Full:
            pass

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    camera.set(cv2.CAP_PROP_FPS, 60)

    positioner = MediaPipeEye3DPositioner(
        camera=camera,
        fov_y=radians(53),
        # std_eye_distance=5.8,
        std_eye_distance=6.0,
        visualize=True,
        result_callback=result_callback
    )

    positioner.start()
    while positioner.running:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
