import socket
import json
import time
import os
import datetime as dt
import pandas as pd
import numpy as np
import vedo

def visualize_dataframe(filename: str):
    df = pd.read_csv(filename)
    N = df.shape[0]
    left_eye = df.loc[:, ["left_x", "left_y", "left_z"]].to_numpy()
    right_eye = df.loc[:, ["right_x", "right_y", "right_z"]].to_numpy()
    plt = vedo.Plotter(size=(1280, 800), interactive=True)
    plt += vedo.Line(left_eye, c="red")
    plt += vedo.Line(right_eye, c="green")
    plt.show(axes=2)

def main():
    filename = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(os.path.join("log", f"data-{filename}.csv"), "w+") as csv_file:
        csv_file.write("timestamp,left_x,left_y,left_z,right_x,right_y,right_z\n")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            ip = "127.0.0.1"
            port = 30001
            while True:
                try:
                    sock.connect((ip, port))
                    print(f"Connected to {ip}:{port}")
                    break
                except ConnectionRefusedError as e:
                    print(f"Failed to connect to {ip}:{port}, retrying...")
                    time.sleep(1)
                    continue

            buffer: str = ""
            error_flag = False
            while not error_flag:
                msg: str = ""
                try:
                    msg = sock.recv(1024).decode(encoding="utf-8")
                except OSError as e:
                    print(f"An error occured: {e}")
                    error_flag = True
                    break
                
                buffer += msg
                lines = buffer.splitlines()
                buffer = ""
                for i, line in enumerate(lines):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Since msg can at most be 1024 characters, a very long message might be broken up
                        # Check if the broken line is at the end of the lines. If so, add it back to the buffer
                        if i == len(lines) - 1:
                            buffer = line
                        else:
                            print(f"Invalid JSON format at line {i} of {len(lines)}: {line}")
                            error_flag = True
                        break
                    finally:
                        lx, ly, lz = obj["left_eye_3d"]["x"], obj["left_eye_3d"]["y"], obj["left_eye_3d"]["z"]
                        rx, ry, rz = obj["right_eye_3d"]["x"], obj["right_eye_3d"]["y"], obj["right_eye_3d"]["z"]
                        csv_file.write(f"{obj["time"]:.6f},{lx:.6f},{ly:.6f},{lz:.6f},{rx:.6f},{ry:.6f},{rz:.6f}\n")

if __name__ == "__main__":
    main()
    # visualize_dataframe(os.path.join("log", "data-2025-07-23-13-07-00.csv"))