# Copyright 2026 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import argparse
import os

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation

from exercise_2 import detect_corners

Coord3D = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Localization = tuple[Coord3D, Quaternion]

# Camera intrinsics (from dataset_info.yaml)
IMAGE_WIDTH = 1640
IMAGE_HEIGHT = 1232

K = np.array(
    [[365.77364921569824, 0.0, 820.0], [0.0, 365.773645401001, 616.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# Distortion coefficients [k1, k2, p1, p2, k3] - plumb_bob model
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)  # all zero = no distortion


def load_dataset(dataset_path: str) -> list[tuple[cv2.Mat, list[Localization]]]:
    dataset = []
    images_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_path):
        print(f'Images directory not found: {images_path}')
        return dataset

    labels_path = os.path.join(dataset_path, 'labels')
    if not os.path.exists(labels_path):
        print(f'Labels directory not found: {labels_path}')
        return dataset

    for filename in os.listdir(images_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_path, filename)
            image: np.ndarray | None = cv2.imread(image_path)
            if image is None:
                print(f'Failed to load image: {image_path}')
                continue
            h, w = image.shape[:2]

            label_path = os.path.join(
                labels_path, filename.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            labels: list[Localization] = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    parts = list(map(float, parts))
                    coords: Coord3D = (parts[0], parts[1], parts[2])
                    orient: Quaternion = (parts[3], parts[4], parts[5], parts[6])
                    labels.append((coords, orient))

            dataset.append((image, labels))

    return dataset


def translation_error(t1: Coord3D, t2: Coord3D) -> float:
    v1 = np.array(list(t1))
    v2 = np.array(list(t2))
    return float(np.linalg.norm(v1 - v2))


def rotation_error(o1: Quaternion, o2: Quaternion):
    q1 = np.array(o1)
    q2 = np.array(o2)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = abs(np.dot(q1, q2))

    dot = np.clip(dot, 0.0, 1.0)

    angle_rad = 2.0 * np.arccos(dot)
    return np.degrees(angle_rad)


def localize_gate(image: np.ndarray) -> list[Localization]:
    # TODO (Exercise 3): Use the code from previous tasks to calculate the camera position relative
    # to the gate.
    h, w = image.shape[:2]

    corners = detect_corners(image)
    if len(corners) != 8:
        return [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))]

    pts_2d = np.array([[cx * w, cy * h] for (cx, cy) in corners], dtype=np.float64)

    outer_gate_size = 2.7
    inner_gate_size = 1.5

    inner_top_y = -(outer_gate_size + inner_gate_size) / 2.0
    inner_bottom_y = -(outer_gate_size - inner_gate_size) / 2.0

    pts_3d = np.array([
        [-outer_gate_size / 2.0, -outer_gate_size, 0.0],  # TL outer 
        [ outer_gate_size / 2.0, -outer_gate_size, 0.0],  # TR outer 
        [-inner_gate_size / 2.0,  inner_top_y,     0.0],  # TL inner
        [ inner_gate_size / 2.0,  inner_top_y,     0.0],  # TR inner
        [-inner_gate_size / 2.0,  inner_bottom_y,  0.0],  # BL inner
        [ inner_gate_size / 2.0,  inner_bottom_y,  0.0],  # BR inner
        [-outer_gate_size / 2.0,  0.0,             0.0],  # BL outer 
        [ outer_gate_size / 2.0,  0.0,             0.0],  # BR outer 
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, DIST_COEFFS, flags=cv2.SOLVEPNP_SQPNP)
    if not success:
        return [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))]

    R_obj_to_cam, _ = cv2.Rodrigues(rvec)
    t_obj_to_cam = tvec

    R_cam_to_obj_cv = R_obj_to_cam.T
    t_cam_to_obj_cv = -R_cam_to_obj_cv @ t_obj_to_cam

    R_cv_to_world = np.array([
        [ 0.0,  0.0,  1.0],
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0]
    ])
    
    R_cam_to_cv_cam = np.array([
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 1.0,  0.0,  0.0]
    ])
    
    t_final_world = R_cv_to_world @ t_cam_to_obj_cv
    tx, ty, tz = t_final_world.flatten()

    R_final_camera_in_world = R_cv_to_world @ R_cam_to_obj_cv @ R_cam_to_cv_cam

    qx, qy, qz, qw = Rotation.from_matrix(R_final_camera_in_world).as_quat()
    
    return [((tx, ty, tz), (qw, qx, qy, qz))]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Load a dataset of images and segmentation labels.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to the dataset directory',
        default='../datasets/positions',
    )
    return parser.parse_args()


def visualize_3d(gt_pose, calc_pose):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    s_out = 2.7
    gate_x = [0, 0, 0, 0, 0]
    gate_y = [s_out/2, -s_out/2, -s_out/2, s_out/2, s_out/2]
    gate_z = [2.7, 2.7, 0.0, 0.0, 2.7]
    ax.plot(gate_x, gate_y, gate_z, 'k-', linewidth=3, label='Gate')

    def draw_pose(pose, name, line_style):
        t, q = pose
        tx, ty, tz = t
        
        qw, qx, qy, qz = q
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        
        ax.quiver(tx, ty, tz, R[0,0], R[1,0], R[2,0], color='r', length=0.8, linestyle=line_style)
        ax.quiver(tx, ty, tz, R[0,1], R[1,1], R[2,1], color='g', length=0.8, linestyle=line_style)
        ax.quiver(tx, ty, tz, R[0,2], R[1,2], R[2,2], color='b', length=0.8, linestyle=line_style)
        
        ax.scatter(tx, ty, tz, color='k', s=40)
        ax.text(tx, ty, tz + 0.2, name, color='k', fontsize=10)

    draw_pose(gt_pose, "Ground Truth", '-')
    draw_pose(calc_pose, "Estimated", '--')

    ax.set_xlabel('Global X (Forward/Backward)')
    ax.set_ylabel('Global Y (Left/Right)')
    ax.set_zlabel('Global Z (Up/Down)')
    
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlim([-4, 1])
    ax.set_ylim([-3, 3])  
    ax.set_zlim([0, 4])  
    
    plt.title("3D Global Coordinate Debugger")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    dataset = load_dataset(args.dataset_path)
    print(f'Loaded {len(dataset)} images and their corresponding labels.')

    result: list[tuple[np.ndarray, list[Localization], list[Localization]]] = []
    for image, labels in dataset:
        calculated_position: list[Localization] = localize_gate(image)
        result.append((image, labels, calculated_position))

    translation_errors: list[float] = []
    orientation_errors: list[float] = []

    for r in result:
        image: np.ndarray = r[0]
        gt_pose: Localization = r[1][0]
        calculated_pose: Localization = r[2][0]

        t_error = translation_error(gt_pose[0], calculated_pose[0])
        r_error = rotation_error(gt_pose[1], calculated_pose[1])

        translation_errors.append(t_error)
        orientation_errors.append(r_error)
        #visualize_3d(gt_pose, calculated_pose)
        #cv2.waitKey(0)

    print('Mean translation error = ', np.mean(translation_errors))
    print('Mean orientation error = ', np.mean(orientation_errors))
