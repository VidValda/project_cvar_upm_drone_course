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
import numpy as np

CornerLabel = tuple[float, float]


def sort_corners(points):
    return sorted(points, key=lambda p: (p[1], p[0]))

def order_points(pts: np.ndarray) -> np.ndarray:
    xSorted = pts[np.argsort(pts[:, 0]), :]
    
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    return np.array([tl, tr, bl, br])

def detect_corners(image: np.ndarray) -> list[CornerLabel]:
    # TODO (Exercise 2): Implement a corner detection algorithm that takes an image as input and returns a list of CornerLabel objects.
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    s = cv2.medianBlur(s,7)

    _,binary_img = cv2.threshold(s,50,255,cv2.THRESH_BINARY)

    kernel = np.ones((15,15))
    binary_img = cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,kernel)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(sorted_contours) < 2:
        return [] 
        
    outer_contour = sorted_contours[0]
    inner_contour = sorted_contours[1]
    
    corners = [None] * 8
    
    for idx, contour in enumerate([outer_contour, inner_contour]):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            tl, tr, bl, br = order_points(pts)

            if idx == 0:
                corners[0] = (float(tl[0]) / w, float(tl[1]) / h)
                corners[1] = (float(tr[0]) / w, float(tr[1]) / h)
                corners[6] = (float(bl[0]) / w, float(bl[1]) / h)
                corners[7] = (float(br[0]) / w, float(br[1]) / h)
            else:
                corners[2] = (float(tl[0]) / w, float(tl[1]) / h)
                corners[3] = (float(tr[0]) / w, float(tr[1]) / h)
                corners[4] = (float(bl[0]) / w, float(bl[1]) / h)
                corners[5] = (float(br[0]) / w, float(br[1]) / h)
                
    if all(c is not None for c in corners):
        return corners
    return []


def load_dataset(dataset_path: str) -> list[tuple[cv2.Mat, list[CornerLabel]]]:
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
            labels: list[CornerLabel] = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    parts = parts[5:]
                    for i in range(0, len(parts), 3):
                        x = float(parts[i])
                        y = float(parts[i + 1])
                        labels.append((x, y))

            labels = sort_corners(labels)

            dataset.append((image, labels))
    return dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Load a dataset of images and segmentation labels.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to the dataset directory',
        default='../datasets/yolo_pose',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    dataset = load_dataset(args.dataset_path)
    print(f'Loaded {len(dataset)} images and their corresponding labels.')

    result: list[list[CornerLabel]] = []
    for image, labels in dataset:
        corners = detect_corners(image)
        corners = sort_corners(corners)
        result.append(corners)

    distances = [1]
    for corners, labels in zip(result, dataset):
        for corner, label in zip(corners, labels[1]):
            distance = np.sqrt((corner[0] - label[0]) ** 2 + (corner[1] - label[1]) ** 2)
            distances.append(distance)

    print(f'Average distance: {np.mean(distances)}')
