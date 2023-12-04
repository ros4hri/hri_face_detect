# Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
from dataclasses import dataclass
import numpy as np
from tf_transformations import quaternion_from_euler


@dataclass
class VectorData:
    x: float
    y: float
    z: float


@dataclass
class QuaternionData:
    x: float
    y: float
    z: float
    w: float


def face_pose_estimation(points_2D, points_3D, K) -> (VectorData, QuaternionData):
    _, rot_vec, trans_vec = cv2.solvePnP(
        points_3D,
        points_2D,
        K,
        None,
        tvec=np.array([0.0, 0.0, 1000.0]),
        useExtrinsicGuess=True,
        flags=4,
    )
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    angles = np.array(angles)
    angles /= (180/np.pi)

    trans_vec = VectorData(*[(coord / 1000) for coord in trans_vec])
    angles_quaternion = QuaternionData(*quaternion_from_euler(*angles))

    return trans_vec, angles_quaternion
