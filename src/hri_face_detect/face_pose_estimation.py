import cv2
import numpy as np


def face_pose_estimation(points_2D, points_3D, K):
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
    return trans_vec, angles
