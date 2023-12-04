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

from builtin_interfaces.msg import Time as Stamp
from copy import deepcopy
from cv_bridge import CvBridge
import cv2
from dataclasses import asdict, astuple, dataclass, InitVar
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from hri_face_detect import (
    face_pose_estimation, OneEuroFilter, QuaternionData, QuatOneEuroFilter, YuNetDetector)
from hri_msgs.msg import (
    FacialLandmarks, IdsList, NormalizedPointOfInterest2D, NormalizedRegionOfInterest2D)
from lifecycle_msgs.msg import State
import math
from mediapipe.python.solutions.face_mesh import FaceMesh
import numpy as np
from PIL import Image as PILImage
import random
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.lifecycle import Node, TransitionCallbackReturn
from rclpy.lifecycle.node import LifecycleState
from rclpy.time import Time
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from tf_transformations import quaternion_from_euler
from tf2_geometry_msgs.tf2_geometry_msgs import Point, PointStamped
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from threading import Lock
from typing import Dict, List, TypeAlias

ColorType: TypeAlias = 'numpy.dtype[numpy.uint8]'
ThreeD: TypeAlias = tuple[int, int, int]
RGBMat: TypeAlias = 'numpy.ndarray[ThreeD, ColorType]'

# minimum image fraction of a face detection bounding box to trigger face mesh detection
MIN_FACE_IMAGE_FRAC_FOR_MESH = 0.02

# max nb of face meshes detected in the same frame
MAX_FACE_MASH_N = 4

# min number of frames a face is detected before it is tracked
MIN_FRAMES_FACE_TRACKING = 4

# max number of frames a face is not detected before it is not tracked anymore
MAX_FRAMES_FACE_RETENTION = 4

# max distance, relative to RoI diagonal, between the RoI centers of two
# successive detections to consider they belong to the same person
MAX_ROIS_REL_DISTANCE = 0.5

# max scale factor between two successive
# regions of interest to consider they
# belong to the same person
MAX_SCALING_ROIS = 1.5

# default size in pixels for the re-published faces
# can be changed via the ROS parameters
# /humans/faces/width and /humans/faces/height
cropped_face_width = 128
cropped_face_height = 128

# filtering parameters
BETA_POSITION = 0.05
BETA_QUATERNION = 0.01
D_CUTOFF_POSITION = 0.5
MIN_CUTOFF_POSITION = 0.3

# diagnostic message publish rate in Hertz
DIAG_PUB_RATE = 1

# face detection processing time in ms triggering a diagnostic warning
FACE_DETECTION_PROC_TIME_WARN = 500.
# face detection processing time in ms triggering a diagnostic error
FACE_DETECTION_PROC_TIME_ERROR = 2000.

# average adult male face key points in mm and own face reference frame
P3D_RIGHT_EYE = (-20.0, -65.5, -5.0)
P3D_LEFT_EYE = (-20.0, 65.5, -5.0)
P3D_RIGHT_EAR = (-100.0, -77.5, -6.0)
P3D_LEFT_EAR = (-100.0, 77.5, -6.0)
P3D_NOSE = (21.0, 0.0, -48.0)
P3D_STOMION = (10.0, 0.0, -75.0)

# ROS4HRI to mediapipe landmarks mapping
# ROS4HRI FacialLandmarks ref:
# https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/.github/media/keypoints_face.png  # noqa
# Mediapipe Landmarks refs:
# https://i.stack.imgur.com/5Mohl.jpg
# https://developers.google.com/static/mediapipe/images/solutions/face_landmarker_keypoints.png
ros4hri_to_mp_landmarks_mapping = {
    FacialLandmarks.RIGHT_EAR: 34,
    FacialLandmarks.RIGHT_PROFILE_1: 227,
    FacialLandmarks.RIGHT_PROFILE_2: 137,
    FacialLandmarks.RIGHT_PROFILE_3: 177,
    FacialLandmarks.RIGHT_PROFILE_4: 215,
    FacialLandmarks.RIGHT_PROFILE_5: 135,
    FacialLandmarks.RIGHT_PROFILE_6: 170,
    FacialLandmarks.RIGHT_PROFILE_7: 171,
    FacialLandmarks.MENTON: 175,
    FacialLandmarks.LEFT_EAR: 264,
    FacialLandmarks.LEFT_PROFILE_1: 447,
    FacialLandmarks.LEFT_PROFILE_2: 366,
    FacialLandmarks.LEFT_PROFILE_3: 401,
    FacialLandmarks.LEFT_PROFILE_4: 435,
    FacialLandmarks.LEFT_PROFILE_5: 364,
    FacialLandmarks.LEFT_PROFILE_6: 395,
    FacialLandmarks.LEFT_PROFILE_7: 396,
    FacialLandmarks.RIGHT_EYEBROW_OUTSIDE: 70,
    FacialLandmarks.RIGHT_EYEBROW_1: 63,
    FacialLandmarks.RIGHT_EYEBROW_2: 105,
    FacialLandmarks.RIGHT_EYEBROW_3: 66,
    FacialLandmarks.RIGHT_EYEBROW_INSIDE: 107,
    FacialLandmarks.LEFT_EYEBROW_OUTSIDE: 300,
    FacialLandmarks.LEFT_EYEBROW_1: 293,
    FacialLandmarks.LEFT_EYEBROW_2: 334,
    FacialLandmarks.LEFT_EYEBROW_3: 296,
    FacialLandmarks.LEFT_EYEBROW_INSIDE: 336,
    FacialLandmarks.RIGHT_EYE_OUTSIDE: 130,
    FacialLandmarks.RIGHT_EYE_TOP_1: 29,
    FacialLandmarks.RIGHT_EYE_TOP_2: 28,
    FacialLandmarks.RIGHT_EYE_INSIDE: 243,
    FacialLandmarks.RIGHT_EYE_BOTTOM_1: 24,
    FacialLandmarks.RIGHT_EYE_BOTTOM_2: 22,
    FacialLandmarks.LEFT_EYE_OUTSIDE: 359,
    FacialLandmarks.LEFT_EYE_TOP_1: 259,
    FacialLandmarks.LEFT_EYE_TOP_2: 258,
    FacialLandmarks.LEFT_EYE_INSIDE: 463,
    FacialLandmarks.LEFT_EYE_BOTTOM_1: 254,
    FacialLandmarks.LEFT_EYE_BOTTOM_2: 252,
    FacialLandmarks.SELLION: 6,
    FacialLandmarks.NOSE_1: 197,
    FacialLandmarks.NOSE_2: 4,
    FacialLandmarks.NOSE: 1,
    FacialLandmarks.NOSTRIL_1: 242,
    FacialLandmarks.NOSTRIL_2: 141,
    FacialLandmarks.NOSTRIL_3: 94,
    FacialLandmarks.NOSTRIL_4: 370,
    FacialLandmarks.NOSTRIL_5: 462,
    FacialLandmarks.MOUTH_OUTER_RIGHT: 61,
    FacialLandmarks.MOUTH_OUTER_TOP_1: 40,
    FacialLandmarks.MOUTH_OUTER_TOP_2: 37,
    FacialLandmarks.MOUTH_OUTER_TOP_3: 0,
    FacialLandmarks.MOUTH_OUTER_TOP_4: 267,
    FacialLandmarks.MOUTH_OUTER_TOP_5: 270,
    FacialLandmarks.MOUTH_OUTER_LEFT: 291,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_1: 321,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_2: 314,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_3: 17,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_4: 84,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_5: 91,
    FacialLandmarks.MOUTH_INNER_RIGHT: 62,
    FacialLandmarks.MOUTH_INNER_TOP_1: 41,
    FacialLandmarks.MOUTH_INNER_TOP_2: 12,
    FacialLandmarks.MOUTH_INNER_TOP_3: 271,
    FacialLandmarks.MOUTH_INNER_LEFT: 292,
    FacialLandmarks.MOUTH_INNER_BOTTOM_1: 403,
    FacialLandmarks.MOUTH_INNER_BOTTOM_2: 15,
    FacialLandmarks.MOUTH_INNER_BOTTOM_3: 179,
    FacialLandmarks.RIGHT_PUPIL: 468,
    FacialLandmarks.LEFT_PUPIL: 473
}

ROS4HRI_LANDMARKS_N = 70


def bound(val, min_val, max_val):
    return max(min_val, min(val, max_val))


@dataclass
class ImagePoint:
    x: int
    y: int
    image_width: InitVar(int)
    image_height: InitVar(int)

    def __post_init__(self, image_width, image_height):
        self.x = bound(self.x, 0, image_width - 1)
        self.y = bound(self.y, 0, image_height - 1)


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    width: int
    height: int
    image_width: InitVar(int)
    image_height: InitVar(int)

    def __post_init__(self, image_width, image_height):
        xmax = self.xmin + self.width
        ymax = self.ymin + self.height
        self.xmin = bound(self.xmin, 0, image_width - 1)
        self.ymin = bound(self.ymin, 0, image_height - 1)
        self.width = bound(xmax - self.xmin, 0, image_width - 1 - self.xmin)
        self.height = bound(ymax - self.ymin, 0, image_height - 1 - self.ymin)

    def diag_length(self):
        return math.dist((0, 0), (self.width, self.height))


@dataclass
class FaceDetection:
    score: float
    bb: BoundingBox
    landmarks: Dict[int, ImagePoint]  # FacialLandmarks to ImagePoint

    def __post_init__(self):
        self.score = bound(self.score, 0., 1.)


@dataclass
class MeshDetection:
    bb: BoundingBox
    landmarks: Dict[int, ImagePoint]  # FacialLandmarks to ImagePoint


def normalized_to_pixel_coordinates(
        x_norm: float, y_norm: float, image_width: int, image_height: int) -> (float, float):
    x_px = bound(int(x_norm * image_width), 0, image_width - 1)
    y_px = bound(int(y_norm * image_height), 0, image_height - 1)
    return x_px, y_px


def pixel_to_normalized_coordinates(
        x_px: int, y_px: int, image_width: int, image_height: int) -> (int, int):
    x_norm = bound(x_px / image_width, 0., 1.)
    y_norm = bound(y_px / image_height, 0., 1.)
    return x_norm, y_norm


def distance_rois(bb1: BoundingBox, bb2: BoundingBox) -> float:
    x1, y1 = bb1.xmin + bb1.width / 2, bb1.ymin + bb1.height / 2
    x2, y2 = bb2.xmin + bb2.width / 2, bb2.ymin + bb2.height / 2
    return math.dist((x1, y1), (x2, y2))


def bbs_match(bb1: BoundingBox, bb2: BoundingBox) -> bool:
    return (
        (distance_rois(bb1, bb2) / bb1.diag_length()) < MAX_ROIS_REL_DISTANCE
        and (1 / MAX_SCALING_ROIS) < (bb1.width / bb2.width) < MAX_SCALING_ROIS
        and (1 / MAX_SCALING_ROIS) < (bb1.height / bb2.height) < MAX_SCALING_ROIS)


class Face:
    last_id = 0

    def __init__(
            self, node: Node, tf_buffer: Buffer, filtering_frame: str,
            deterministic_id: bool = False):

        # generate unique ID
        if deterministic_id:
            self.id = 'f%05d' % Face.last_id
            Face.last_id = (Face.last_id + 1) % 10000
        else:
            # for a 5 char long ID
            self.id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))

        self.initial_detection_time: Time = None
        self.nb_frames_visible = 0
        self.nb_frames_since_last_detection = 0

        self.score = 0.
        self.bb: BoundingBox = None
        self.landmarks: Dict[FacialLandmarks, ImagePoint] = dict()

        self.node = node
        self.tf_buffer = tf_buffer
        self.filtering_frame = filtering_frame
        self.head_transform: TransformStamped = None
        self.gaze_transform: TransformStamped = None
        self.one_euro_filters_xyz: List[OneEuroFilter] = [None]*3
        self.one_euro_filter_quaternion: List[QuatOneEuroFilter] = None

        # True once the publishers are initialised
        self.ready = False
        self.do_publish = False

    def initialise_publishers(self):
        """
        Initialise all the publishers for this face.

        Not done in the constructor as we typically wait for a few frames
        before publishing anything (to avoid creating too many spurious faces
        due to detection noise).
        """
        # already initialised?
        if self.ready:
            return

        ns = f'/humans/faces/{self.id}'
        self.roi_pub = self.node.create_publisher(NormalizedRegionOfInterest2D, ns + '/roi', 1)
        self.cropped_pub = self.node.create_publisher(Image, ns + '/cropped', 1)
        self.aligned_pub = self.node.create_publisher(Image, ns + '/aligned', 1)
        self.landmarks_pub = self.node.create_publisher(FacialLandmarks, ns + '/landmarks', 1)

        self.node.get_logger().info(f'New face: {self}')
        self.ready = True

    def publish(self, src_image: RGBMat, image_msg_header: Header):
        if not self.ready:
            self.node.get_logger().error(
                'Trying to publish face information but publishers have not been created yet!')
            return

        self.publish_normalized_roi(src_image, image_msg_header)
        self.publish_facial_landmarks(src_image, image_msg_header)
        self.publish_cropped_face(src_image, image_msg_header)
        self.publish_aligned_face(src_image, image_msg_header)

    def publish_normalized_roi(self, src_image: RGBMat, image_msg_header: Header):
        img_height, img_width, _ = src_image.shape
        msg = NormalizedRegionOfInterest2D()
        msg.header = image_msg_header
        msg.xmin, msg.ymin = pixel_to_normalized_coordinates(
            self.bb.xmin, self.bb.ymin, img_width, img_height)
        msg.xmax, msg.ymax = pixel_to_normalized_coordinates(
            self.bb.xmin + self.bb.width, self.bb.ymin + self.bb.height, img_width, img_height)
        msg.c = self.score
        self.roi_pub.publish(msg)

    def publish_facial_landmarks(self, src_image: RGBMat, image_msg_header: Header):
        img_height, img_width, _ = src_image.shape
        msg = FacialLandmarks()
        msg.header = image_msg_header
        msg.landmarks = [NormalizedPointOfInterest2D() for _ in range(ROS4HRI_LANDMARKS_N)]
        for idx, landmark in self.landmarks.items():
            x, y = pixel_to_normalized_coordinates(landmark.x, landmark.y, img_width, img_height)
            msg.landmarks[idx].x = x
            msg.landmarks[idx].y = y
            msg.landmarks[idx].c = self.score
        msg.height = img_height
        msg.width = img_width
        self.landmarks_pub.publish(msg)

    def publish_cropped_face(self, src_image: RGBMat, image_msg_header: Header):
        # no-one interested in the face image? skip it!
        if self.cropped_pub.get_subscription_count() == 0:
            return

        roi = src_image[
            self.bb.ymin:(self.bb.ymin + self.bb.height),
            self.bb.xmin:(self.bb.xmin + self.bb.width)]

        sx = cropped_face_width * 1.0 / self.bb.width
        sy = cropped_face_height * 1.0 / self.bb.height

        scale = min(sx, sy)

        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        scaled_h, scaled_w = scaled.shape[:2]

        output = np.zeros((cropped_face_width, cropped_face_height, 3), np.uint8)

        x_offset = int((cropped_face_width - scaled_w) / 2)
        y_offset = int((cropped_face_height - scaled_h) / 2)

        output[y_offset:(y_offset+scaled_h), x_offset:(x_offset + scaled_w)] = scaled

        msg = CvBridge().cv2_to_imgmsg(output, encoding='bgr8')
        msg.header = image_msg_header
        self.cropped_pub.publish(msg)

    def publish_aligned_face(self, src_image: RGBMat, image_msg_header: Header):
        """
        Align given face in img based on left and right eye coordinates.

        This function is adapted from MIT-licensed DeepFace.
        Author: serengil
        Original source:
        https://github.com/serengil/deepface/blob/f07f278/deepface/detectors/FaceDetector.py#L68
        """
        # no-one interested in the face image? skip it!
        if self.aligned_pub.get_subscription_count() == 0:
            return

        img_height, img_width, _ = src_image.shape
        x, y, w, h = self.bb.xmin, self.bb.ymin, self.bb.width, self.bb.height

        # expand the ROI a little to ensure the rotation does not introduce black zones
        xm1 = max(0, x - w // 2)
        xm2 = min(x + w + w // 2, img_width)
        ym1 = max(0, y - h // 2)
        ym2 = min(y + h + h // 2, img_height)
        preroi = src_image[ym1:ym2, xm1:xm2]

        left_eye = self.landmarks[FacialLandmarks.LEFT_PUPIL]
        right_eye = self.landmarks[FacialLandmarks.RIGHT_PUPIL]

        # -----------------------
        # find rotation direction

        if left_eye.y > right_eye.y:
            point_3rd = (right_eye.x, left_eye.y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye.x, right_eye.y)
            direction = 1  # rotate inverse direction of clock

        # find length of triangle edges
        a = math.dist(astuple(left_eye), point_3rd)
        b = math.dist(astuple(right_eye), point_3rd)
        c = math.dist(astuple(right_eye), astuple(left_eye))

        # apply cosine rule
        if b != 0 and c != 0:

            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / math.pi  # radian to degree

            # rotate base image

            if direction == -1:
                angle = 90 - angle

            img = PILImage.fromarray(preroi)  # convert to a PIL image to rotate it
            preroi = np.array(img.rotate(-direction * angle, PILImage.BILINEAR))

        roi = preroi[(y - ym1):(y - ym1 + h), (x - xm1):(x - xm1 + w)]

        sx = cropped_face_width * 1.0 / w
        sy = cropped_face_height * 1.0 / h

        scale = min(sx, sy)

        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        scaled_h, scaled_w = scaled.shape[:2]

        output = np.zeros((cropped_face_width, cropped_face_height, 3), np.uint8)

        x_offset = int((cropped_face_width - scaled_w) / 2)
        y_offset = int((cropped_face_height - scaled_h) / 2)

        output[y_offset:(y_offset + scaled_h), x_offset:(x_offset + scaled_w)] = scaled

        msg = CvBridge().cv2_to_imgmsg(output, encoding='bgr8')
        msg.header = image_msg_header
        self.aligned_pub.publish(msg)

    def compute_6d_pose(
            self, K: np.ndarray, camera_optical_frame: str, stamp: Stamp):
        # use the face mesh landmarks to compute the pose if all the necessary ones are found,
        # otherwise use the ones extracted by the face detector which are guaranteed
        landmarks_2d_to_3d = {
            FacialLandmarks.RIGHT_PUPIL: P3D_RIGHT_EYE,
            FacialLandmarks.LEFT_PUPIL: P3D_LEFT_EYE,
            FacialLandmarks.RIGHT_EAR: P3D_RIGHT_EAR,
            FacialLandmarks.LEFT_EAR: P3D_LEFT_EAR,
            FacialLandmarks.NOSE: P3D_NOSE,
            FacialLandmarks.MOUTH_INNER_TOP_2: P3D_STOMION}

        if all(lm_key in self.landmarks for lm_key in landmarks_2d_to_3d.keys()):
            points_2D = np.array(
                [astuple(self.landmarks[lm_key]) for lm_key in landmarks_2d_to_3d.keys()],
                dtype='double')
            points_3D = np.array(list(landmarks_2d_to_3d.values()))
        else:
            # compute stomion from average of left and right mouth
            mouth_right = self.landmarks[FacialLandmarks.MOUTH_OUTER_RIGHT]
            mouth_left = self.landmarks[FacialLandmarks.MOUTH_OUTER_LEFT]
            landmark_stomion = (
                (mouth_right.x + mouth_left.x) / 2,
                (mouth_right.y + mouth_left.y) / 2)
            points_2D = np.array(
                [
                    astuple(self.landmarks[FacialLandmarks.RIGHT_PUPIL]),
                    astuple(self.landmarks[FacialLandmarks.LEFT_PUPIL]),
                    astuple(self.landmarks[FacialLandmarks.NOSE]),
                    landmark_stomion
                ], dtype='double')
            points_3D = np.array([P3D_RIGHT_EYE, P3D_LEFT_EYE, P3D_NOSE, P3D_STOMION])

        trans_vec, angles_quaternion = face_pose_estimation(points_2D, points_3D, K)

        if self.tf_buffer and self.filtering_frame:
            point_trans_vec = PointStamped(
                header=Header(frame_id=camera_optical_frame, stamp=stamp),
                point=Point(**asdict(trans_vec)))

            try:
                point_trans_vec = self.tf_buffer.transform(point_trans_vec, self.filtering_frame)
                trans_vec.x = point_trans_vec.point.x
                trans_vec.y = point_trans_vec.point.y
                trans_vec.z = point_trans_vec.point.z
            except Exception as e:
                self.node.get_logger().debug(
                    f'Unable to transform from frame "{camera_optical_frame}" to filtering frame '
                    f'"{self.filtering_frame}" (not the same tree?). Leaving the point in  its '
                    f'original frame. Original error was: {e}')

        current_time = self.node.get_clock().now().nanoseconds / 1e9

        if not self.one_euro_filters_xyz[0]:
            self.one_euro_filters_xyz[0] = OneEuroFilter(
                current_time,
                trans_vec.x,
                beta=BETA_POSITION,
                d_cutoff=D_CUTOFF_POSITION,
                min_cutoff=MIN_CUTOFF_POSITION)
            self.one_euro_filters_xyz[1] = OneEuroFilter(
                current_time,
                trans_vec.y,
                beta=BETA_POSITION,
                d_cutoff=D_CUTOFF_POSITION,
                min_cutoff=MIN_CUTOFF_POSITION)
            self.one_euro_filters_xyz[2] = OneEuroFilter(
                current_time,
                trans_vec.z,
                beta=BETA_POSITION,
                d_cutoff=D_CUTOFF_POSITION,
                min_cutoff=MIN_CUTOFF_POSITION)
        else:
            trans_vec.x = self.one_euro_filters_xyz[0](current_time, trans_vec.x)[0]
            trans_vec.y = self.one_euro_filters_xyz[1](current_time, trans_vec.y)[0]
            trans_vec.z = self.one_euro_filters_xyz[2](current_time, trans_vec.z)[0]

        if not self.one_euro_filter_quaternion:
            self.one_euro_filter_quaternion = QuatOneEuroFilter(
                current_time, astuple(angles_quaternion), MIN_CUTOFF_POSITION, BETA_QUATERNION)
        else:
            angles_quaternion = QuaternionData(
                *self.one_euro_filter_quaternion(current_time, astuple(angles_quaternion)))

        if self.tf_buffer and self.filtering_frame:
            point_trans_vec = PointStamped(
                header=Header(frame_id=self.filtering_frame, stamp=stamp),
                point=Point(**asdict(trans_vec)))

            try:
                point_trans_vec = self.tf_buffer.transform(point_trans_vec, camera_optical_frame)
            except Exception as e:
                self.node.get_logger().debug(
                    f'An error occured while transforming from frame "{self.filtering_frame}" to '
                    f'filtering frame "{camera_optical_frame}". Original error was: {e}')

            trans_vec.x = point_trans_vec.point.x
            trans_vec.y = point_trans_vec.point.y
            trans_vec.z = point_trans_vec.point.z

        # calculating angle
        self.head_transform = TransformStamped()
        self.head_transform.header = Header(stamp=stamp, frame_id=camera_optical_frame)
        self.head_transform.child_frame_id = f'face_{self.id}'
        self.head_transform.transform.translation = Vector3(**asdict(trans_vec))
        self.head_transform.transform.rotation = Quaternion(**asdict(angles_quaternion))

        self.gaze_transform = TransformStamped()
        self.gaze_transform.header = Header(stamp=stamp, frame_id=f'face_{self.id}')
        self.gaze_transform.child_frame_id = f'gaze_{self.id}'
        self.gaze_transform.transform.translation = Vector3(x=0., y=0., z=0.)
        self.gaze_transform.transform.rotation = Quaternion(
            **asdict(QuaternionData(*quaternion_from_euler(-np.pi / 2, 0, -np.pi / 2))))

    def __del__(self):
        if not self.ready:
            return

        detect_time = (self.node.get_clock().now() - self.initial_detection_time).nanoseconds / 1e9
        self.node.get_logger().info(
            f'Face [{self}] lost. It remained visible for {detect_time:.2f}sec')

        self.node.destroy_publisher(self.roi_pub)
        self.node.destroy_publisher(self.cropped_pub)
        self.node.destroy_publisher(self.aligned_pub)
        self.node.destroy_publisher(self.landmarks_pub)

        self.ready = False

    def __repr__(self):
        return self.id


class FaceDetector:
    def __init__(self, confidence_threshold: float, image_scale: float):
        self.confidence_threshold = confidence_threshold
        self.image_scale = image_scale
        self.detector = YuNetDetector()

    @staticmethod
    def _extract_face_detection(
            raw_detection: List, scale: float, image_width: int, image_height: int
            ) -> FaceDetection:
        score = float(raw_detection[0]) / 100.
        scaled_raw_coords = [int(x*scale) for x in raw_detection[1:15]]
        bb = BoundingBox(*scaled_raw_coords[0:4], image_width, image_height)

        landmarks: Dict[FacialLandmarks, ImagePoint] = dict()
        landmarks[FacialLandmarks.RIGHT_PUPIL] = ImagePoint(
            *scaled_raw_coords[4:6], image_width, image_height)
        landmarks[FacialLandmarks.LEFT_PUPIL] = ImagePoint(
            *scaled_raw_coords[6:8], image_width, image_height)
        landmarks[FacialLandmarks.NOSE] = ImagePoint(
            *scaled_raw_coords[8:10], image_width, image_height)
        landmarks[FacialLandmarks.MOUTH_OUTER_RIGHT] = ImagePoint(
            *scaled_raw_coords[10:12], image_width, image_height)
        landmarks[FacialLandmarks.MOUTH_OUTER_LEFT] = ImagePoint(
            *scaled_raw_coords[12:14], image_width, image_height)

        return FaceDetection(score, bb, landmarks)

    def detect(self, img: RGBMat) -> List[FaceDetection]:
        img_height, img_width, _ = img.shape

        scaled_img = cv2.resize(
            img, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA)
        scaled_img_height, scaled_img_width, _ = scaled_img.shape

        raw_face_detections = self.detector.detect(
            scaled_img, scaled_img_width, scaled_img_height, scaled_img.strides[0])
        face_detections = [
            self._extract_face_detection(d, 1./self.image_scale, img_width, img_height)
            for d in raw_face_detections]
        valid_face_detections = [
            d for d in face_detections
            if d.score > self.confidence_threshold and d.bb.width > 0 and d.bb.height > 0]

        return valid_face_detections


class MeshDetector:
    def __init__(self):
        self.detector = FaceMesh(
            static_image_mode=False, max_num_faces=MAX_FACE_MASH_N, refine_landmarks=True)

    @staticmethod
    def _extract_mesh_detection(
            raw_landmarks: List, image_width: int, image_height: int) -> MeshDetection:
        xmin = image_width - 1
        ymin = image_height - 1
        xmax = 0
        ymax = 0

        landmarks: Dict[FacialLandmarks, ImagePoint] = dict()
        for ros4hri_idx, mp_idx in ros4hri_to_mp_landmarks_mapping.items():
            landmark_norm = raw_landmarks.landmark[mp_idx]
            x, y = normalized_to_pixel_coordinates(
                landmark_norm.x, landmark_norm.y, image_width, image_height
            )
            landmarks[ros4hri_idx] = ImagePoint(x, y, image_width, image_height)
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        bb = BoundingBox(xmin, ymin, xmax - xmin, ymax - ymin, image_width, image_height)

        return MeshDetection(bb, landmarks)

    def detect(self, img: RGBMat) -> List[MeshDetection]:
        img_height, img_width, _ = img.shape

        mesh_detections: List[MeshDetection] = list()
        mesh_results = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if mesh_results.multi_face_landmarks:
            mesh_detections = [
                self._extract_mesh_detection(raw_landmarks, img_width, img_height)
                for raw_landmarks in mesh_results.multi_face_landmarks]

        return mesh_detections


class NodeFaceDetect(Node):
    def __init__(self):
        super().__init__('hri_face_detect')
        self.image_lock = Lock()
        self.proc_lock = Lock()

        self.declare_parameter(
            'processing_rate', 30, ParameterDescriptor(
                description='Best effort frequency for processing input images'))
        self.declare_parameter(
            'confidence_threshold', 0.75, ParameterDescriptor(
                description='Face detection confidence threshold'))
        self.declare_parameter(
            'image_scale', 0.5, ParameterDescriptor(
                description='Input scale for the image processing pipeline wrt 640x480 pixels'))
        self.declare_parameter(
            'face_mesh', True, ParameterDescriptor(
                description='Enable face mesh output for near faces'))
        self.declare_parameter(
            'filtering_frame', 'camera_color_optical_frame', ParameterDescriptor(
                description='Reference frame for tf filtering'))
        self.declare_parameter(
            'deterministic_ids', False, ParameterDescriptor(
                description='Enable use of non-random increasing face IDs'))
        self.declare_parameter(
            'debug', False, ParameterDescriptor(
                description='Enable debugging output image window'))

        self.get_logger().info('State: Unconfigured.')

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.face_detector = None
        self.mesh_detector = None
        self.get_logger().info('State: Unconfigured.')
        return super().on_cleanup(state)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.processing_rate = self.get_parameter('processing_rate').value
        self.filtering_frame = self.get_parameter('filtering_frame').value
        self.deterministic_ids = self.get_parameter('deterministic_ids').value
        self.debug = self.get_parameter('debug').value

        self.face_detector = FaceDetector(
            self.get_parameter('confidence_threshold').value,
            self.get_parameter('image_scale').value)
        self.mesh_detector = MeshDetector() if self.get_parameter('face_mesh').value else None

        self.get_logger().info('State: Inactive.')
        return super().on_configure(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.reset_faces()
        self.destroy_ros_interfaces()
        self.get_logger().info('State: Inactive.')
        return super().on_deactivate(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.image = None
        self.image_msg_header = None
        self.new_image = False
        self.skipped_images = 0
        self.last_id = 0
        self.knownFaces: Dict[str, Face] = dict()

        self.start_skipping_ts = self.get_clock().now()
        self.detection_start_proc_time = self.get_clock().now()
        self.detection_proc_duration_ms = 0.

        self.tf_broadcaster = TransformBroadcaster(node=self)
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self)
        self.faces_pub = self.create_publisher(IdsList, '/humans/faces/tracked', 1)
        self.image_sub = self.create_subscription(Image, 'image', self.image_callback, 1)
        self.image_info_sub = self.create_subscription(
            CameraInfo, 'camera_info', self.info_callback, 1)
        self.proc_timer = self.create_timer(
            1/self.get_parameter('processing_rate').value, self.process_image)

        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
        self.diag_timer = self.create_timer(1/DIAG_PUB_RATE, self.do_diagnostics)

        self.get_logger().info(
            f'Waiting for images to be published on {self.image_sub.topic_name} .')
        self.get_logger().info('State: Active.')
        return super().on_activate(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        if state.id == State.PRIMARY_STATE_ACTIVE:
            self.reset_faces()
            self.destroy_ros_interfaces()
        self.face_detector = None
        self.mesh_detector = None
        self.get_logger().info('State: Finalized.')
        return super().on_shutdown(state)

    def reset_faces(self):
        now = self.get_clock().now()
        for id in self.knownFaces.keys():
            del self.knownFaces[id]
        self.faces_pub.publish(IdsList(header=Header(stamp=now.to_msg()), ids=[]))
        self.faces_pub.wait_for_all_acked(Duration(seconds=1))

    def destroy_ros_interfaces(self):
        self.destroy_timer(self.diag_timer)
        self.destroy_publisher(self.diag_pub)

        self.destroy_timer(self.proc_timer)
        self.destroy_subscription(self.image_info_sub)
        self.destroy_subscription(self.image_sub)
        self.destroy_publisher(self.faces_pub)

        self.tf_listener.unregister()
        del self.tf_broadcaster

    def do_diagnostics(self):
        now = self.get_clock().now()
        arr = DiagnosticArray(header=Header(stamp=now.to_msg()))
        msg = DiagnosticStatus(
            name='Social perception: Face analysis: Detection', hardware_id='none')

        current_proc_duration = (now - self.detection_start_proc_time).nanoseconds / 1e9
        if ((current_proc_duration > FACE_DETECTION_PROC_TIME_ERROR) and self.image_lock.locked()):
            msg.level = DiagnosticStatus.ERROR
            msg.message = 'Face detection process not responding'
        elif self.detection_proc_duration_ms > FACE_DETECTION_PROC_TIME_WARN:
            msg.level = DiagnosticStatus.WARN
            msg.message = 'Face detection processing is slow'
        else:
            msg.level = DiagnosticStatus.OK

        msg.values = [
            KeyValue(key='Package name', value='hri_face_detect'),
            KeyValue(key='Currently detected faces', value=str(len(self.knownFaces))),
            KeyValue(key='Last detected face ID', value=str(self.last_id)),
            KeyValue(
                key='Detection processing time', value=f'{self.detection_proc_duration_ms:.2f}ms')]

        arr.status = [msg]
        self.diag_pub.publish(arr)

    def info_callback(self, msg: CameraInfo):
        if not hasattr(self, 'msg'):
            self.msg = msg
            self.k = np.zeros((3, 3), np.float32)
            self.k[0][0:3] = self.msg.k[0:3]
            self.k[1][0:3] = self.msg.k[3:6]
            self.k[2][0:3] = self.msg.k[6:9]

    def image_callback(self, msg: Image):
        with self.image_lock:
            self.image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_msg_header = msg.header

            if self.new_image:
                self.skipped_images += 1
                if self.skipped_images > 100:
                    now = self.get_clock().now()
                    skip_time = (now - self.start_skipping_ts).nanoseconds / 1e9
                    self.get_logger().warn(
                        "Face_detect's processing too slow. "
                        f'Skipped 100 new incoming image over the last {skip_time:.1f}sec')
                    self.start_skipping_ts = now
                    self.skipped_images = 0
            self.new_image = True

    def process_image(self):
        if (not self.new_image) or (not self.proc_lock.acquire(blocking=False)):
            return

        with self.image_lock:
            self.detection_start_proc_time = self.get_clock().now()
            image = deepcopy(self.image)
            image_msg_header = deepcopy(self.image_msg_header)
            self.new_image = False

        # copy the list of face ID before iterating over detection, so that we
        # can delete non-existant faces at the end.
        knownIds = list(self.knownFaces.keys())

        face_detections = self.face_detector.detect(image)

        # run fase mesh detection only if it is enabled, at least one face is detected and it is
        # sufficiently big
        image_area_px = image.shape[0] * image.shape[1]
        if (
            self.mesh_detector
            and len(face_detections)
            and any((d.bb.width * d.bb.height) / image_area_px > MIN_FACE_IMAGE_FRAC_FOR_MESH
                    for d in face_detections)
        ):
            mesh_detections = self.mesh_detector.detect(image)

            if len(face_detections) and len(mesh_detections):
                # find best association between faces and meshes using as cost the bounding boxes
                # distances.
                # in the cost matrix the rows are ordered by face detection and columns by meshes.
                cost_matrix = np.array([
                    [distance_rois(fd.bb, md.bb) for md in mesh_detections]
                    for fd in face_detections
                ])
                fd_indices, md_indices = linear_sum_assignment(cost_matrix)
                for fd_idx, md_idx in zip(fd_indices, md_indices):
                    # substitute the landmarks of the matching face detections with the mesh
                    # detection ones
                    face_detections[fd_idx].landmarks = mesh_detections[md_idx].landmarks

        currentIds = []

        for detection in face_detections:

            # have we seen this face before? -> check whether or not bounding boxes overlap
            face = next(
                (face for face in self.knownFaces.values() if bbs_match(face.bb, detection.bb)),
                None)

            if not face:
                face = Face(self, self.tf_buffer, self.filtering_frame, self.deterministic_ids)
                face.initial_detection_time = Time().from_msg(image_msg_header.stamp)
                self.knownFaces[face.id] = face
                self.last_id = face.id

            currentIds.append(face.id)

            # update the face with its current position and landmarks
            face.score = detection.score
            face.bb = detection.bb
            face.landmarks = detection.landmarks

            face.nb_frames_visible += 1
            face.do_publish = True

            # if it is a 3nd frame, we create all the required publishers.
            if face.nb_frames_visible == MIN_FRAMES_FACE_TRACKING:
                face.initialise_publishers()

        # iterate over faces not seen anymore,
        # and unregister corresponding publishers
        for id in knownIds:
            if id not in currentIds:
                face = self.knownFaces[id]
                face.nb_frames_since_last_detection += 1
                face.do_publish = False
                if face.nb_frames_since_last_detection > MAX_FRAMES_FACE_RETENTION:
                    del self.knownFaces[id]

        for id in currentIds:
            face = self.knownFaces[id]
            if face.ready and face.do_publish:
                face.publish(image, image_msg_header)
                if hasattr(self, 'K'):
                    face.compute_6d_pose(self.K, image_msg_header.frame_id, image_msg_header.stamp)
                    self.tf_broadcaster.sendTransform(face.head_transform)
                    self.tf_broadcaster.sendTransform(face.gaze_transform)

        self.detection_proc_duration_ms = (
            self.get_clock().now() - self.detection_start_proc_time).nanoseconds / 1e6

        self.faces_pub.publish(IdsList(
            header=image_msg_header,
            ids=[face.id for face in self.knownFaces.values() if face.ready]))

        if self.debug:
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            self.get_logger().info(
                f'{len(currentIds)} faces detected, {len(self.knownFaces)} faces tracked')
            for id in currentIds:
                face = self.knownFaces[id]
                if not face.ready:
                    continue
                cv2.rectangle(
                    image, (face.bb.xmin, face.bb.ymin),
                    (face.bb.xmin + face.bb.width, face.bb.ymin + face.bb.height),
                    (255, 255, 0), 2)
                cv2.putText(
                    image, f'{face.id}, {face.score}', (face.bb.xmin, face.bb.ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                for landmark in face.landmarks.values():
                    cv2.circle(image, (landmark.x, landmark.y), 2, (0, 255, 255), cv2.FILLED)

            cv2.imshow('Face Detection', image)
            cv2.waitKey(5)

        self.proc_lock.release()


def main(args=None):
    rclpy.init(args=args)
    node = NodeFaceDetect()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == '__main__':
    main()
