# ROS4HRI Face Detection

A [ROS4HRI](https://wiki.ros.org/hri)-compiant ROS node to perform fast face
detection using
[YuNet face detector](https://github.com/ShiqiYu/libfacedetection) and
[Mediapipe Face Mesh](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md).
The former performs well at greater distances (depending on image resolution
and image scaling applied) and extracts 5 keypoints.
The latter works only at close distances and extracts all the ROS4HRI-defined
landmarks.

## ROS API

### Parameters

- `/humans/faces/width` (int, default: 128):
  Image width of the published cropped and aligned detected faces

- `/humans/faces/height` (int, default: 128):
  Image height of the published cropped and aligned detected faces

- `~processing_rate` (int, default: 30):
  Image processing logic execution rate in Hertz.

- `~face_mesh` (bool, default: true):
  It enables the additional Mediapipe Face Mesh detection.

- `~confidence_threshold` (double, default: 0.75):
  Candidate face detections with confidence lower that this threshold are not
  published.

- `~image_scale` (double, default: 0.5):
  The YuNet face detector accepts input image of dynamic size.
  This parameter controls the rescale factor applied to the input image before running the YuNet face detector.
  Lower image scale results in less processing time required and lower detection
  confidences.
  The output data (e.g., RoI) is invariant with this parameter and always refers
  to the original input image size.

- `~filtering_frame` (string, default: "camera_color_optical_frame"):
  The reference frame the estimated face pose should be transformed to before
  performing the filtering operations.
  Due to the proximity between the camera frame and the detected faces, and
  considering that cameras can be mounted on frequently moving robot's
  components (e.g., robot's head), directly filtering a face pose expressed in 
  camera optical frame might reduce the filtering quality.

- `~deterministic_ids` (bool, default: false):
  If true the face ids start from "f00000" and increases by one for each new
  face. If false it is a random five letters sequence.

- `~debug` (bool, default: false):
  If true opens a windows showing the input image with face detections
  overlayed.

### Topics

`hri_face_body_matcher` follows the ROS4HRI conventions
([REP-155](https://www.ros.org/reps/rep-0155.html)).
If the topic message type is not indicated, the ROS4HRI convention is implied.

#### Subscribed

- `/camera/color/image_raw`
  ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html))
- `/camera/color/camera_info`
  ([sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html))

#### Published

- `/humans/faces/<faceID>/roi`
- `/humans/faces/<faceID>/landmarks`
- `/humans/faces/<faceID>/cropped`
- `/humans/faces/<faceID>/aligned`
- `/humans/faces/tracked`
- `/diagnostics`
  ([diagnostic_msgs/DiagnosticArray](https://docs.ros.org/en/noetic/api/diagnostic_msgs/html/msg/DiagnosticArray.html))

## Execution

```bash
roslaunch hri_face_detect detect.launch rgb_camera:=<input camera namespace>
```

## Visualization

[hri_rviz](../hri_rviz) provides Humans, an rviz plugin to easily visualize 
faces bounding boxes and facial landmarks (the latter not published by this
package).
