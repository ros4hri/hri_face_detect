ROS4HRI Face Detection
======================

A simple 'test' ROS node to perform fast face detection using [Google
Mediapipe](https://google.github.io/mediapipe/solutions/face_detection).

The node publishes detected faces using the ROS4HRI conventions. It specifically publishes:

- `humans/faces/<faceID>/roi` as `sensor_msgs/RegionOfInterest.msg`
- `humans/faces/<faceID>/cropped` as `sensor_msgs/Image.msg`
- `humans/faces/<faceID>/landmarks` as `hri_msgs/FacialLandmarks.msg`
- `humans/faces/tracked` as `hri_msgs/IdsList.msg`

**This node is meant to be used as a baseline for testing purpose. Actual ROS4HRI
pipelines should instead use eg the Mediapipe Holistic model.**

Parameters
----------

- ``camera_info`` (string, default: ``/camera/color/camera_info``): the name of the camera parameters topic.
- ``image_topic`` (string, default: ``/camera/color/image_raw``): the name of the topic where the images gets published.
- ``face_mesh`` (``True``(default) or ``False``): if true, the node applies 
Mediapipe face mesh estimation on the incoming images. Otherwise, it applies
Mediapipe face estimation. 
- ``max_num_faces`` (integer, default: ``10``): the maximum number of faces that the face mesh estimator can detect.

Pre-requistes
-------------

```
> pip install mediapipe
```
Documentation
-------------

- [ROS4HRI to Mediapipe facial landmarks indexing](doc/mediapipe_face_mesh_2_ros4hri.md)
