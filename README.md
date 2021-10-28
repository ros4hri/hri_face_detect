ROS4HRI Face Detection
======================

A simple 'test' ROS node to perform fast face detection using [Google
Mediapipe](https://google.github.io/mediapipe/solutions/face_detection).

The node publishes detected faces using the ROS4HRI conventions. It specifically
publishes:

- `humans/faces/<faceID>/roi` as `hri_msgs/RegionOfInterestStamped.msg`
- `humans/faces/tracked` as `hri_msgs/IdsList.msg`

**This node is meant to be used as a baseline for testing purpose. Actual ROS4HRI
pipelines should instead use eg the Mediapipe Holistic model.**

Pre-requistes
-------------

```
> pip install mediapipe
```

