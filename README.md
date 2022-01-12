ROS4HRI Face Detection
======================

A simple 'test' ROS node to perform fast face detection using [Google
Mediapipe](https://google.github.io/mediapipe/solutions/face_detection).

The node publishes detected faces using the ROS4HRI conventions. It specifically
publishes:

- `humans/faces/<faceID>/roi` as `hri_msgs/RegionOfInterest.msg`
- `humans/faces/tracked` as `hri_msgs/IdsList.msg`

**This node is meant to be used as a baseline for testing purpose. Actual ROS4HRI
pipelines should instead use eg the Mediapipe Holistic model.**

Parameters
----------

- ``_debug`` (``True`` or ``False`` (default)): display a window with the
  detected faces highlighted
- ``_preallocate_topics`` (``True`` or ``False`` (default)): 
  if set to true, face IDs will be generated as a sequence of integers,
  starting at 00001, and ROS publishers for each face are created *before*
  the faces are detected. This makes it possible for client to subscribe
  early to these topics, to avoid missing any message. This is useful to eg
  annotate datasets.

  if false, face IDs will be a random set of 5 characters in [0-9a-f], and
  face subtopics will be created on the fly.

Pre-requistes
-------------

```
> pip install mediapipe
```

