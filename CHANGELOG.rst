^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_face_detect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* Added the dependency on python-mediapipe
* Publish an Empty msg on /hri_detect_face/ready when ready to start
  This is eg required for automated testing, to ensure the node is fully
  ready before publishing the first frames.
* added minimal node setup
* Added basic readme
* Simple, rough node using Google Mediapipe to perform fast face detection
* Contributors: Séverin Lemaignan
