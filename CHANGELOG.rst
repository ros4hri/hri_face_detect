^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_face_detect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1.0 (2022-01-18)
------------------
* publish cropped faces under subtopic /cropped
* add _preallocate_topics parameter (instead of hard-coded constant)
* code formatting
* RegionOfInterestStamped -> regionOfInterest to match changes in hri_msgs 0.2.1
* Contributors: Séverin Lemaignan

1.0.1 (2021-11-09)
------------------
* Added the dependency on python-mediapipe
* Publish an Empty msg on /hri_detect_face/ready when ready to start
  This is eg required for automated testing, to ensure the node is fully
  ready before publishing the first frames.
* added minimal node setup
* Added basic readme
* Simple, rough node using Google Mediapipe to perform fast face detection
* Contributors: Séverin Lemaignan
