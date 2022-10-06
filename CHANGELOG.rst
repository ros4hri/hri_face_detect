^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_face_detect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.4.4 (2022-10-06)
------------------
* fix FacialLandmark object initialisation
  When face_mesh=False, the arguments for the FacialLandmarks
  objects initialisation were not correctly disposed, as the
  first element in a FacialLandmarks message is supposed to be a
  Header.
* Contributors: lorenzoferrini

1.4.3 (2022-08-31)
------------------
* more update to hri_msgs-0.8.0
* Contributors: Séverin Lemaignan

1.4.2 (2022-08-31)
------------------
* update to hri_msgs-0.8.0
* Contributors: Séverin Lemaignan

1.4.1 (2022-08-02)
------------------
* ensure face id are strings starting with a letter
* [cosmetic] code formatting
* pep8 code formatting
* add tool to record faces
* Contributors: Séverin Lemaignan

1.4.0 (2022-04-29)
------------------
* large refactor of the code
  In particular:
  - added a Face class to maintain the state of a detected face
  - reworked how detection results are returned, to simplify code
* publish aligned versions of the face under /humans/faces/<id>/aligned
  (aligned faces are rotated such as the eyes are always horizontal)
* warn about faces height and width having to be equal
* store various face publishers as dict to ease future extension
* Delegated face estimation process to function.
* publish empty list of faces upon closing to clean up state
* update launch file to match hri_fullbody arguments names
* [doc] node suitable for production
* Contributors: Séverin Lemaignan, lorenzoferrini

1.3.1 (2022-03-01)
------------------
* Use tf frame from source image
* Contributors: lorenzoferrini

1.3.0 (2022-03-01)
------------------
* changing the frames name syntax from face<id> to face_<id> and gaze<id> to
  gaze_<id> for compliance with ROS4HRI spec
* [minor] adding default value for camera topics in launch
* Documentation update
* Fixed the default number of detectable faces to 10
* Facial Landmark msg implementation
  Fully implemented facial landmark msg publishing for both basic
  face detection and face mesh detection
* Contributors: lorenzoferrini

1.2.0 (2022-02-14)
------------------
* mediapipe Face-mesh based face detection
  It is now possible to decide between two Mediapipe different
  solutions for face detection: face_detection and face_mesh.
  Since the overall performance (taking into account cpu, memory and
  detection results) appears to be better in the latter case,
  face_mesh detection will be the default option.
* add missing deps
* [minor] launch file modified according to new features available
  It is now possible to specify the solution to use
  (face_detection/face_mesh) and the maximum number of faces
  detectable by the face_mesh model as launch file parameters
* max_num_faces as initialization parameter for FaceDetector class
* [WiP] Correcting face orientation and introducing gaze frame
  Face and gaze frame orientation according to ROS4HRI convention.
* Facial landmarks publishing
  Now publishing the facial landmarks according to the ROS4HRI
  definition, on the topic /humans/faces/<body_id>/landmarks.
  Additionally, the face frame is published now as face\_<body_id>
  and the debug code has been removed.
* first rough implementation of PnP head pose estimation
* Contributors: Séverin Lemaignan, lorenzoferrini

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
