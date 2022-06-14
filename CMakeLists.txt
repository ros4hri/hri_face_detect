cmake_minimum_required(VERSION 3.0.2)
project(hri_face_detect)

find_package(catkin REQUIRED)

catkin_package(
    CATKIN_DEPENDS 
    sensor_msgs
    std_msgs
    tf
    tf_conversions
    hri_msgs)

catkin_python_setup()

catkin_install_python(PROGRAMS nodes/hri_face_detect/detect nodes/hri_face_detect/record
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

