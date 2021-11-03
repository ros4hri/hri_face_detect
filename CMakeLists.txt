cmake_minimum_required(VERSION 3.0.2)
project(hri_face_detect)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS nodes/hri_face_detect/detect
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

