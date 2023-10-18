cmake_minimum_required(VERSION 3.8)
project(hri_face_detect)

# configure the ROS package
find_package(catkin REQUIRED
  COMPONENTS pybind11_catkin
)
catkin_package(
  CATKIN_DEPENDS
    diagnostic_msgs
    hri_msgs
    pybind11_catkin
    pybind11-dev
    python3-mediapipe
    python3-numpy
    python3-opencv
    python3-pil
    python3-scipy
    rospy
    sensor_msgs
    std_msgs
    tf
)

# build the face detection library as a shared library
include(ExternalProject)
externalproject_add(libfacedetection
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/libfacedetection
  CMAKE_ARGS
    -DDEMO=OFF
    -DBUILD_SHARED_LIBS=ON
    -DUSE_OPENMP=OFF
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
)

# build the python bindings to the face detection library
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG)
pybind11_add_module(yunet_detector src/yunet_detector.cpp)
add_dependencies(yunet_detector libfacedetection)
target_include_directories(yunet_detector PRIVATE
  ${PYBIND11_INCLUDE_DIRS}
  ${CMAKE_INSTALL_PREFIX}/include
)
target_link_libraries(yunet_detector PRIVATE
  ${CMAKE_INSTALL_PREFIX}/lib/libfacedetection.so
)
install(
  TARGETS yunet_detector
  LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# setup and install the ROS pyton package
catkin_python_setup()
catkin_install_python(
  PROGRAMS scripts/detect
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

