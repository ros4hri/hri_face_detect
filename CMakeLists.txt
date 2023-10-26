cmake_minimum_required(VERSION 3.16)
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
    python3-tk
    rospy
    sensor_msgs
    std_msgs
    tf
)

### OLDER ATTEMPT TO INCLUDE THE YUNET PROJECT AS EXTERNAL PROJECT WITH ORIGINAL CMAKELISTS.
### IT DID NOT WORK DUE TO ${CATKIN_DEVEL_PREFIX} INCONSISTENCIES IN THE BUILD FARM.

# # build the face detection library as a shared library
# include(ExternalProject)
# externalproject_add(libfacedetection
#   SOURCE_DIR ${CMAKE_SOURCE_DIR}/libfacedetection
#   INSTALL_DIR "${CATKIN_DEVEL_PREFIX}"
#   CMAKE_ARGS
#     -DDEMO=OFF
#     -DBUILD_SHARED_LIBS=ON
#     -DUSE_OPENMP=OFF
#     -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#   BUILD_BYPRODUCTS
#     <INSTALL_DIR>/lib/libfacedetection.so
#     <INSTALL_DIR>/lib/libfacedetection.so.v0.0.3
# )

# # build the python bindings to the face detection library
# set(PYBIND11_FINDPYTHON ON)
# find_package(pybind11 CONFIG)
# pybind11_add_module(yunet_detector src/yunet_detector.cpp)
# add_dependencies(yunet_detector libfacedetection)
# target_include_directories(yunet_detector PRIVATE
#   ${PYBIND11_INCLUDE_DIRS}
#   ${CATKIN_DEVEL_PREFIX}/include
# )
# target_link_libraries(yunet_detector PRIVATE
#   ${CATKIN_DEVEL_PREFIX}/lib/libfacedetection.so
# )

# # install the face detection library
# install(
#   DIRECTORY ${CATKIN_DEVEL_PREFIX}/include/facedetection
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# )
# install(
#   FILES
#     ${CATKIN_DEVEL_PREFIX}/lib/libfacedetection.so
#     ${CATKIN_DEVEL_PREFIX}/lib/libfacedetection.so.v0.0.3
#   DESTINATION DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# )
# install(
#   TARGETS yunet_detector
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # setup and install the ROS pyton package
# catkin_python_setup()
# catkin_install_python(
#   PROGRAMS scripts/detect
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
# install(
#   DIRECTORY launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

### END OF OLD ATTEMPT

### YuNet face detection library
option(ENABLE_AVX512 "use avx512" OFF)
option(ENABLE_AVX2 "use avx2" ON)
option(ENABLE_NEON "whether use neon, if use arm please set it on" OFF)
option(USE_OPENMP "Use OpenMP" OFF)

message("AVX512 = ${ENABLE_AVX512}")
message("AVX2 = ${ENABLE_AVX2}")
message("NEON = ${ENABLE_NEON}")
message("OpenMP = ${USE_OPENMP}")

set(FDT_LIB_NAME   facedetection)
set(FDT_BASE_DIR   ${PROJECT_SOURCE_DIR}/libfacedetection)
set(FDT_SRC_DIR    ${FDT_BASE_DIR}/src)
set(FDT_INC_DIR    ${FDT_BASE_DIR}/src)

# set compiler flags
set(FDT_COMPILE_DEF)
set(FDT_COMPILE_FEAT cxx_std_11)
set(FDT_COMPILE_OPT -Wno-null-dereference)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  #use -O3 to speedup
  set(FDT_COMPILE_OPT ${FDT_COMPILE_OPT} -O3)
ENDIF()

if(ENABLE_AVX512)
	set(FDT_COMPILE_DEF ${FDT_COMPILE_DEF} _ENABLE_AVX512)
	set(FDT_COMPILE_OPT ${FDT_COMPILE_OPT} -mavx512bw)
endif()

if(ENABLE_AVX2)
  set(FDT_COMPILE_DEF ${FDT_COMPILE_DEF} -D_ENABLE_AVX2)
	set(FDT_COMPILE_OPT ${FDT_COMPILE_OPT} -mavx2 -mfma)
endif()

if(ENABLE_NEON)
	message("Using ENON")
  set(FDT_COMPILE_DEF ${FDT_COMPILE_DEF} _ENABLE_AVX2)
endif()

if(USE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(FDT_COMPILE_OPT ${FDT_COMPILE_OPT} ${OpenMP_CXX_FLAGS})
  endif()
endif()

# Create the shared library
file(GLOB_RECURSE FDT_SRC_FILES ${FDT_SRC_DIR}/*.cpp)
list(SORT FDT_SRC_FILES)
add_library(${FDT_LIB_NAME} SHARED ${FDT_SRC_FILES})

include(GenerateExportHeader)
generate_export_header(${FDT_LIB_NAME})
string(TOLOWER ${FDT_LIB_NAME} LOWER_PROJECT_NAME)
set(INSTALLHEADER_FILES
  ${FDT_INC_DIR}/facedetectcnn.h
  ${CMAKE_BINARY_DIR}/${LOWER_PROJECT_NAME}_export.h
)
set(FDT_INC_DIR ${FDT_INC_DIR} ${CMAKE_BINARY_DIR})

target_include_directories(${FDT_LIB_NAME} PUBLIC ${FDT_INC_DIR})
target_compile_definitions(${FDT_LIB_NAME} PUBLIC ${FDT_COMPILE_DEF})
target_compile_features(${FDT_LIB_NAME} PUBLIC ${FDT_COMPILE_FEAT})
target_compile_options(${FDT_LIB_NAME} PUBLIC ${FDT_COMPILE_OPT})
set_target_properties(${FDT_LIB_NAME} PROPERTIES
  PUBLIC_HEADER "${INSTALLHEADER_FILES}" # Install head files
  CXX_EXTENSIONS OFF
)

# Install target
install(
  TARGETS ${FDT_LIB_NAME}
  LIBRARY DESTINATION "${CATKIN_PACKAGE_LIB_DESTINATION}"
  PUBLIC_HEADER DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
### End of YuNet face detection library

# build the python bindings to the face detection library
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG)
pybind11_add_module(yunet_detector src/yunet_detector.cpp)
target_include_directories(yunet_detector PRIVATE ${PYBIND11_INCLUDE_DIRS} ${FDT_INC_DIR})
target_link_libraries(yunet_detector PRIVATE ${FDT_LIB_NAME})

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
