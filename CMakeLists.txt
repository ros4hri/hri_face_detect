cmake_minimum_required(VERSION 3.8)
project(hri_face_detect)

# Global flags
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)  # Find python before pybind11
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)

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
### END OF OLD ATTEMPT

#########################################################################################
#########################################################################################

### YuNet face detection library

# this part of the CMakeLists is based on the original YuNet library:
# https://github.com/ShiqiYu/libfacedetection/blob/master/CMakeLists.txt

option(ENABLE_AVX512 "use avx512" OFF)
option(ENABLE_AVX2 "use avx2" ON)
option(ENABLE_NEON "whether use neon, if use arm please set it on" OFF)
option(USE_OPENMP "Use OpenMP" OFF)

message(STATUS "AVX512 = ${ENABLE_AVX512}")
message(STATUS "AVX2 = ${ENABLE_AVX2}")
message(STATUS "NEON = ${ENABLE_NEON}")
message(STATUS "OpenMP = ${USE_OPENMP}")

set(FDT_LIB_NAME   facedetection)
set(FDT_BASE_DIR   ${PROJECT_SOURCE_DIR})
set(FDT_SRC_DIR    ${FDT_BASE_DIR}/src/libfacedetectcnn)
set(FDT_INC_DIR    ${FDT_BASE_DIR}/include/libfacedetectcnn)

# set compiler flags
set(FDT_COMPILE_DEF)
set(FDT_COMPILE_FEAT cxx_std_11)
set(FDT_COMPILE_OPT -Wno-null-dereference)

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
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
)

### End of YuNet face detection library

#########################################################################################
#########################################################################################
ament_python_install_package(${PROJECT_NAME}
  SCRIPTS_DESTINATION lib/${PROJECT_NAME}
)

# build the python bindings to the face detection library
pybind11_add_module(yunet_detector src/hri_face_detect/yunet_detector.cpp)
target_include_directories(yunet_detector PRIVATE ${PYBIND11_INCLUDE_DIRS} include)
target_link_libraries(yunet_detector PRIVATE ${FDT_LIB_NAME})
install(TARGETS yunet_detector
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_lint_common REQUIRED)

  set(AMENT_LINT_AUTO_FILE_EXCLUDE
    include/libfacedetectcnn/facedetectcnn.h
    src/libfacedetectcnn/facedetectcnn-data.cpp
    src/libfacedetectcnn/facedetectcnn-model.cpp
    src/libfacedetectcnn/facedetectcnn.cpp)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
