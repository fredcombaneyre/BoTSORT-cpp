CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(botsort VERSION 1.0 LANGUAGES CXX)

include(FetchContent)
include(CheckCXXCompilerFlag)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Build Type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

# Collect all source files
file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Create library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Set include directories
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Find and link OpenCV
# find_package(OpenCV REQUIRED)
set( OpenCV_DIR 
    "/ESPDEV/BoTSORT-cpp/opencv" )
set( OpenCV_INCLUDE_DIRS 
    "/ESPDEV/BoTSORT-cpp/opencv/usr/include/opencv4" )
set( OpenCV_LIB_DIR 
    "/ESPDEV/BoTSORT-cpp/opencv/lib/aarch64-linux-gnu" )
set( OpenCV_LIBS 
    "opencv_photo"
	"opencv_videoio"
	"opencv_video"
	"opencv_calib3d"
	"opencv_features2d"
	"opencv_imgproc"
	"opencv_core"
	"opencv_imgcodecs"
	"opencv_dnn"
	"opencv_flann"
    "opencv_videostab" )

    #libopencv_photo.so.4.5d => /lib/aarch64-linux-gnu/libopencv_photo.so.4.5d 
	#libopencv_videoio.so.4.5d => /lib/aarch64-linux-gnu/libopencv_videoio.so.4.5d 
	#libopencv_video.so.4.5d => /lib/aarch64-linux-gnu/libopencv_video.so.4.5d 
	#libopencv_calib3d.so.4.5d => /lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5d 
	#libopencv_features2d.so.4.5d => /lib/aarch64-linux-gnu/libopencv_features2d.so.4.5d 
	#libopencv_imgproc.so.4.5d => /lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5d 
	#libopencv_core.so.4.5d => /lib/aarch64-linux-gnu/libopencv_core.so.4.5d 
	#libopencv_imgcodecs.so.4.5d => /lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5d 
	#libopencv_dnn.so.4.5d => /lib/aarch64-linux-gnu/libopencv_dnn.so.4.5d 
	#libopencv_flann.so.4.5d => /lib/aarch64-linux-gnu/libopencv_flann.so.4.5d 
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
# target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)
if(NOT EIGEN3_FOUND)
  set(BUILD_TESTING OFF CACHE INTERNAL "")
  FetchContent_Declare(eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         ${EIGEN_VERSION}
    GIT_SHALLOW     ON)
  FetchContent_MakeAvailable(eigen)
  unset(BUILD_TESTING CACHE)
endif()

# Find and link Eigen3
# find_package(Eigen3 REQUIRED)
# target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Find and link Boost
# find_package(Boost REQUIRED COMPONENTS filesystem)
# target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

# Find and link CUDA
find_package(CUDA QUIET)

if(NOT ${CUDA_FOUND})
    message(WARNING "CUDA not found, ReID won't be built")
else()
    message(STATUS "CUDA version ${CUDA_VERSION_STRING} found")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

    target_sources(${PROJECT_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/TRT_InferenceEngine/TensorRT_InferenceEngine.cpp
        ${PROJECT_SOURCE_DIR}/src/ReID.cpp)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} nvonnxparser nvinfer)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
else()
    add_definitions(-DPROFILE=1)
    add_compile_options(-O3)
endif()