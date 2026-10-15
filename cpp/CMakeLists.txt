cmake_minimum_required(VERSION 3.16)
project(yolo_ort_arm64 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

set(ONNXRUNTIME_ROOT
    "$ENV{HOME}/ai_camera/onnxruntime-linux-aarch64-1.18.0"
    CACHE PATH "ONNX Runtime root"
)

add_executable(yolo_ort
    preview.cpp
    detector.cpp
    object_tracker.cpp
)
target_include_directories(yolo_ort PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${LIBCAMERA_INCLUDE_DIRS}
)

target_link_directories(yolo_ort PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
    ${LIBCAMERA_LIBRARY_DIRS}
)

target_link_libraries(yolo_ort PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    ${LIBCAMERA_LIBRARIES}
)

target_compile_options(yolo_ort PRIVATE
    ${LIBCAMERA_CFLAGS_OTHER}
)

set_target_properties(yolo_ort PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
