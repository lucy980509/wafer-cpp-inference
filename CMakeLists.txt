cmake_minimum_required(VERSION 3.20)

project(WaferCppInference)


# =========================================================
# Build Configuration
#
# This project requires:
# - OpenCV
# - ONNX Runtime
#
# Update the paths below according to your local environment.
# =========================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# =========================================================
# OpenCV
# =========================================================

set(OpenCV_DIR "C:/opencv/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)



# =========================================================
# ONNX Runtime
# =========================================================

set(ONNXRUNTIME_DIR "C:/onnxruntime/onnxruntime-win-x64-1.28.1")

include_directories(
    ${ONNXRUNTIME_DIR}/include
)



# =========================================================
# Executable
# =========================================================

add_executable(
    wafer_inference
    src/main.cpp
)



# =========================================================
# Libraries
# =========================================================

target_link_libraries(
    wafer_inference
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)