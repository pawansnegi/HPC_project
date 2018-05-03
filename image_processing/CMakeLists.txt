cmake_minimum_required(VERSION 2.8)
## build option flags

#
# Instruct CMake to run moc/uic/rcc automatically when needed.

project(img_procsr)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
#
include_directories(
    ${Qt5Widgets_INCLUDE_DIRS}
)

## Setup the Source Files
add_executable( img_procsr
    src/main.cpp 
    src/face_detect/detect.cpp
    src/face_recog/recog.cpp
    src/face_track/track.cpp
)
target_link_libraries(img_procsr ${OpenCV_LIBS})
