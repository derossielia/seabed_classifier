cmake_minimum_required(VERSION 3.10)
project(seabed_classifier)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")

add_executable(seabed_classifier ${SOURCES})
target_link_libraries(seabed_classifier ${OpenCV_LIBS})

