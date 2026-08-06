#minimum CMake version
cmake_minimum_required(VERSION 3.10)

#project name
project(seabed_classifier LANGUAGES CXX)

#dependencies
find_package(OpenCV REQUIRED)

#where to find header files
include_directories(include ${OpenCV_INCLUDE_DIRS})

#compilation target
add_executable(seabed_classifier 
    src/main.cpp 
    src/preprocessing.cpp 
    src/classifier.cpp
)

#connect executable with OpenCV's compiled libraries
target_link_libraries(seabed_classifier ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 11)          
set(CMAKE_CXX_STANDARD_REQUIRED ON)

