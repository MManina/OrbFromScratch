cmake_minimum_required(VERSION 3.10)

project(OrbFromScratch)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

add_executable(OrbFromScratch main.cpp Orb.cpp)

target_link_libraries(OrbFromScratch ${OpenCV_LIBS})