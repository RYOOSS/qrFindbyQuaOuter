cmake_minimum_required(VERSION 3.16)
project(qrFindbyQuaOuter)

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/include/eigen3)
include_directories(/usr/local/include/zxing)
include_directories(/usr/local/include)
include_directories(.)
link_directories(/usr/local/lib)

add_executable(qrFindbyQuaOuter main.cpp imageUtils.cpp)
target_link_libraries(qrFindbyQuaOuter ${OpenCV_LIBS} zbar zxing)