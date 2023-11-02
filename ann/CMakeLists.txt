cmake_minimum_required(VERSION 3.26)
project(ml)

set(CMAKE_CXX_STANDARD 17)

include_directories(ann)

find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)

add_executable(ml
        ann.cpp
        ann.hpp
        node.cpp
        node.hpp
        layer.cpp
        layer.hpp
        common.hpp
        main.cpp)

target_link_libraries(ml PRIVATE sfml-graphics sfml-window sfml-system)
