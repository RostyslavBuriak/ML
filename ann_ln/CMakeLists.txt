cmake_minimum_required(VERSION 3.26)
project(ml)

set(CMAKE_CXX_STANDARD 14)

include_directories(ann)

add_executable(ml
        ann.hpp
        main.cpp
        ann.cpp
)
