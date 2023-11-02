cmake_minimum_required(VERSION 3.26)
project(ml)

set(CMAKE_CXX_STANDARD 17)

add_executable(ml
        ann.hpp
        main.cpp
        ann.cpp
)
