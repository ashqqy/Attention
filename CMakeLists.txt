cmake_minimum_required(VERSION 3.14)

project(attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tensor_lib INTERFACE)

target_include_directories(tensor_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_subdirectory(tests)
