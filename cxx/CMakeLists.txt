set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

cmake_minimum_required(VERSION 3.17)
project(test-app VERSION 1.0.3)

add_library(vecxd STATIC vector.h main.cxx)

add_executable(test-app main.cxx)

target_link_libraries(test-app PRIVATE vecxd)
target_compile_features(test-app PRIVATE cxx_std_20)