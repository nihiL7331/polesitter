cmake_minimum_required(VERSION 3.20)
project(polesitter_demo VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(polesitter_unary unary.c)
add_executable(polesitter_binary binary.c)
add_executable(polesitter_unary_2d unary_2d.c)

target_include_directories(polesitter_unary PRIVATE ../src)
target_include_directories(polesitter_binary PRIVATE ../src)
target_include_directories(polesitter_unary_2d PRIVATE ../src)

find_package(raylib QUIET)

if(NOT raylib_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        master
    )
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(raylib)
endif()

target_link_libraries(polesitter_unary PRIVATE raylib)
target_link_libraries(polesitter_binary PRIVATE raylib)
target_link_libraries(polesitter_unary_2d PRIVATE raylib)

if(NOT MSVC)
    target_link_libraries(polesitter_unary PRIVATE m)
    target_link_libraries(polesitter_binary PRIVATE m)
    target_link_libraries(polesitter_unary_2d PRIVATE m)
endif()

if(MSVC)
    set(POLESITTER_FLAGS /W4 /WX /O2)
else()
    set(POLESITTER_FLAGS -Wall -Wextra -O3 -ffast-math -march=native)
endif()

target_compile_options(polesitter_unary PRIVATE ${POLESITTER_FLAGS})
target_compile_options(polesitter_binary PRIVATE ${POLESITTER_FLAGS})
target_compile_options(polesitter_unary_2d PRIVATE ${POLESITTER_FLAGS})
