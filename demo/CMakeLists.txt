cmake_minimum_required(VERSION 3.20)
project(polesitter_demo VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(polesitter_demo main.c)

target_include_directories(polesitter_demo PRIVATE ../src)

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

target_link_libraries(polesitter_demo PRIVATE raylib)

if(NOT MSVC)
    target_link_libraries(polesitter_demo PRIVATE m)
endif()

if(MSVC)
    add_compile_options(polesitter_demo PRIVATE
        /W4 /WX /O2
    )
else()
    add_compile_options(polesitter_demo PRIVATE
        -Wall -Wextra -O3 -ffast-math -march=native
    )
endif()
