cmake_minimum_required(VERSION 3.20)
project(polesitter VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(polesitter INTERFACE)

target_include_directories(polesitter INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  enable_testing()

  add_executable(polesitter_tests tests/main.c)

  target_link_libraries(polesitter_tests PRIVATE polesitter m)

  target_compile_options(polesitter_tests PRIVATE
    -Wall -Wextra -pedantic -O3 -march=native
  )

  add_test(NAME core_tests COMMAND polesitter_tests)
endif()
