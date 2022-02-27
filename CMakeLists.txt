cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(tile-processor-gpu LANGUAGES CUDA CXX)

# get submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()


if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/cuda-samples/README.md")
    message(FATAL_ERROR "The submodule cuda-samples was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# create the executable
add_executable(${PROJECT_NAME}
    src/test_tp.cu
    src/dtt8x8.cu
    src/dtt8x8.h
    src/geometry_correction.cu
    src/geometry_correction.h
    src/test_tp.cu
    src/TileProcessor.cuh
    src/TileProcessor.h
    src/tp_defines.h)
include_directories(external/cuda-samples/Common)
set_target_properties(${PROJECT_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


