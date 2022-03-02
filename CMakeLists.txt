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
            message(
                FATAL_ERROR
                "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# Check submodules
foreach(SUBMOD IN LISTS "cuda-samples")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/${SUBMOD}/README.md")
        message(
            FATAL_ERROR 
            "The submodule ${SUBMOD} was not found, check git submodules.")
    endif()
endforeach()


# Check for clt
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/clt")
    message(FATAL_ERROR "clt folder not found, see the README.md")
endif()

include(CTest)

# set(CMAKE_CUDA_FLAGS "-gencode arch=compute_75,code=sm_75")

# create the source files as a lib
add_library(src_files STATIC
    src/dtt8x8.cu
    src/dtt8x8.h
    src/geometry_correction.cu
    src/geometry_correction.h
    src/TileProcessor.cuh
    src/TileProcessor.h
    src/tp_defines.h)
# target_compile_features(src_files PUBLIC cxx_std_11)
set_target_properties(src_files PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# make tests executable
add_executable(${PROJECT_NAME} src/test_tp.cu)
include_directories(external/cuda-samples/Common)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE src_files)
target_compile_definitions(${PROJECT_NAME} PRIVATE CLT_BASE_PATH="${PROJECT_SOURCE_DIR}/clt/")

