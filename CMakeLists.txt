cmake_minimum_required(VERSION 3.16)

# ============================================================
# Find HIP compiler from current environment
# ============================================================

find_program(HIPCC_EXECUTABLE hipcc)

if(NOT HIPCC_EXECUTABLE)
    message(FATAL_ERROR
        "hipcc not found. Please source DTK env.sh first."
    )
endif()

set(CMAKE_CXX_COMPILER
    "${HIPCC_EXECUTABLE}"
    CACHE FILEPATH "HIP C++ compiler"
)
message(STATUS "HIP compiler: ${CMAKE_CXX_COMPILER}")

# ============================================================
# Project
# ============================================================

project(rccl_api_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# GoogleTest
# ============================================================

set(GTEST_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/tools/googletest"
)

if(NOT EXISTS "${GTEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "GoogleTest not found: ${GTEST_SOURCE_DIR}"
    )
endif()

add_subdirectory(
    "${GTEST_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/googletest"
)

# ============================================================
# RCCL
# ============================================================

if(NOT DEFINED ENV{ROCM_PATH})
    message(FATAL_ERROR
        "ROCM_PATH is not set. Please source the DTK env.sh before configuring."
    )
endif()

set(ROCM_PATH "$ENV{ROCM_PATH}")

# RCCL header provided by DTK/ROCm
set(RCCL_INCLUDE_DIR "${ROCM_PATH}/include/rccl")

if(NOT EXISTS "${RCCL_INCLUDE_DIR}/rccl.h")
    message(FATAL_ERROR
        "RCCL header rccl.h not found: ${RCCL_INCLUDE_DIR}/rccl.h"
    )
endif()

# RCCL library provided by DTK/ROCm
find_library(RCCL_LIBRARY
    NAMES rccl
    PATHS
        "${ROCM_PATH}/lib"
        "${ROCM_PATH}/lib64"
        "${ROCM_PATH}/rccl/lib"
    NO_DEFAULT_PATH
)

if(NOT RCCL_LIBRARY)
    message(FATAL_ERROR
        "RCCL library librccl.so not found under ROCM_PATH=${ROCM_PATH}"
    )
endif()

get_filename_component(RCCL_LIBRARY_DIR
    "${RCCL_LIBRARY}"
    DIRECTORY
)

message(STATUS "ROCM_PATH:        ${ROCM_PATH}")
message(STATUS "RCCL include dir: ${RCCL_INCLUDE_DIR}")
message(STATUS "RCCL library:     ${RCCL_LIBRARY}")
message(STATUS "RCCL library dir: ${RCCL_LIBRARY_DIR}")

# ============================================================
# Test sources
# ============================================================

set(TEST_SOURCES

    # Common
    src/common/rccl_test_common.cpp
    src/common/rccl_test_utils.cpp
    src/common/rccl_comm_utils.cpp
    src/comm/test_comm_api_coverage.cpp
    src/comm/test_comm_shrink_api_coverage.cpp
    src/comm/test_comm_window_api_coverage.cpp

    # Misc
    src/misc/test_version.cpp

    # Collective
    src/collective/test_allreduce.cpp
    src/collective/test_collectives.cpp
    src/collective/test_group.cpp
    src/collective/test_p2p.cpp
    src/collective/test_collective_api_coverage.cpp

    # Communication
    # src/comm/test_comm_init.cpp
    # src/comm/test_comm_lifecycle.cpp
    # src/comm/test_comm_info.cpp
    src/comm/test_comm_split.cpp
    src/comm/test_comm_shrink.cpp
    src/comm/test_comm_window.cpp

    # Memory
    src/memory/test_mem.cpp
    src/memory/test_register.cpp
    src/memory/test_mem_api_coverage.cpp

    # Datatype
    src/datatype/test_redop.cpp
    src/datatype/test_redop_api_coverage.cpp

    # Error
    src/error/test_error.cpp

    # MSCCL
    src/msccl/test_msccl.cpp

    # Miscellaneous
    src/misc/test_allreduce_bias.cpp
    src/misc/test_unique_id.cpp
)

# ============================================================
# Test executable
# ============================================================

add_executable(rccl_api_test
    ${TEST_SOURCES}
)

target_include_directories(rccl_api_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${RCCL_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/src/common
)

# ============================================================
# Link libraries
# ============================================================

target_link_libraries(rccl_api_test
    PRIVATE
        ${RCCL_LIBRARY}
        GTest::gtest
        GTest::gtest_main
)

# ============================================================
# Compiler warnings
# ============================================================

target_compile_options(rccl_api_test
    PRIVATE
        -Wall
        -Wextra
)

# ============================================================
# Runtime library search path
# ============================================================

set_target_properties(rccl_api_test PROPERTIES

    BUILD_RPATH
        "${RCCL_LIBRARY_DIR};$ENV{HIP_PATH}/lib;$ENV{ROCM_PATH}/lib"

    RUNTIME_OUTPUT_DIRECTORY
        "${PROJECT_BINARY_DIR}/bin"
)

# ============================================================
# GoogleTest integration
# ============================================================

include(GoogleTest)

gtest_discover_tests(rccl_api_test)