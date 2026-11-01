# VR SimReady Physics, native edition: ovrtx + ovxr rendering, ovphysx (PhysX) simulation, all on
# one ovstage. Voice, catalog and HUD text come from the Python sidecar (python -m vrsim.sidecar).
cmake_minimum_required(VERSION 3.24)
project(vrsim_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_deps/fetch" CACHE PATH "Shared FetchContent directory")

# --- ovrtx / ovstage / ovxr: NVIDIA's own package modules, pinned to the ovxr release ---------
FetchContent_Declare(ovxr_repo
    GIT_REPOSITORY https://github.com/NVIDIA-Omniverse/ovxr.git
    GIT_TAG v0.1.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(ovxr_repo)
list(APPEND CMAKE_MODULE_PATH "${ovxr_repo_SOURCE_DIR}/examples/c/cmake")
include(ovrtx)
include(ovstage)
include(ovxr)
ovrtx_fetch()
ovstage_fetch()
ovxr_fetch()

# --- ovphysx: prebuilt C SDK (github.com/NVIDIA-Omniverse/PhysX releases, ovphysx-0.7.0) --------
# It pins the same ovstage build as ovxr 0.1, so both share the package fetched above.
set(OVPHYSX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_deps/ovphysx-sdk/ovphysx" CACHE PATH "Extracted ovphysx SDK")
list(APPEND CMAKE_PREFIX_PATH "${OVPHYSX_SDK_DIR}" "${ovstage_DIR}/../../..")
set(OVPHYSX_FETCH_OVSTAGE OFF CACHE BOOL "" FORCE)
find_package(ovphysx REQUIRED)

# --- header-only helpers --------------------------------------------------------------------
# Headers only: SOURCE_SUBDIR points nowhere so the OpenXR loader is not built (ovxr owns it).
FetchContent_Declare(openxr_sdk
    URL https://github.com/KhronosGroup/OpenXR-SDK/archive/refs/tags/release-1.1.43.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR headers-only)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
    DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(openxr_sdk nlohmann_json)
configure_file("${nlohmann_json_SOURCE_DIR}/json.hpp" "${CMAKE_BINARY_DIR}/include/nlohmann/json.hpp" COPYONLY)

add_executable(vrsim_cpp
    src/main.cpp
    src/stage.cpp
    src/physics.cpp
    src/xr_input.cpp
    src/sidecar.cpp)
target_include_directories(vrsim_cpp PRIVATE
    src
    "${openxr_sdk_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/include")
target_compile_definitions(vrsim_cpp PRIVATE
    VRSIM_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    NOMINMAX WIN32_LEAN_AND_MEAN XR_NO_PROTOTYPES)
target_link_libraries(vrsim_cpp PRIVATE ovxr::ovxr ovrtx::ovrtx ovstage::ovstage ovphysx::ovphysx ws2_32)
if(MSVC)
    target_compile_options(vrsim_cpp PRIVATE /W4 /permissive- /utf-8)
endif()

ovrtx_setup_runtime(vrsim_cpp)
ovstage_setup_runtime(vrsim_cpp)
ovxr_setup_runtime(vrsim_cpp)
if(WIN32)
    ovphysx_copy_runtime_dlls(vrsim_cpp)
    ovphysx_set_ovstage_runtime_env(vrsim_cpp)
endif()
