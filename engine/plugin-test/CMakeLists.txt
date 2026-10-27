# sotto-plugin-test: loads a VST3 plugin the way Sotto Voce's engine will, outside any DAW.
#
#   cmake -S engine/plugin-test -B build/plugin-test -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/plugin-test --config Release
#
# The Steinberg VST3 SDK (MIT licence since 3.8) is fetched at configure time, without its GUI
# library. Set SOTTO_VST3_SDK_DIR to use a checkout you already have.
# -DSOTTO_BUILD_SDK_EXAMPLES=ON also builds the SDK's example plugins (ADelay), which CI uses to
# test this tool without a commercial plugin.
cmake_minimum_required(VERSION 3.19)
if(APPLE)
    # Apple silicon only: Sotto Voce runs on M-series Macs, and Apple's macOS 27 developer tools no
    # longer include Intel (x86_64) libraries, so a universal build fails to link there.
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")
endif()
project(sotto_plugin_test CXX)
if(APPLE)
    enable_language(OBJCXX)      # the Mac module loader (module_mac.mm) is Objective-C++
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SOTTO_BUILD_SDK_EXAMPLES "Also build the VST3 SDK example plugins (for testing)" OFF)
set(SOTTO_VST3_SDK_DIR "" CACHE PATH "Existing VST3 SDK checkout (optional)")
set(SOTTO_VST3_SDK_TAG "v3.8.0_build_66" CACHE STRING "VST3 SDK release to fetch")

set(SMTG_ENABLE_VSTGUI_SUPPORT OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VST3_HOSTING_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES ${SOTTO_BUILD_SDK_EXAMPLES} CACHE BOOL "" FORCE)
set(SMTG_RUN_VST_VALIDATOR OFF CACHE BOOL "" FORCE)
set(SMTG_CREATE_PLUGIN_LINK OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_AUV2_BUILDS OFF CACHE BOOL "" FORCE)

if(SOTTO_VST3_SDK_DIR)
    set(vst3sdk_SOURCE_DIR "${SOTTO_VST3_SDK_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(vst3sdk
        GIT_REPOSITORY https://github.com/steinbergmedia/vst3sdk.git
        GIT_TAG ${SOTTO_VST3_SDK_TAG}
        GIT_SHALLOW TRUE
        GIT_SUBMODULES base pluginterfaces public.sdk cmake
    )
    FetchContent_GetProperties(vst3sdk)
    if(NOT vst3sdk_POPULATED)
        FetchContent_Populate(vst3sdk)
    endif()
endif()
if(APPLE AND NOT CMAKE_GENERATOR STREQUAL "Xcode")
    # The VST3 SDK asks xcodebuild for the Xcode version and stops if there isn't one. With only
    # Apple's Command Line Tools (no Xcode app) there's no answer, although the compiler is fine.
    execute_process(COMMAND xcodebuild -version RESULT_VARIABLE _xcb OUTPUT_QUIET ERROR_QUIET)
    if(NOT _xcb EQUAL 0)
        message(STATUS "No Xcode app found: building with the Command Line Tools")
        set(ENV{XCODE_VERSION} "16.0")
        set(XCODE_VERSION "16.0")
    endif()
endif()
add_subdirectory(${vst3sdk_SOURCE_DIR} ${CMAKE_BINARY_DIR}/vst3sdk EXCLUDE_FROM_ALL)
set(SDK ${vst3sdk_SOURCE_DIR})

set(host_sources
    host_core.h
    ${SDK}/public.sdk/source/common/memorystream.cpp
    ${SDK}/public.sdk/source/vst/hosting/plugprovider.cpp
)
if(APPLE)
    list(APPEND host_sources ${SDK}/public.sdk/source/vst/hosting/module_mac.mm)
    set_source_files_properties(${SDK}/public.sdk/source/vst/hosting/module_mac.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
elseif(WIN32)
    list(APPEND host_sources ${SDK}/public.sdk/source/vst/hosting/module_win32.cpp)
else()
    list(APPEND host_sources ${SDK}/public.sdk/source/vst/hosting/module_linux.cpp)
endif()

# The command-line test host and the library Sotto Voce's engine loads share the same hosting code.
add_executable(sotto-plugin-test main.cpp wav.h ${host_sources})
add_library(sotto_vst3host SHARED vst3host.cpp ${host_sources})
set_target_properties(sotto_vst3host PROPERTIES CXX_VISIBILITY_PRESET hidden PREFIX "")
foreach(t sotto-plugin-test sotto_vst3host)
    target_link_libraries(${t} PRIVATE sdk_hosting)
    if(APPLE)
        find_library(COCOA Cocoa)
        target_link_libraries(${t} PRIVATE ${COCOA})
    elseif(UNIX)
        target_link_libraries(${t} PRIVATE ${CMAKE_DL_LIBS} pthread)
    endif()
    if(MSVC)
        target_compile_options(${t} PRIVATE /utf-8)
    endif()
endforeach()
if(SOTTO_BUILD_SDK_EXAMPLES)
    add_dependencies(sotto-plugin-test adelay)
endif()
