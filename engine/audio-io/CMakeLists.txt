# sotto_audio_io: the soundcard side of Sotto Voce's engine (see audio_io.c).
#
#   cmake -S engine/audio-io -B build/audio-io -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/audio-io --config Release
#
# Plain C with no dependencies: PortAudio is loaded at run time from python-sounddevice's copy.
cmake_minimum_required(VERSION 3.16)
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")
endif()
project(sotto_audio_io C)
set(CMAKE_C_STANDARD 11)
add_library(sotto_audio_io SHARED audio_io.c)
set_target_properties(sotto_audio_io PROPERTIES C_VISIBILITY_PRESET hidden PREFIX "")
if(WIN32)
    target_link_libraries(sotto_audio_io PRIVATE winmm)
endif()
if(MSVC)
    target_compile_options(sotto_audio_io PRIVATE /O2 /W3)
else()
    target_compile_options(sotto_audio_io PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
    if(UNIX AND NOT APPLE)
        target_link_libraries(sotto_audio_io PRIVATE ${CMAKE_DL_LIBS} m)
    endif()
endif()
