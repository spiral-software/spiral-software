##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

# specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required(VERSION 3.14)

set(CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

project(SPIRAL
        VERSION 8.3.0
        DESCRIPTION "SPIRAL Project"
        LANGUAGES C CXX)

set(PROJECT_VERSION_TAG  )

##  Prevent building directly into the source tree
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" SPIRAL_COMPILE_INPLACE)
if (SPIRAL_COMPILE_INPLACE)
    if (WIN32)
        message (FATAL_ERROR "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
                 rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
                 mkdir <build>
                 cd <build>
                 cmake ..
                 then build the solution with VS")
    else()
        message (FATAL_ERROR "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
                 rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
                 mkdir <build>
                 cd <build>
                 cmake ..
                 make | make install")
    endif()
endif()
  
set(SPIRAL_CONFIG_DIR ${SPIRAL_SOURCE_DIR}/config)
set(SPIRAL_GEN_INC ${SPIRAL_BINARY_DIR}/include/spiral)

##  set(CMAKE_C_STANDARD 99)

include_directories(${SPIRAL_GEN_INC})

# Check for the git commit hash, if using a git repo
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    find_package(Git)
    if (Git_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                        OUTPUT_VARIABLE SPIRAL_GIT_HASH ERROR_QUIET)
        execute_process(COMMAND ${GIT_EXECUTABLE} branch --show-current
                        OUTPUT_VARIABLE SPIRAL_GIT_BRANCH ERROR_QUIET)
        execute_process(COMMAND ${GIT_EXECUTABLE} config --get remote.origin.url
                        OUTPUT_VARIABLE SPIRAL_GIT_REMOTE ERROR_QUIET)
    endif()
endif()
if (NOT SPIRAL_GIT_HASH)
   set(SPIRAL_GIT_HASH "<Unknown>")
endif()
if (NOT SPIRAL_GIT_BRANCH)
   set(SPIRAL_GIT_BRANCH "<Unknown>")
endif()
if (NOT SPIRAL_GIT_REMOTE)
   set(SPIRAL_GIT_REMOTE "<Unknown>")
endif()
string(STRIP ${SPIRAL_GIT_BRANCH} SPIRAL_GIT_BRANCH)
string(STRIP ${SPIRAL_GIT_REMOTE} SPIRAL_GIT_REMOTE)

if ((NOT DEFINED CMAKE_BUILD_TYPE) OR (NOT CMAKE_BUILD_TYPE))
    set(CMAKE_BUILD_TYPE Release)
endif()

string(TIMESTAMP SPIRAL_BUILD_DATE_UTC "%m/%d/%Y" UTC)
string(TIMESTAMP SPIRAL_BUILD_TIME_UTC "%H:%M:%S" UTC)

configure_file (
    ${SPIRAL_CONFIG_DIR}/spiral_build_info.h.in
    ${SPIRAL_GEN_INC}/spiral_build_info.h
)

# Determine the system's endian type

include (TestBigEndian)
TEST_BIG_ENDIAN(ENDIAN_TYPE)
message(STATUS "My endian type = ${ENDIAN_TYPE}")
if (ENDIAN_TYPE)
    set(ENDIANNESS "BIG_ENDIAN")
else()
    set(ENDIANNESS "LITTLE_ENDIAN")
endif()
message(STATUS "Set ENDIANNESS = ${ENDIANNESS}")
configure_file (
    ${SPIRAL_SOURCE_DIR}/gap/src/machine_endian.h.in
    ${SPIRAL_SOURCE_DIR}/gap/src/machine_endian.h
)

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/ScriptConfig.cmake")
if (WIN32)
    configure_file (
        ${SPIRAL_CONFIG_DIR}/spiral.bat.in
        ${SPIRAL_BINARY_DIR}/gap/spiral.bat
    )
    configure_file (
        ${SPIRAL_CONFIG_DIR}/spiral_debug.bat.in
        ${SPIRAL_BINARY_DIR}/gap/spiral_debug.bat
    )
    configure_file (
        ${SPIRAL_CONFIG_DIR}/_spiral_win.g.in
        ${SPIRAL_BINARY_DIR}/gap/_spiral_win.g
    )
else ()
    configure_file (
        ${SPIRAL_CONFIG_DIR}/spiral.in
        ${SPIRAL_BINARY_DIR}/gap/spiral
    )
    configure_file (
        ${SPIRAL_CONFIG_DIR}/spirald.in
        ${SPIRAL_BINARY_DIR}/gap/spirald
    )
    configure_file (
        ${SPIRAL_CONFIG_DIR}/_spiral.g.in
        ${SPIRAL_BINARY_DIR}/gap/_spiral.g
    )
endif ()

##  Rule to install the script files when install target is built

if (WIN32)
    set (SPIRAL_STARTUP_FILES
            ${SPIRAL_BINARY_DIR}/gap/spiral.bat
            ${SPIRAL_BINARY_DIR}/gap/spiral_debug.bat
            ${SPIRAL_BINARY_DIR}/gap/_spiral_win.g
        )
else ()
    set (SPIRAL_STARTUP_FILES
            ${SPIRAL_BINARY_DIR}/gap/spiral
            ${SPIRAL_BINARY_DIR}/gap/spirald
            ${SPIRAL_BINARY_DIR}/gap/_spiral.g
        )
endif ()

install (FILES ${SPIRAL_STARTUP_FILES}
         DESTINATION ${SPIRAL_SOURCE_DIR}
         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                     GROUP_EXECUTE GROUP_READ
                     WORLD_EXECUTE WORLD_READ
        )

add_subdirectory(gap)

add_subdirectory ( support )

##  Find any subdirectories in namespaces/packages -- these will be the names
##  of spiral packages.  Add subdirectory entries for any names found *if* they
##  have a CMakeLists.txt file present.

set ( _pkg_folder "${CMAKE_SOURCE_DIR}/namespaces/packages" )
file ( GLOB _pkg_names RELATIVE ${_pkg_folder} CONFIGURE_DEPENDS ${_pkg_folder}/* )
message(STATUS "Found in namespaces/packages: = ${_pkg_names}" )

foreach ( dir ${_pkg_names} )
    if ( IS_DIRECTORY ${_pkg_folder}/${dir} AND EXISTS "${_pkg_folder}/${dir}/CMakeLists.txt" )
        # subdirectory exists and contains a CMakeLists.txt file -- add subir
        message ( STATUS "Added sub-directory ${dir} for spiral-package-${dir}" )
        add_subdirectory ( "namespaces/packages/${dir}" )
    endif ()
endforeach ()

##  Find python3 -- may be used by spiral-packages to marshall/run examples

find_package (Python3 COMPONENTS Interpreter)
if (${Python3_FOUND})
    ##  It exists, executable is ${Python3_EXECUTABLE}
    message ( STATUS "Found Python3: Version = ${Python3_VERSION}, Executable = ${Python3_EXECUTABLE}")
else ()
    message ( SEND_ERROR "Python3 NOT FOUND: Python is required to run tests")
endif ()


add_subdirectory(tests)                ## holds test scripts

##  We will only build the documentation if the following conditions are satisfied:
##    SPIRAL_MAKE_DOCS is defined when cmake is invoked (e.g., cmake -DSPIRAL_MAKE_DOCS=True ..)
##    Sphinx is installed (and possibly Doxygen down the road)

if ( "X${SPIRAL_MAKE_DOCS}X" STREQUAL "XX" OR NOT SPIRAL_MAKE_DOCS )
    message ( STATUS "SPIRAL_MAKE_DOCS is not defined, do NOT build documentation" )
else ()
    ##  Add the CMakeIncludes folder so the FindSphinx module is found
    set ( CMAKE_MODULE_PATH "${SPIRAL_SOURCE_DIR}/config/CMakeIncludes" ${CMAKE_MODULE_PATH} )

    ##  initially no Doxygen extraction needed, skip for now
    ##  find_package ( Doxygen )	##  Don't make required -- but check if found
    find_package ( Sphinx )		##  REQUIRED
    if ( ${Sphinx_FOUND} )
	message ( STATUS "Found Sphinx, add subdirectory 'docs' to build documentation" )
	add_subdirectory (docs)
    else ()
	message ( STATUS "Sphinx not found -- skipping 'docs', do NOT build documentation" )
    endif ()
endif ()

enable_testing()

