##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
## 
## See LICENSE file for full information
##

cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

##  Setup preprocessor definitions

set(SYSSRCS_COMPILE_DEFNS
    SYS_HAS_TIME_PROTO SYS_HAS_IOCTL_PROTO SYS_HAS_READ_PROTO
    SYS_HAS_MISC_PROTO 
    )

message(STATUS "CMAKE_HOST_APPLE = ${CMAKE_HOST_APPLE}")
if (WIN32)
    list (APPEND SYSSRCS_COMPILE_DEFNS SYS_HAS_MALLOC_PROTO SYS_HAS_STRING_PROTO)
else()
    ##  Not Windows ... linux or Darwin/Mac
    if (${CMAKE_HOST_APPLE})
	list (APPEND SYSSRCS_COMPILE_DEFNS SYS_IS_BSD)
    else ()
	list (APPEND SYSSRCS_COMPILE_DEFNS SYS_IS_USG SYS_HAS_SIG_T=void)
    endif ()
endif()

##  Is this a 64 bit processor?
##  cmake_host_system_information(RESULT PROCIS64BIT QUERY IS_64BIT)
##  message (STATUS "Processor is 64 bit = ${PROCIS64BIT}")
##  Could be 64 bit system but building for 32 bit; test sizeof(void_p) instead

if (${CMAKE_SIZEOF_VOID_P} EQUAL "8")
    set(SPIRAL_COMPILE_DEFNS SYS_IS_64_BIT WIN64)
else()
    set(SPIRAL_COMPILE_DEFNS )
endif()

if (WIN32)
    list(APPEND SPIRAL_COMPILE_DEFNS
		WIN32 YY_NO_UNISTD_H _USE_MATH_DEFINES _CONSOLE inline=__inline)
endif()

string(COMPARE EQUAL "${CMAKE_BUILD_TYPE}" "Debug" SPIRAL_COMPILE_DEBUG)
if (SPIRAL_COMPILE_DEBUG)
    list(APPEND SPIRAL_COMPILE_DEFNS _DEBUG)
else()
    list(APPEND SPIRAL_COMPILE_DEFNS NDEBUG)
endif()

set(SPIRAL_COMPILE_FLAGS -w)
  
set(PROJECT gap)

project(${PROJECT})

set(SOURCES
    system.c
    gap.c
    memmgr.c
    scanner.c
    idents.c
    read.c
    eval.c
    spiral_delay_ev.c
    spiral_fft.c
    spiral_spl_prog.c
    spiral.c
    globals.c
    fft.c
    sys_conf_wrap.c
    args.c
    comments.c
    type_wrap.c
    hooks.c
    namespaces.c
    tables.c
    spiral_bag_list.c
    double.c
    complex.c
    lex.yy.c
    interface.c
    buf_list.c
    iface.c
    iface_table.c
    iface_original.c
    iface_console.c
    iface_batch.c
    debug.c
    md5.c
    hash.c
    namespaces_bin.c
    integer.c
    integer4.c
    rational.c
    cyclotom.c
    unknown.c
    finfield.c
    polynom.c
    permutat.c
    word.c
    costab.c
    tietze.c
    agcollec.c
    aggroup.c
    pcpresen.c
    list.c
    plist.c
    set.c
    vector.c
    vecffe.c
    range.c
    blister.c
    gstring.c
    string4.c
    record.c
    statemen.c
    function.c
    coding.c
    GapUtils.c
    GapUtils.h
    agcollec.h
    aggroup.h
    args.h
    blister.h
    buf_list.h
    coding.h
    comments.h
    complex.h
    conf.h
    costab.h
    cyclotom.h
    debug.h
    double.h
    endian.h
    eval.h
    exceptions_def.h
    exceptions_msg.h
    fft.h
    finfield.h
    function.h
    gstring.h
    hash.h
    hooks.h
    idents.h
    ieee754.h
    iface.h
    iface_batch.h
    iface_console.h
    iface_original.h
    iface_tcpip.h
    integer.h
    integer4.h
    interface.h
    ipc.h
    list.h
    machine_endian.h
    md5.h
    memmgr.h
    namespaces.h
    namespaces_bin.h
    objects.h
    pcpresen.h
    permutat.h
    plist.h
    polynom.h
    range.h
    rational.h
    read.h
    record.h
    scanner.h
    set.h
    spiral.h
    spiral_bag_list.h
    spiral_delay_ev.h
    spiral_fft.h
    spiral_fft.h
    spiral_spl_prog.h
    statemen.h
    string4.h
    system.h
    system_types.h
    tables.h
    tietze.h
    unknown.h
    vecffe.h
    vector.h
    word.h
)

##  Additional modules to compile when not on Windows...
if (!WIN32)
    list(APPEND SOURCES iface_tcpip.c ipc.c)
endif()

add_executable(${PROJECT} ${SOURCES})
target_compile_definitions(${PROJECT} PRIVATE ${SPIRAL_COMPILE_DEFNS})
target_compile_options    (${PROJECT} PRIVATE ${SPIRAL_COMPILE_FLAGS})

set_property(SOURCE system.c PROPERTY COMPILE_DEFINITIONS ${SYSSRCS_COMPILE_DEFNS})

set_target_properties(${PROJECT} PROPERTIES OUTPUT_NAME_DEBUG "${PROJECT}d")

if (WIN32)
    target_link_libraries(${PROJECT} shlwapi.lib)
else()
    target_link_libraries(${PROJECT} m)
endif()

install(TARGETS ${PROJECT} DESTINATION ${SPIRAL_SOURCE_DIR}/gap/bin)

enable_testing()

