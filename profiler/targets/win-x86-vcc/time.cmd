@echo off

REM  Copyright (c) 2018-2025, Carnegie Mellon University
REM  See LICENSE for details

REM  Use cmake to build the project (PROJECT=rdtsc_time) for C language (SUFFIX=c) 

IF "%1"=="build" (
    REM Build Phase
    IF EXIST .\build ( rd /s /q build )
    md build && cd build
    cmake -DPROJECT:STRING=rdtsc_time -DSUFFIX:STRING=c .. < nul
    cmake --build . --config Release --target install < nul
    cd ..
) ELSE (
    REM Run Phase
    IF EXIST .\rdtsc_time.exe (
        .\rdtsc_time.exe > time.txt
    ) ELSE (
        type nul > time.txt
    )
)
