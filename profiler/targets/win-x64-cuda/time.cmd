@echo off

REM  Copyright (c) 2018-2025, Carnegie Mellon University
REM  See LICENSE for details

REM  Use cmake to build the project (PROJECT=time) for CUDA language (SUFFIX=cu) 

IF "%1"=="build" (
    REM Build Phase
    RENAME testcode.c testcode.cu
    IF EXIST .\build ( rd /s /q build )
    md build && cd build
    cmake -DPROJECT:STRING=time -DSUFFIX:STRING=cu .. < nul
    cmake --build . --config Release --target install < nul
    cd ..
) ELSE (
    REM Run Phase
    IF EXIST .\time.exe (
        .\time.exe > time.txt
    ) ELSE (
        type nul > time.txt
    )
)
