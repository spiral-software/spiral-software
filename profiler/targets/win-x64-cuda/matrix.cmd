@echo off

REM  Copyright (c) 2018-2025, Carnegie Mellon University
REM  See LICENSE for details

REM  Use cmake to build the project (PROJECT=matrix) for CUDA language (SUFFIX=cu) 

IF "%1"=="build" (
    REM Build Phase
    RENAME testcode.c testcode.cu
    IF EXIST .\build ( rd /s /q build )
    md build && cd build
    cmake -DPROJECT:STRING=matrix -DSUFFIX:STRING=cu .. < nul
    cmake --build . --config Release --target install < nul
    cd ..
) ELSE (
    REM Run Phase
    IF EXIST .\matrix.exe (
        .\matrix.exe > matrix.txt
    ) ELSE (
        type nul > matrix.txt
    )
)
