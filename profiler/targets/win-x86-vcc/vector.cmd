@echo off

REM  Copyright (c) 2018-2025, Carnegie Mellon University
REM  See LICENSE for details

REM  Use cmake to build the project (PROJECT=cvector) for C language (SUFFIX=c) 

IF "%1"=="build" (
    REM Build Phase
    IF EXIST .\build ( rd /s /q build )
    md build && cd build
    cmake -DPROJECT:STRING=cvector -DSUFFIX:STRING=c .. < nul
    cmake --build . --config Release --target install < nul
    cd ..
) ELSE (
    REM Run Phase
    IF EXIST .\cvector.exe (
        .\cvector.exe > vector.txt
    ) ELSE (
        type nul > vector.txt
    )
)
