@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details

REM  Use cmake to build the project (PROJECT=rdtsc_time) for C language (SUFFIX=c) 

set SGBETEMPDIR=%cd%
COPY ..\..\targets\common\CMakeLists.txt %SGBETEMPDIR%\CMakeLists.txt
IF EXIST .\build ( rd /s /q build )
md build && cd build
cmake -DPROJECT:STRING=rdtsc_time -DSUFFIX:STRING=c .. < nul
cmake --build . --config Release --target install < nul
cd ..

IF EXIST .\rdtsc_time.exe (
    .\rdtsc_time.exe > time.txt
) ELSE (
    type nul > time.txt
)
