@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details

REM  Use cmake to build the project (PROJECT=time) for CUDA language (SUFFIX=cu) 

set SGBETEMPDIR=%cd%
COPY ..\..\targets\common\CMakeLists.txt %SGBETEMPDIR%\CMakeLists.txt
RENAME testcode.c testcode.cu
rm -rf build && md build && cd build
cmake -DPROJECT:STRING=time -DSUFFIX:STRING=cu .. < nul
cmake --build . --config Release --target install < nul
cd ..

IF EXIST .\time.exe (
    .\time.exe > time.txt
) ELSE (
    type nul > time.txt
)
