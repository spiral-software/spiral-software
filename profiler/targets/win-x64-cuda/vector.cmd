@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details

REM  Use cmake to build the project (PROJECT=cvector) for CUDA language (SUFFIX=cu) 

set SGBETEMPDIR=%cd%
COPY ..\..\targets\common\CMakeLists.txt %SGBETEMPDIR%\CMakeLists.txt
RENAME testcode.c testcode.cu
rm -rf build && md build && cd build
cmake -DPROJECT:STRING=cvector -DSUFFIX:STRING=cu .. < nul
cmake --build . --config Release --target install < nul
cd ..

IF EXIST .\cvector.exe (
    .\cvector.exe > vector.txt
) ELSE (
    type nul > vector.txt
)
