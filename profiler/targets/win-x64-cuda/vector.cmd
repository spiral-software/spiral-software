@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details

set SGBETEMPDIR=%cd%

REM  Taking a simple approach here: copy CMakeLists.txt from win-x64-nvcc
REM  target to the temp directory; run cmake to configure then build the
REM  target and finally, if successful, execute the target.

COPY ..\..\targets\common\CMakeLists.txt %SGBETEMPDIR%\CMakeLists.txt
RENAME testcode.c testcode.cu
cmake -DPROJECT:STRING=cvector .
cmake --build . --config Release --target install

IF EXIST .\cvector.exe (
    .\cvector.exe > vector.txt
) ELSE (
    type nul > vector.txt
)
