@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details 

REM  Use cmake to build the project (PROJECT=matrix) for C language (SUFFIX=c) 

set SGBETEMPDIR=%cd%
COPY ..\..\targets\common\CMakeLists.txt %SGBETEMPDIR%\CMakeLists.txt
IF EXIST .\build ( rd /s /q build )
md build && cd build
cmake -DPROJECT:STRING=matrix -DSUFFIX:STRING=c .. < nul
cmake --build . --config Release --target install < nul
cd ..

IF EXIST .\matrix.exe (
    .\matrix.exe > matrix.txt
) ELSE (
    type nul > matrix.txt
)
