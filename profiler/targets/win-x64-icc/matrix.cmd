@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details

set SGBETEMPDIR=%cd%

set OLDPATH=%PATH%
set PATH_FOR_PROFILER_COMPILER="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.3.210\windows\bin"
set PATH=%PATH%;%PATH_FOR_PROFILER_COMPILER%

REM  When SPIRAL is installed using the windows installer and the Intel icl compiler
REM  is selected the user must specify the path to iclvars.bat; this file is then
REM  updated with that path.  If a user builds SPIRAL standalone the
REM  PATH_FOR_PROFILER_COMPILER variable should be customized for the user's
REM  environment. 

iclvars.bat intel64 > nul && make matrix -R -C ../../targets/win-x64-icc GAP=%SGBETEMPDIR%\testcode.c STUB=%SGBETEMPDIR%\testcode.h CC="icl" CFLAGS="/O3 /G7 /QxSSSE3" OUTDIR=%SGBETEMPDIR% -s > matrix.txt

set PATH=%OLDPATH%
