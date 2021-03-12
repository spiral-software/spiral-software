@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details

set SGBETEMPDIR=%cd%

set OLDPATH=%PATH%
set PATH_FOR_PROFILER_COMPILER="C:\Program Files (x86)\mingw-w64\i686-8.1.0-posix-dwarf-rt_v6-rev0\mingw32\bin"
set PATH=%PATH%;%PATH_FOR_PROFILER_COMPILER%

REM  When SPIRAL is installed using the windows installer and the GNU GCC compiler
REM  is selected the user must specify the path to gcc.exe; this file is then
REM  updated with that path.  If a user builds SPIRAL standalone the
REM  PATH_FOR_PROFILER_COMPILER variable should be customized for the user's
REM  environment. 

make matrix -R -C ../../targets/win-x86-gcc GAP=%SGBETEMPDIR%\testcode.c STUB=%SGBETEMPDIR%\testcode.h CC="gcc" CFLAGS="-O2 -Wall -fomit-frame-pointer -march=native -std=c99" OUTDIR=%SGBETEMPDIR% -s > matrix.txt

set PATH=%OLDPATH%
