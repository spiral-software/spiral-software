@echo off

REM  Copyright (c) 2018-2019, Carnegie Mellon University
REM  See LICENSE for details

rem call "%VS160COMNTOOLS%\VsDevCmd.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" > nul

set SGBETEMPDIR=%cd%
make -R -C ../../targets/win-x86-vcc GAP=%SGBETEMPDIR%\testcode.c STUB=%SGBETEMPDIR%\testcode.h CC="cl" CFLAGS="-O2" OUTDIR=%SGBETEMPDIR% -s  > time.txt

