@echo off

REM  Copyright (c) 2018-2019, Carnegie Mellon University
REM  See LICENSE for details

set SGBETEMPDIR=%cd%
make matrix -R -C ../../targets/win-x86-gcc GAP=%SGBETEMPDIR%\testcode.c STUB=%SGBETEMPDIR%\testcode.h CC="gcc" CFLAGS="-O2 -Wall -fomit-frame-pointer -march=native -std=c99" OUTDIR=%SGBETEMPDIR% -s > matrix.txt
