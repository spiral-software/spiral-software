@echo off

REM  Copyright (c) 2018-2019, Carnegie Mellon University
REM  See LICENSE for details

set SGBETEMPDIR=%cd%
make -R -C ../../targets/win-x86-gcc GAP=%SGBETEMPDIR%\testcode.c STUB=%SGBETEMPDIR%\testcode.h CC="gcc" CFLAGS="-O3 -march=native -std=c99 -Wno-implicit -Wno-aggressive-loop-optimizations" OUTDIR=%SGBETEMPDIR% -s  > time.txt
