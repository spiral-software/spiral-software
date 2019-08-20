@echo off
set SGBETEMPDIR=%cd%
make matrix -R -C ../../targets/win-x86-gcc GAP=%SGBETEMPDIR%\testcode.c STUB=%SGBETEMPDIR%\testcode.h CC="gcc" CFLAGS="-O2 -Wall -fomit-frame-pointer -march=native -std=c99" OUTDIR=%SGBETEMPDIR% -s > matrix.txt
