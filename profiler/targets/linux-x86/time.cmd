#!/bin/sh
TEMPDIR=$PWD

make -R -C ../../targets/linux-x86 GAP="$TEMPDIR/testcode.c" STUB="$TEMPDIR/testcode.h" CC="gcc" CFLAGS="-O2 -Wall -fomit-frame-pointer -march=native -std=c99" OUTDIR="$TEMPDIR" -s > time.txt
