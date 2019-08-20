#!/bin/sh
TEMPDIR=$PWD

make -R -C ../../targets/linux-ppc64le-gcc GAP="$TEMPDIR/testcode.c" STUB="$TEMPDIR/testcode.h" CC="gcc" CFLAGS="-O2 -Wall -fomit-frame-pointer -std=c99" OUTDIR="$TEMPDIR" -s > time.txt
