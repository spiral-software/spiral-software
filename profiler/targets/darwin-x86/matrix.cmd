#!/bin/sh

#  Copyright (c) 2018-2021, Carnegie Mellon University
#  See LICENSE for details

TEMPDIR=$PWD

make matrix -R -C ../../targets/darwin-x86 GAP="$TEMPDIR/testcode.c" STUB="$TEMPDIR/testcode.h" CC="gcc" CFLAGS="-O2 -Wall -fomit-frame-pointer -march=native -std=c99" OUTDIR="$TEMPDIR" -s > matrix.txt
