#!/bin/sh

#  Copyright (c) 2018-2021, Carnegie Mellon University
#  See LICENSE for details

##  Use cmake to build the project (PROJECT=rdtsc_time) for C language (SUFFIX=c) 

TEMPDIR=$PWD
cp -f ../../targets/common/CMakeLists.txt $TEMPDIR/CMakeLists.txt
rm -rf build && mkdir build && cd build
cmake -DPROJECT:STRING=rdtsc_time -DSUFFIX:STRING=c -DEXFLAGS:STRING=-march\=native .. > /dev/null
make install > /dev/null
cd ..

if [ -f ./rdtsc_time ]; then
    ./rdtsc_time > time.txt
else
    touch time.txt
fi
