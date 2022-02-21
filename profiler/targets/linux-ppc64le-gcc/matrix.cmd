#!/bin/sh

#  Copyright (c) 2018-2021, Carnegie Mellon University
#  See LICENSE for details

##  Use cmake to build the project (PROJECT=matrix) for C language (SUFFIX=c) 

TEMPDIR=$PWD
cp -f ../../targets/common/CMakeLists.txt $TEMPDIR/CMakeLists.txt
rm -rf build && mkdir build && cd build
cmake -DPROJECT:STRING=matrix -DSUFFIX:STRING=c -DEXLIBS:STRING=papi .. > /dev/null
make install > /dev/null
cd ..

if [ -f ./matrix ]; then
    ./matrix > matrix.txt
else
    touch matrix.txt
fi
