#!/bin/sh

#  Copyright (c) 2018-2021, Carnegie Mellon University
#  See LICENSE for details

##  Use cmake to build the project (PROJECT=matrix) for CUDA language (SUFFIX=cu) 

TEMPDIR=$PWD
cp -f ../../targets/common/CMakeLists.txt $TEMPDIR/CMakeLists.txt
mv testcode.c testcode.cu
rm -rf build && mkdir build && cd build
cmake -DPROJECT:STRING=cvector -DSUFFIX:STRING=cu .. > /dev/null
make install > /dev/null
cd ..

if [ -f ./cvector ]; then
    ./cvector > vector.txt
else
    touch vector.txt
fi
