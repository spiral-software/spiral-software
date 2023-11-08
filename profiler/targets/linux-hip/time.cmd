#!/bin/sh

#  Copyright (c) 2018-2023, Carnegie Mellon University
#  See LICENSE for details

##  Use cmake to build the project (PROJECT=time) for HIP

TEMPDIR=$PWD
cp -f ../../targets/common/CMakeLists.txt $TEMPDIR/CMakeLists.txt > /dev/null
mv testcode.c testcode.cpp
rm -rf build && mkdir build && cd build
cmake -DPROJECT:STRING=time -DSUFFIX:STRING=cpp -DCMAKE_CXX_COMPILER=hipcc .. > /dev/null
make install > /dev/null
cd ..

if [ -f ./time ]; then
    ./time > time.txt
else
    touch time.txt
fi
