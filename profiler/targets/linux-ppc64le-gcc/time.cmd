#!/bin/sh

#  Copyright (c) 2018-2021, Carnegie Mellon University
#  See LICENSE for details

##  Use cmake to build the project (PROJECT=papi_time) for C language (SUFFIX=c) 

TEMPDIR=$PWD
cp -f ../../targets/common/CMakeLists.txt $TEMPDIR/CMakeLists.txt
rm -rf build && mkdir build && cd build
cmake -DPROJECT:STRING=papi_time -DSUFFIX:STRING=c .. > /dev/null
make install > /dev/null
cd ..

if [ -f ./papi_time ]; then
    ./papi_time > time.txt
else
    touch time.txt
fi
