#!/bin/sh

#  Copyright (c) 2018-2025, Carnegie Mellon University
#  See LICENSE for details

##  Use cmake to build the project (PROJECT=matrix) for CUDA language (SUFFIX=cu) 

if [ "$1" = "build" ]; then
    ##  Build the code
    mv testcode.c testcode.cu
    rm -rf build && mkdir build && cd build
    cmake -DPROJECT:STRING=cvector -DSUFFIX:STRING=cu .. > /dev/null
    make install > /dev/null
    cd ..
else
    ##  Run the code built earlier
    if [ -f ./cvector ]; then
	./cvector > vector.txt
    else
	touch vector.txt
    fi
fi
