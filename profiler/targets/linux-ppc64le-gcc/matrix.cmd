#!/bin/sh

#  Copyright (c) 2018-2025, Carnegie Mellon University
#  See LICENSE for details

##  Use cmake to build the project (PROJECT=matrix) for C language (SUFFIX=c) 

if [ "$1" = "build" ]; then
    ##  Build the code
    rm -rf build && mkdir build && cd build
    cmake -DPROJECT:STRING=matrix -DSUFFIX:STRING=c -DEXLIBS:STRING=papi .. > /dev/null
    make install > /dev/null
    cd ..
else
    ##  Run the code
    if [ -f ./matrix ]; then
        ./matrix > matrix.txt
    else
        touch matrix.txt
    fi
fi

