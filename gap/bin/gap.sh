#!/bin/sh
GAP_DIR = @srcdir@/gap
GAP_MEM = 16m
GAP_PRG = gap

exec $GAP_DIR/bin/$GAP_PRG -m $GAP_MEM -l $GAP_DIR/lib/ -h $GAP_DIR/doc/ $*
