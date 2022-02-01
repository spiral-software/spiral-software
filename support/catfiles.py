#! python

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

import sys
import subprocess
import os

##  Simple script to concatenate files (used to build a xxxx.generator.g script)
##  Usage: catfiles dest input1 ... inputN
##  At least one input file must be specified

if len(sys.argv) < 3:
    print ( 'Usage: ' + sys.argv[0] + ' dest_script input1 ... inputN' )
    sys.exit ( 'missing argument(s)' )

dest = sys.argv[1]
if os.path.exists(dest):
    os.remove(dest)

nscrpt = len(sys.argv) - 2
fils = sys.argv[2:2+nscrpt]             ## all args after name and dest

concat = ''.join ( [open(f).read() for f in fils] )
fconc  = open ( dest, 'w' )
fconc.write ( concat )
fconc.close ()

sys.exit ( 0 )

