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

##  print ( sys.argv[0] + ': dest = ' + dest + ' and ' + str(nscrpt) + ' input files: ' )
##  print ( fils )

##  build a command string like this: cat input1 ... inputN > dest

cmdstr = 'cat '
for val in fils:
    cmdstr = cmdstr + val + ' '

cmdstr = cmdstr + '> ' + dest
##  print ( cmdstr )

result = subprocess.run ( cmdstr, shell=True, check=True )
res = result.returncode

if (res != 0):
    print ( result )
    sys.exit ( res )

sys.exit ( res )

