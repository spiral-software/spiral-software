#! python

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

import sys
import subprocess
import os

if len(sys.argv) < 4:
    print ( 'Usage: ' + sys.argv[0] + ' gpu_flag gap_exe_name gap_fil1 ... gap_filN' )
    sys.exit ( 'missing argument(s)' )

gpu_flag = sys.argv[1]
gap_exe = sys.argv[2]
gap_dir = os.path.dirname(gap_exe)

nscrpt = len(sys.argv) - 3
fils = sys.argv[3:3+nscrpt]             ## all args after name and gap_exe

##  print ( sys.argv[0] + ': gap_exe = ' + gap_exe + ' gpu-required-flag = ' + gpu_flag + ' and ' + str(nscrpt) + ' input files: ' )
##  print ( fils )

##  build a command string like this: cat gap_fil1 ... gap_filN | gap_exe

cmdstr = 'cat '
for val in fils:
    cmdstr = cmdstr + val + ' '

cmdstr = cmdstr + '| ' + gap_exe
##  print ( cmdstr )

spiral_path = os.getenv('SPIRAL_HOME', default=gap_dir)
if sys.platform == 'win32':
    checkgpustr = spiral_path + '/gap/bin/checkforGpu.exe'
else:
    checkgpustr = spiral_path + '/gap/bin/checkforGpu'

##  print ( 'Test for GPU using ' + checkgpustr )

if (gpu_flag == 'True' or gpu_flag == 'true' or gpu_flag == 'TRUE'):
    ##  GPU is required, test for one
    ##  print ( 'Testing for  GPU' )
    result = subprocess.run ( checkgpustr ) ## , shell=True, check=True
    res = result.returncode
    if (res != 0):
        print ( 'No suitable GPU found: Skipping test' )
        sys.exit ( 0 )   ## exit normally so failed test is not indicated
    ##  else:
    ##      print ( 'A GPU was found -- run the test' )

result = subprocess.run ( cmdstr, shell=True, check=True )
res = result.returncode

if (res != 0):
    print ( result )
    sys.exit ( res )

sys.exit ( res )

