#! python

import sys
import subprocess

if len(sys.argv) < 3:
    print ( 'Usage: ' + sys.argv[0] + ' gap_exe_name gap_fil1 ... gap_filN' )
    sys.exit ( 'missing argument(s)' )

gap_exe = sys.argv[1]

nscrpt = len(sys.argv) - 2
fils = sys.argv[2:2+nscrpt]             ## ll args after name and gap_exe

##  print ( sys.argv[0] + ': gap_exe = ' + sys.argv[1] + ' and ' + str(nscrpt) + ' input files:' )
##  print ( fils )

##  build a command string like this: cat gap_fil1 ... gap_filN | gap_exe

cmdstr = 'cat '
for val in fils:
    cmdstr = cmdstr + val + ' '

cmdstr = cmdstr + '| ' + gap_exe
##  print ( cmdstr )

result = subprocess.run ( cmdstr, shell=True, check=True )
res = result.returncode

if (res != 0):
    print ( result )
    sys.exit ( res )

sys.exit ( res )

