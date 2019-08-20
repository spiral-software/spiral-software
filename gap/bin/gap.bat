@echo off
rem #########################################################################
rem
rem gap.bat                     GAP                          Martin Schoenert
rem
rem This is a  batch file for the  MS-DOS operating system  that starts  GAP.
rem This is the place  where  you  make  all  the  necessary  customizations.
rem Then copy this file to a directory in your search path,  e.g.,  'C:\DOS'.
rem If you later move GAP to another location you must only change this file.
rem


rem #########################################################################
rem
rem GAP_DIR . . . . . . . . . . . . . . . . . . . . directory where GAP lives
rem
rem Set 'GAP_DIR' to the name of the directory where you have installed  GAP,
rem i.e., the directory with the subdirectories  'lib',  'grp',  'doc',  etc.
rem This name must not end  with  the  backslash  directory  separator ('\').
rem The default is  'C:\gap3r4p4',  i.e., directory 'gap3r4p4' on drive 'C:'.
rem You have to change this unless you have installed  GAP in this  location.
rem
set GAP_DIR=C:\gap3r4p4


rem #########################################################################
rem
rem GAP_MEM . . . . . . . . . . . . . . . . . . . amount of initial workspace
rem
rem Set 'GAP_MEM' to the amount of memory GAP shall use as initial workspace.
rem The default is 4 MByte, which is the minimal reasonable amount of memory.
rem You have to change it if you want  GAP to use a larger initial workspace.
rem If you are not going to run  GAP  in parallel with other programs you may
rem want to set this value close to the  amount of memory your  computer has.
rem
set GAP_MEM=4m


rem #########################################################################
rem
rem GAP_DIRSWAP . . . . . . . . . . . . .  directory where GAP should swap to
rem
rem Set 'GAP_DIRSWAP' to the name of the directory where  GAP  should put the
rem swap  file,  i.e.,  the file  'pg??????.386'  used  for  virtual  memory.
rem The drive of this directory must have at least  'GAP_MEM'+1MB free space.
rem The default is 'GAP_DIR', i.e., the swapfile is put in the GAP directory.
rem You may want to change this to a  ramdisk drive or a  faster local drive.
rem You must delete it if you already set 'GO32TMP', e.g., in 'autoexec.bat'.
rem
set GAP_DIRSWAP=%GAP_DIR%
set GO32TMP=%GAP_DIRSWAP%


rem #########################################################################
rem
rem GAP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . run GAP
rem
rem You  probably should  not change  this line,  which  finally starts  GAP.
rem
%GAP_DIR%\bin\gapdjg -m %GAP_MEM% -l %GAP_DIR%/lib/; -h %GAP_DIR%\doc\ %1 %2 %3 %4 %5 %6 %7 %8




