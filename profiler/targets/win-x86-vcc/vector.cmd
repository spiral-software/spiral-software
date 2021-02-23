@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details 

REM For Visual Studio 2017 and later the environment variables are set by: VsDevCmd.bat.  The path
REM to VsDevCmd.bat can be determined with vswhere.exe, which is (by default) located at:
REM "C:\Program Files (x86)\Microsoft Visual Studio\Installer" We'll use this to locate
REM VsDevCmd.bat, then run it to set the environment for nmake...

for /f "usebackq delims=#" %%a in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere" -latest -property installationPath`) do call "%%a\Common7\Tools\VsDevCmd.bat" > nul

set SGBETEMPDIR=%cd%
nmake /C /S /f ../../targets/win-x86-vcc/Makefile GAP=%SGBETEMPDIR%\testcode.c STUB=%SGBETEMPDIR%\testcode.h CC="cl" OUTDIR=%SGBETEMPDIR% vector  > vector.txt
