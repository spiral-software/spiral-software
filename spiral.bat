@echo off

set SPIRAL_DIR=%~dp0
set GAP_MEM=2048m

Rem uncomment for debugging profiler
Rem set PROFILER_LOCAL_ARGS=--debug --keeptemp

set SPIRAL_CONFIG_SPIRAL_DIR=%SPIRAL_DIR%
set SPIRAL_CONFIG_PATH_SEP=\

set OLDPATH=%PATH%
set PATH=%PATH%;%SPIRAL_DIR%profiler\bin

"%SPIRAL_DIR%gap\bin\gap.exe" -m %GAP_MEM% -x 1000 -l "%SPIRAL_DIR%gap\lib" "%SPIRAL_DIR%_spiral_win.g"

set PATH=%OLDPATH%
