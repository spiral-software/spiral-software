@echo off

REM  Copyright (c) 2018-2021, Carnegie Mellon University
REM  See LICENSE for details

set basepath=%~dp0

REM use the extraargs var to add local config options to the profiler (eg., --keeptemp)
set extraargs=

shift
python "%basepath%\localprofiler.py" %* %extraargs%
