
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

TestFailExit := function()
    Print("TEST FAILED\n");
    Exit(-1);
end;


TestSkipExit := function()
    Print("Skipping test\n");
    Exit(86);
end;


#F  GetBasicProfilerTestFName (<pf>)
#F      Get the filename associated with Basic Profiler Test
#F      if <pf> return Success file name
#F      otherwise return Failure file name

GetBasicProfilerTestFName := function(pf)
    local path, sep;
    sep  := Conf("path_sep");
    path := Conf("spiral_dir");
    path := Concat(path, sep, "build");
    if pf then
        path := Concat(path, sep, "PROFILER_RUN_SUCCESS");
    else
        path := Concat(path, sep, "PROFILER_RUN_FAILED");
    fi;
    return path;
end;

#F  ClearBasicProfilerTestResults () -- remove results of prior run(s) of basic profiler test

ClearBasicProfilerTestResults := function()
    SysRemove(GetBasicProfilerTestFName(true));
    SysRemove(GetBasicProfilerTestFName(false));
    return;
end;


#F  MarkBasicProfilerTest (<pf>)
#F      Mark (create the filename) associated with Basic Profiler Test
#F      if <pf> ==> test successed, use success file name
#F      otherwise ==> test failed, use Failure file name

MarkBasicProfilerTest := function(pf)
    local path;
    path := GetBasicProfilerTestFName(pf);
    PrintTo(path, "");
    return;
end;


#F  CheckBasicProfilerTest () -- Return True if basic profiler test passed, otherwise, False

CheckBasicProfilerTest := function()
    local res, file;
    file := GetBasicProfilerTestFName(true);
    res  := CheckFileExists(file, "");
    
    if SysVerbose() > 0 then
        Print("Marker file: ", file);
        if res then PrintLine(" Found, return true"); else PrintLine(" NOT Found"); fi;
    fi;
    if res then return res; fi;
    
    file := GetBasicProfilerTestFName(false);
    res := CheckFileExists(file, "");
    if SysVerbose() > 0 then
        Print("Marker file: ", file);
        if res then PrintLine(" Found, return false"); else PrintLine(" NOT Found"); fi;
    fi;
    return false;
end;
