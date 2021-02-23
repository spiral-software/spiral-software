# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(spiral.sys);

#F ===========================================
#F SPIRAL Configuration Layer 'sys_conf'
#F ===========================================
#F

#F Simple accessor functions
#F =========================
#F

#F Conf( <key> )
#F    Return a config value with key <key>, and generate an error if the value
#F    is not defined.
#F
Conf := function ( key )
    Constraint(IsString(key));
    return gap_config_val_t(
	    config_demand_val(key));
end;

#F ConfHasVal( <key> )
#F    Return 'true' of configuration value with key <key> is defined, 'false'
#F    otherwise.
#F
ConfHasVal := function ( key )
    Constraint(IsString(key));
    if config_get_val(key) = "NULL" then return false;
    else return true;
    fi;
end;

#F OS Related functions
#F ====================
#F

#F SYS_EXEC ( <command> [, <args>] )
#F    Portable 'execute' system call for external programs execution
#F
SYS_EXEC := function ( arg ) 
    local cmd;
    cmd := command_quotify_static(Concatenation(List(arg, String)));
    if SysVerbose() = 2 then Print("gap: ", cmd, "\n"); fi;
    return Exec(cmd);
end;

#F SysRemove ( <file-name> )
#F    Delete the file from the filesystem
#F
SysRemove := function ( file )
    return sys_rm(file);
end;

#F SysMkdir ( <dir_name> )
#F    Create a directory
#F
SysMkdir := function ( file )
    return sys_mkdir(file);
end;

#F SysTmpName ( )
#F    Returns temporary file name
#F
SysTmpName := function ( )
    SysMkdir(Conf("tmp_dir"));
    return TmpName();
end;

#F IntExec( <command> )
#F     Execute a command and return its completion status
_todoIntExec := function( arg )
    local t,cmd,p;
    t := SysTmpName();
    cmd := MkString( arg, "; echo \"status := $? ;\" >",t);
    p := EmptyPackage(p);
    if SysVerbose() = 2 then Print("gap: ", cmd, "\n"); fi;
    Exec(cmd);
    READ(t,p);
    SysRemove(t);
    return p.status;
end;

#F SysExec( <command> [, <args>] )
#F     Execute a command and return its completion status
SysExec := function( arg )
    local cmd;
    cmd := MkString( arg );
    if SysVerbose() > 0 then Print("gap: ", cmd, "\n"); fi;
    return IntExec( cmd );
end;

#F ListFiles( <shell-pattern> )
#F     List files matching pattern
ListFiles := function( pattern )
    local t,p,cmd;
    t := SysTmpName();
    p := EmptyPackage(p);
    cmd := MkString(
	"perl -e 'print(\"ListFiles:=[\",",
	"join(\",\n\",map{s/(.*)/\"$1\"/and $_}<",pattern,">),",
	"\"];\n\")' >",t);
    if 0 = SysExec( cmd ) and READ(t,p) then
	p := p.ListFiles;
    else
	p := [];
    fi;
    SysRemove(t);
    return p;
end;

#F DifferFiles( <file1>, <file2> [, <diffopts>] )
#F     Use unix diff to compare files.
#F     Returns true if files differ.
DifferFiles := function( arg )
    local opts;
    opts := When( Length(arg) > 2, arg[3], "");
    return 0 <> SysExec( "diff ",opts," ",arg[1]," ",arg[2]," >/dev/null");
end;

#F Dirname( <path> )
#F     Get "a/path/to/" part of "a/path/to/name" (a la unix dirname).
#F     See also: Basename
Dirname := function( path )
    local p;
    p := LastPosition(path,PATH_SEP[1]);
    return When(p>0, Take(path,p),"./");
end;

#F Basename( path )
#F     Get "name.ext" part of "a/path/to/name.ext".
#F     Optionally, Basename( path, ".newext" ) yields "name.newext".
#F     See also: Dirname
Basename := function( arg )
    local fname,ext,p;
    fname := Drop(arg[1],LastPosition(arg[1],PATH_SEP[1]));
    if Length(arg) > 1 then
	p := LastPosition(fname,'.');
	fname := Take( fname, When(p>1,p-1,Length(fname)) );
	Append( fname, arg[2] );
    fi;
    return fname;
end;

#F FileExists( path )
#F     True if file can be read.
FileExists := fname -> 0 = SysExec("test -r ",fname);
