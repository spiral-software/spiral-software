# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

SPIRAL_DIR := Conf("spiral_dir");
if ContainsElement(SPIRAL_DIR, ' ') then
	tmpnspdir := WinShortPathName(SPIRAL_DIR);
	if IsString(tmpnspdir) and Length(tmpnspdir) > 0 then
		SPIRAL_DIR := tmpnspdir;
	fi;
fi;

SPIRAL_PATHS := [  SPIRAL_DIR :: PATH_SEP :: "namespaces", "." ];

PACKAGES_PATHS := [ SPIRAL_DIR :: PATH_SEP :: "namespaces" :: PATH_SEP :: "packages", "." :: PATH_SEP :: "packages" ];

SPIRAL_LOAD_PATHS := [];

SPIRAL_PKG_ROOT := SPIRAL_DIR :: PATH_SEP :: "namespaces" :: PATH_SEP :: "spiral";

# NOTE:
# point pkg to an actual package object
# set package id correctly (maybe move this to be a field pkg.__id__?)
# set pkg.__path__ for Include to work 
#
_pkg_resolve2 := function(pkg, usage)
    local inner, ass;
	
    if IsGapVar(pkg) then
		return EmptyPackage(_GlobalVar(pkg)); # force variable to be in Global scope
    elif Type(pkg) = T_RECELM then
		inner := pkg;
		while Type(inner[1]) = T_RECELM do 
			inner := inner[1];
		od;  
		if not IsGapVar(inner[1]) then 
			Error(usage);
		fi;
		inner[1] := _GlobalVar(inner[1]); 
		return EmptyPackage(pkg);
    else 
		Error(usage); 
    fi;
end;


_Load := function(path, pkg)
    local init, dir, file, file_candidates, dir_candidates, ext, top, i, found;

    file_candidates := List(Concat(SPIRAL_PATHS, PACKAGES_PATHS, SPIRAL_LOAD_PATHS), base -> Concat(base, path, ".g"));
    dir_candidates := List(Concat(SPIRAL_PATHS, PACKAGES_PATHS, SPIRAL_LOAD_PATHS), base -> Concat(base, path, Conf("path_sep")));

    if Global.LoadStack <> [] then
		top := Last(Global.LoadStack);
		if top <> pkg then
			if not ContainsElement(top.__packages__, pkg) then
				Add(top.__packages__, pkg);
			fi;
		fi;
    fi;

    Add(Global.LoadStack, pkg);
    if not IsBound(pkg.__packages__) then
		pkg.__packages__ := [];
    fi;

    for file in file_candidates do
		InfoRead1( "#I  Load tries \"", file, "\"\n" );
		if sys_exists(file)=1 and READ(file, pkg) then 
			pkg.__file__ := file;
			RemoveLast(Global.LoadStack, 1);
			return file; 
		fi;	
    od;

    for dir in dir_candidates do
        file := Concat(dir, "init.g");
		InfoRead1( "#I  Load tries (dir) \"", file, "\"\n" );
		pkg.__files__ := [ file ];
		pkg.__dir__ := dir;
		if sys_exists(file)=1 and READ(file, pkg) then 
			RemoveLast(Global.LoadStack, 1);
			return file; 
		fi;	
    od;

    RemoveLast(Global.LoadStack, 1);
    
    Error("package '", pkg, "' is not installed. Tried ", 
		Concatenation(
			ConcatList(file_candidates, x->Concat("\n   ", x)),
			ConcatList(dir_candidates, x->Concat("\n   ", x, "init.g"))));
end;


_Include := function(path, pkg)
    local file;

    file := path :: ".gi"; InfoRead1( "#I  Include tries \"", file, "\"\n" );
    if sys_exists(file)=1 then 
        if READ(file, pkg) then 
            Add(pkg.__files__, file);
            return file; 
        else
            return Error("can't read the include file '", file, "'");
        fi;
    fi;
    return Error("include file '", file, "' does not exist");
end;


_LoadRedirect := function(path, pkg)
    local file, top;
    file := path :: PATH_SEP :: "init.g"; InfoRead1( "#I  LoadRedirect tries \"", file, "\"\n" );

    if Global.LoadStack <> [] then
		top := Last(Global.LoadStack);
		if top <> pkg then
			if not ContainsElement(top.__packages__, pkg) then
				Add(top.__packages__, pkg);
			fi;
		fi;
    fi;

    Add(Global.LoadStack, pkg);
    if not IsBound(pkg.__packages__) then
		pkg.__packages__ := [];
    fi;

    if sys_exists(file)=1 then 
		Local.__dir__ := path :: PATH_SEP;
        if READ(file, pkg) then 
            Add(pkg.__files__, file);
			RemoveLast(Global.LoadStack, 1);
            return file; 
        else
			RemoveLast(Global.LoadStack, 1);
            return Error("can't read the redirected file '", file, "'");
        fi;
    fi;
    RemoveLast(Global.LoadStack, 1);
    return Error("redirected file '", file, "' does not exist");
end;

#############################################################################
##
#F  Load( <pkg> ) 
##  Load <pkg> into namespace of the same name. No quotes needed.
##  Example: Load(spiral.formgen);
##
Load := UnevalArgs( function ( pkg )
    local path, usage, res;
    usage := "Load( package.subpackage1.subpackage2... )\n";
    pkg   := _pkg_resolve2(pkg, usage);
    path  := PATH_SEP :: PathNSSpec(NSId(pkg));
    res := _Load(path, pkg);
    WarnUndefined(res, Eval(pkg));
    return res;
end);

LoadImport := UnevalArgs( function ( pkg )
    local path, usage, res;
    usage := "LoadImport( package.subpackage1.subpackage2... )\n";
    pkg   := _pkg_resolve2(pkg, usage);
    path  := PATH_SEP :: PathNSSpec(NSId(pkg));
    res := _Load(path, pkg);
    WarnUndefined(res, Eval(pkg));
    Import(pkg);
    return res;
end);

Include := UnevalArgs( function ( pkg )
    local path, pkg, usage, ret, ns;
    usage := "Include( subfile )\n";
    if not IsGapVar(pkg) then 
		Error(usage); 
	fi;
    if not IsBound(Local.__dir__) then 
		Error("You can only Include() files from packages loaded with Load()"); 
	fi;
    path  := Local.__dir__ :: NameOf(pkg);
    return _Include(path, Local);
end);

#F LoadRedirect(<subdir-identifier>)
#F
#F This function should be used in init.g packages to redirect loading to subdir/init.g
#F Useful for multi-language directories:
#F mymodule/
#F    c/
#F    spiral/
#F        init.g <------- Include(file1); 
#F        file1.gi
#F    init.g  <--------  LoadRedirect(spiral)
#F
#F This setup allows for Load(mymodule) from within Spiral
#F
LoadRedirect := UnevalArgs( function ( pkg )
    local path, pkg, usage, ret, ns;
    usage := "LoadRedirect( subdir )\n";
    if not IsGapVar(pkg) then 
		Error(usage); 
	fi;
    if not IsBound(Local.__dir__) then 
		Error("You can only LoadRedirect() from within packages loaded with Load()"); 
	fi;
    path  := Local.__dir__ :: NameOf(pkg);
    return _LoadRedirect(path, Local);
end);




