# -*- Mode: shell-script -*- 

PkgCVS := pkg -> List(Filtered(FileManager.files, x->x.pkg=pkg), x->x.id);

PkgFiles := pkg -> List(Filtered(FileManager.files, x->x.pkg=pkg), x->x.file);

PkgStats := pkg -> let(ff := Filtered(FileManager.files, x->x.pkg=pkg), 
    [ Length(ff), 
      Sum(ff, x->x.lines),
      Length(NSFields(pkg)) ]);

DisplayPkgStats := pkg -> let(stats := PkgStats(pkg), 
    Print(stats[1], " files\n", stats[2], " lines of code\n", 
	  stats[3], " identifiers\n"));

_pkg_resolve := function(pkg, usage)
    local inner;
    if Type(pkg) in [T_VAR, T_VARAUTO] then
	pkg := _WriteVar(pkg);
	return pkg;
    elif Type(pkg) = T_RECELM then
	inner := pkg;
	while Type(inner[1]) = T_RECELM do inner := inner[1]; od;  
	if not Type(inner[1]) in [T_VAR, T_VARAUTO] then Error(usage); fi;
	inner[1] := _WriteVar(inner[1]);
	return pkg;
    else Error(usage); 
    fi;
end;

_load := function(path, pkg, err_msg_pkg)
    local init1, init2, init3;
    init1 := Concatenation(path, ".g");
    InfoRead1( "#I  load tries \"", init1, "\"\n" );
    if sys_exists(init1)=1 and READ(init1, pkg) then return init1; fi;	

    init2 := Concatenation(path, PATH_SEP, "init.g");
    InfoRead1( "#I  load tries \"", init2, "\"\n" );
    if sys_exists(init2)=1 and READ(init2, pkg) then return init2; fi;	

    init3 := path;
    InfoRead1( "#I  load tries \"", init3, "\"\n" );
    if sys_exists(init3)=1 and READ(init3, pkg) then return init3; fi;	
    
    Error( "package '", Child(err_msg_pkg,1), "' is not installed. ",
           "Tried '", init1, "', '", init2, "', and '", init3, "'");
end;

#############################################################################
##
#F  load( <pkg> ) . . . . . . . . . . .  load package (new experimental function)
##  Loads <pkg> into namespace of the same name. No quotes needed.
##  Example: load(spiral.formgen);
##
load := UnevalArgs( function ( pkg )
    local path, usage, res;
    usage := "load( package.subpackage1.subpackage2... )\n";
    pkg   := _pkg_resolve(pkg, usage);
    path  := Concatenation(CurrentDir(), PATH_SEP, PathNSSpec(pkg));
    res := _load(path, pkg, pkg);
    WarnUndefined(res, Eval(pkg));
    return res;
end);

loadlocal := UnevalArgs( function ( pkg )
    local path, usage, ret, ns;
    usage := "load( package.subpackage1.subpackage2... )\n";
    #pkg   := _pkg_resolve(pkg, usage);
    path  := Concatenation(CurrentDir(), PATH_SEP, PathNSSpec(pkg));
    return _load(path, Local, pkg);
    ## if we do it like this (and remove prev. line) then
    ## package.subpackage will stay visible 
    # ns := EmptyPackage(pkg);
    # ret := _load(path, ns, pkg);
    # NamespaceAdd(Local, ns); 
    # return ret;
end);

loadimport := UnevalArgs( function ( pkg )
    local path, usage, res;
    usage := "load( package.subpackage1.subpackage2... )\n";
    pkg   := _pkg_resolve(pkg, usage);
    path  := Concatenation(CurrentDir(), PATH_SEP, PathNSSpec(pkg));
    res   := _load(path, pkg, pkg);
    WarnUndefined(res, Eval(pkg));
    PushNamespace(Eval(pkg));
    return res;
end);
