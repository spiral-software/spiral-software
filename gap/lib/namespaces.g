# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(spiral.namespaces);

###############################################################################
_Import := ns >> Checked(BagType(ns) = T_NAMESPACE,
    When(IsBound(ns.__import__), ns.__import__(), PushNamespace(ns)));

#Import := arg -> DoForAll(arg, ns -> _Import(ns));

Import := function(arg)
    local top;

    if Global.LoadStack <> [] then
		top := Last(Global.LoadStack);
		DoForAll(arg, ns -> When(not ContainsElement(top.__packages__, ns), Add(top.__packages__, ns), false));
    fi;
    DoForAll(arg, ns -> _Import(ns));
end;


ImportAll := function(ns)
    _Import(ns);
    DoForAll(List(RecNamespace(ns), x->x), subns -> When(IsNamespace(subns), _Import(subns)));
end;

ImportGlobal := ns -> PushNamespaceGlobal(ns);

Unimport     := () -> PopNamespace();

UnimportAll  := () -> PopAllNamespaces();

CurrentImports := () -> CurrentFile().imports;

Package := UnevalArgs( function(name) 
    local pkg;
    pkg := StartPackage(name);
    if not IsBound(pkg.__doc__) then
		pkg.__doc__ := CommentBuffer();
    else
		Append(pkg.__doc__, CommentBuffer());
	fi;
    ClearCommentBuffer();
end);


###############################################################################
#F Dir(<pkg>) - list contents of a package
##
Dir := function(ns)
    local dir;
    dir := Filtered(NSFields(ns), x->IsBound(ns.(x)));
    Sort(dir);
    return dir;
end;


###############################################################################
#F DirFuncs(<pkg>) - list functions of a package
##
DirFuncs := ns -> Filtered(Dir(ns), x -> IsFunc(ns.(x)));


###############################################################################
#F DirPkgs(<pkg>) - list subpackages of a package
##
DirPkgs := ns -> Filtered(Dir(ns), x -> IsNamespace(ns.(x)));

###############################################################################
#F DirClasses(<pkg>) - list classes of a package
##
DirClasses := ns -> Filtered(Dir(ns), x -> IsClass(ns.(x)));


###############################################################################
#F DirOther(<pkg>) - list non-functions of a package
##
DirOther := ns -> Filtered(Dir(ns), x -> not (IsFunc(ns.(x)) or IsNamespace(ns.(x)) or IsClass(ns.(x))));


###############################################################################

FileManager.stats := self >> let(
    packages := Filtered(Set(List(FileManager.files, x->x.pkg)), IsNamespace),
    Print(Length(self.files), " loaded files\n",
	  Sum(self.files, x->x.lines), " lines of code\n",
	  Length(packages), " packages\n",
	  Sum(packages, x->Length(NSFields(x))), " identifiers\n"));


FileManager.findIdent := meth(self, name) 
    if Type(DelayedValueOf(name)) = T_VARAUTO then
		Eval(DelayedValueOf(name));
	fi;
    return Filtered(self.files, x -> x.pkg<>false and IsBound(x.pkg.(name)));
end;

