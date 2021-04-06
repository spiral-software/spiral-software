# -*- Mode: shell-script -*-

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(spiral.util);

Declare(IntString);

Concat := ConcatenationString;

PrintOps := CantCopy(rec(Print := s->s.print()));

PrintErr := arg->ApplyFunc(PrintTo, Concatenation(["*errout*"], arg));

PrintLine := arg->Print(ApplyFunc(Print, arg), "\n");

VPrintLine := arg->When(arg[1]>0, PrintTo("*errout*", ApplyFunc(Print, Drop(arg, 1)), "\n"));

SystemRecFields := [ ("__doc__"), ("__call__"),
                     ("__bases__"), ("operations"), ("__name__"), ("__cid__") ];

IsSystemRecField := name -> (name) in SystemRecFields;

UserRecFields := r -> Filtered(RecFields(r), f -> not IsSystemRecField(f));

UserNSFields := r -> Filtered(NSFields(r), f -> not IsSystemRecField(f));

UserRecValues := r -> List(UserRecFields(r), f -> r.(f));

###############################################################################
Declare(IsClass);

cid := rec(key:=1, nextKey := function() cid.key:=cid.key+1; return cid.key; end );

getID := function(obj)
  if IsClass(obj) and not IsBound(obj.__cid__) then  obj.__cid__ := cid.nextKey(); fi;
  return obj.__cid__;
end;
###############################################################################




###############################################################################
#F Inherit(<base1>, ..., <baseN>, <rec>) . . . . . . . . . . pseudo inheritance
##    Returns the result of copying fields from all bases into <rec> sequentia-
##    lly. This simulates inheritance, because all fields are just copied.
##
## See also: WithBases  (true inheritance)
##
##
Inherit := function (arg)
    local  result, base, field;
    if Length(arg) < 1 then
    Error("Usage: Inherit(<base1>, ..., <baseN>, <obj>)");
    fi;

    result := rec(  );
    for base in arg do
        for field in RecFields( base )  do
        result.(field) := base.(field);
        if IsBound(base.__doc__) then result.__doc__ := base.__doc__; fi;
        od;
    od;

    return _moveBasesUp(result);
end;

###############################################################################
#F InheritTab(<base1>, ..., <baseN>, <tab>) . . . . . . . . . . . . copy fields
##
## Same as Inherit, but works for namespaces, i.e., tab(...) .
##
InheritTab := function (arg)
    local  result, base, field;
    if Length(arg) < 1 then
    Error("Usage: InheritTab(<base1>, ..., <baseN>, <obj>)");
    fi;

    result := tab(  );
    for base in arg do
        for field in NSFields( base )  do
        result.(field) := base.(field);
        if IsBound(base.__doc__) then result.__doc__ := base.__doc__; fi;
        od;
    od;

    return result;
end;

CopyFields := Inherit;
CopyFieldsTab := InheritTab;

###############################################################################
#F WithBases(<base1>, ..., <baseN>, <obj>)
##
##
WithBases := function (arg)
    local  result, bases;
    if Length(arg) < 1 then
    Error("Usage: WithBases(<base1>, ..., <baseN>, <obj>)");
    fi;
    result := ShallowCopy(Last(arg));
    bases := Flat(arg{[1..Length(arg)-1]});
    if bases<>[] then result.__bases__ := bases; fi;
    _moveBasesUp(result);
    result.isClass := false;
    return result;
end;

_withBasesClass := function (arg)
    local  result, bases;
    result := ShallowCopy(Last(arg));
    bases := Flat(arg{[1..Length(arg)-1]});
    if bases<>[] then result.__bases__ := bases; fi;
    _moveBasesUp(result);
    return result;
end;



CompileClass := function (cls)
    local  result, base, field;
    if not IsBound(cls.__bases__) then 
        return cls; 
    else 
        result := rec(  );
        for base in Reversed(cls.__bases__) do
            CompileClass(base);
            for field in RecFields( base )  do
                result.(field) := base.(field);
            od;
        od;
        # override stuff from __bases__ with cls fields
        for field in RecFields( cls )  do
            result.(field) := cls.(field);
        od;
        # modify <cls> inplace to the result
        for field in RecFields( result )  do
            cls.(field) := result.(field);
        od;
        return _moveBasesDown(cls);
    fi;
end;

abstract := () -> ((arg) >> Error("not implemented"));

###############################################################################
#
#

AllClasses := WeakRef(tab());


###############################################################################
#R ClassOps . . . . . . . . . . . . . . . . . . . . . . Operations for classes
##

ClassOps := CantCopy(
    rec(
    operations := OpsOps,
    name := "ClassOps",
    Print := cls -> Print(cls.__name__),
    \= := (c1,c2) -> Same(c1,c2),
    \< := (c1,c2) -> Cond(IsClass(c1) and IsClass(c2) , getID(c1) < getID(c2), 
                          IsClass(c1), false,
                          true), #XXX
    ));

Declare(ClassBase);
ClassBase := CantCopy(rec(
    __bases__ := [],
    __name__ := "ClassBase",
    name := "ClassBase", # backwards compatiblity
    operations := ClassOps,
    isClass:=true,
    __cid__ := 0,

    # Printing class tree
    _basesTree := meth(self)
        local recursion;
        recursion := function(base, level) 
                        local e;
                        if not Same(base, ClassBase) then
                            Print(Blanks(level*4), base.__name__, "\n"); 
                            for e in base.__bases__ do
                                recursion(e, level+1);
                            od;
                        fi;
                    end;
        recursion(self, 0);
    end,

    _descendants := meth(self)
        local f, result;
        result := [];
        for f in UserNSFields(AllClasses) do
            if self in AllClasses.(f).__bases__ then Add(result, AllClasses.(f)); fi;
        od;
        return result;
    end,

    _all_descendants := meth(self)
        local r, s;
        r := [];
        s := [self];
        while Length(s)>0 do
            s := Set(ConcatList(s, e -> e._descendants()));
            Append(r, s);
        od;
        return r;
    end,

));

###############################################################################
#F IsClass(<obj>)
##
IsClass := x -> IsRec(x) and ((IsBound(x.isClass) and x.isClass) or
                             (IsBound(x.operations) and Same(x.operations, ClassOps)));


###############################################################################
#F Class(<ident>, <base1>, ..., <baseN>, <rec>) . . . . . . . . declare a class
##
Class := UnevalArgs(
    function(arg)
        local nam, z, bases, obj, result, comment, args, i, loc;
    SaveDefLine();
    comment := CommentBuffer();
    ClearCommentBuffer();
    if Length(arg) < 1
        then Error("Usage: Class(ident, base1, base2, ...)"); fi;
    nam := arg[1];
    if (BagType(nam) <> T_DELAY or not BagType(Child(nam, 1)) in [T_VAR, T_VARAUTO])
        and not BagType(nam) in [T_VAR, T_VARAUTO] then
        Error("<ident> is not a variable (but", TYPE(nam), ")\n",
          "Usage: Class(ident, base1, base2, ...)");
    fi;
    args := Drop(arg, 1);
    for i in [1..Length(args)] do
        args[i] := Eval(args[i]);
    od;
    z := SplitBy(args, IsClass);

    bases := Concatenation(z[1], [ClassBase]);
    obj := rec(__name__ := NameOf(nam));
    # make sure we do not pick up the default "defined in gap/lib/util.g" comment from above rec(...)
    obj.__doc__ := comment; 
    obj.name := obj.__name__; # backwards compatibility
    obj.__cid__ := cid.nextKey();
    obj := Concatenation([obj], z[2]);

    result := _withBasesClass(bases, ApplyFunc(Inherit, obj));
    Assign(nam, CantCopy(result));
    if IsBound(AllClasses.(result.__name__)) then
        Print("\nWarning: \"", result.__name__, "\" class redefined.\n");
        loc := DocLoc(AllClasses.(result.__name__));
        Print("last defined in \"", loc[1], "\":", loc[2], "\n");
        loc := DocLoc(result);
        Print("redefined in \"", loc[1], "\":", loc[2], "\n");
    fi;
    AllClasses.(result.__name__) := result;
    return result;
end);


#F PrintDel
#F
#F Print a list with the given delimiter between elements
PrintDel := function(lst,del)
    local i;
    if Length(lst) = 0 then return; fi;
    Print(lst[1]);
    for i in [2..Length(lst)] do
       Print(del, lst[i]);
    od;
end;

PrintCS := function(lst)
    PrintDel(lst, ", ");
end;



InfixPrint := function(lst, sep, print_func)
    local first, c;
    first := true;
    for c in lst do
        if first then first := false;
        else Print(sep); fi;
        print_func(c);
    od;
end;

PrintPad := function(text, width)
    if width > 0 then
        # align text left
        if width>Length(text) then
            return Print(text, Replicate(width-Length(text), ' '));
        fi;
    else
        # align text right
        if -width>Length(text) then
            return Print(Replicate(-width-Length(text), ' '), text);
        fi;
    fi;
    return Print(text);
end;

#F Pseudo class which denote base class of lists
Class(ListClass);

###############################################################################
#F ObjId( <obj>) ) . . . . . returns first superclass of <obj>
#F
#F Returns ListClass pseudo class is <obj> is a list
#F
ObjId := _ObjId;

_orig_ObjId := obj ->
   Cond(IsList(obj), ListClass,
        IsRec(obj) and IsBound(obj.__bases__), obj.__bases__[1],
    obj);

###############################################################################
#F RecCopy( <recA>, <recB ) . . copies the fields of record <recB> into <recA>
#F
RecCopy := function(a,b)
   local field;

   for field in RecFields(a) do
      Unbind( a.(field) );
   od;
   for field in RecFields(b) do
      a.(field) := b.(field);
   od;
end;

MergeIntoRecord := function (arg)
    local dest, src, field;
    if Length(arg) < 2 then
    Error("Usage: MergeIntoRecord(<dest>, <src1>, ..., <srcN>)");
    fi;

    dest := arg[1];
    for src in arg{[2..Length(arg)]} do
        for field in RecFields( src )  do
        dest.(field) := src.(field);
        od;
    od;
end;


Append2 := function ( arg )
    if Length(arg) > 1 then
    Append(arg[1], Concatenation(List(arg{[2..Length(arg)]}, String)));
    fi;
end;

Replace := function ( string, old, new )
    return String(ReplacedString(string, old, new));
end;


ReplaceAll := function ( string, old, new )
	local retstr, oldstr;
	retstr := String(string);
	repeat
		oldstr := retstr;
		retstr := Replace(oldstr, old, new);
	until retstr = oldstr;
	return retstr;
end;


###############################################################################
#F DoForAll( <list>, <func> )
#F     Applies <func> to each element in the list, discarding the result
#F
DoForAll := function(list, func)
    local i;
    if IsList(list) then
    for i in list do func(i); od;
    elif BagType(list) in [T_REC, T_NAMESPACE] then
    if NumArgs(func)=1 then
        for i in RecFields(list) do func(list.(i)); od;
    else
        for i in RecFields(list) do func(i, list.(i)); od;
    fi;
    else Error("<list> must be a list or a record");
    fi;
end;

###############################################################################
#F DoForAllButLast( <list>, <func> )
#F     Applies <func> to each element in the list except the last one,
#F     discarding the result
#F
DoForAllButLast := function(list, func)
    local i;
    for i in list{[1..Length(list)-1]} do func(i); od;
end;

###############################################################################
#F DoForAllWithSep( <list>, <func>, <sep_func> )
#F     Applies <func> to each element in the list, interleaving the calls with
#F     calls to <sep_func>. Discards the result
#F
DoForAllWithSep := function(list, func, sep_func)
    local e, xsep;
    xsep := Ignore;
    if IsList(list) then
    for e in list do
        xsep();
        func(e);
        xsep := sep_func;
    od;
    elif IsRec(list) then
    for e in UserRecFields(list) do
        xsep();
        func( list.(e) );
        xsep := sep_func;
    od;
    else
    Error("<list> must be a list or a record");
    fi;
end;


###############################################################################
#F ExtractFilePath(<fileName>) -- given a full path to the file, returns the 
#F                                directory without file name
#F
ExtractFilePath := function(fileName)
    local i, len;
    len := Length(fileName);
    for i in [len, len-1 .. 1] do 
        if fileName[i] in ['\\','/'] then return fileName{[1..i]}; fi;
    od;
    return "";
end;


###############################################################################
#F ExtractFileName(<fileName>) -- given a full path to the file, returns file 
#F                                name without a directory
#F
ExtractFileName := function(fileName)
    local i, len;
    len := Length(fileName);
    for i in [len, len-1 .. 1] do 
        if fileName[i] in ['\\','/'] then return fileName{[i+1..len]}; fi;
    od;
    return fileName;
end;


###############################################################################
#F IsAbsolutePath(<path>) -- returns true for a path that starts at root
#F
IsAbsolutePath := (path) -> Cond( Length(path)=0, false, 
                                  path[1]='\\' or path[1]='/', true,  
                                  Length(path)<3, false,
                                  path[2]=':' and path[3]='\\', true,
                                  false);

###############################################################################
#F ToWindowsPath(<path>) -- substitutes '/' by '\\'
#F
ToWindowsPath := (path) -> String(List(path, e->Cond(e='/', '\\', e)));

###############################################################################
#F Chain( <stmt1>, ... <res> )
#F     Functional form of statement sequence. For example
#F     printAndReturn := x -> Chain(Print(x), x);
#F
Chain := UnevalArgs(
    function (arg)
    DoForAllButLast(arg, (x) -> Eval(x));
    return Eval(Last(arg));
    end
);

###############################################################################

VariableStore := rec(
   _cnt := 0,
   _base := "X",

   newVar := meth(self)
       self._cnt := self._cnt + 1;
       return DelayedValueOf( ConcatenationString(self._base, String(self._cnt)) );
   end,

   reset := meth(self)
       self._cnt := 0;
   end
);

PrintMatEl := el -> Print(When(el=0, " ", el));

PrintMatRow := function(row, sep, print_el)
   DoForAllButLast(row, e -> Print(PrintMatEl(e), sep));
   if Length(row) > 0 then print_el(Last(row)); fi;
end;

PrintMat := function(mat)
   local row, first;
   Print("[ ");
   first := true;
   for row in mat do
      if first then first := false;
      else Print(", \n  "); fi;
      Print("[ ", PrintMatRow(row, ", ", PrintMatEl), " ]");
   od;
   Print(" ]\n");
end;

MapMat := (mat,func) ->
    List(mat, r -> List(r, func));

_eq := (a, b, thr) -> Cond(IsDouble(a) or IsDouble(b), AbsFloat(a-b) <= thr, a=b);

_elt := (i, thr) -> let(small := AbsComplex(Complex(i)) < thr,
    Cond(small, "  ",
    _eq(i, 1,thr), " 1",
    _eq(i,-1,thr), "-1",
    IsRat(i) and i>0, " q",
    IsRat(i) and i<0, "-q",
    IsInt(i) and i>0, " z",
    IsInt(i) and i<0, "-z",
    IsDouble(i) and i>0, " d",
    IsDouble(i) and i<0, "-d",
    IsComplex(i),
            Cond(_eq(ReComplex(i), 0, thr),
                 Cond(_eq(ImComplex(i),  1, thr), " j",
                      _eq(ImComplex(i), -1, thr), "-j",
                      " I"),
                 _eq(ImComplex(i), 0, thr),
                 Cond(_eq(ReComplex(i),  1, thr), " 1",
                      _eq(ReComplex(i), -1, thr), "-1",
                      ReComplex(i) < 0, " d",
                      "-d"),
                 " C"),
        IsCyc(i),
            Cond(i =  E(4), " j",
                 i = -E(4), "-j",
                 Im(i) = 0, Cond(i>0, " d", "-d"),
                 Re(i) = 0, " I",
                 " C"),
        " N"));

VisualMatStructure := x -> MapMat(x, i->_elt(i, 1e-10));

VisualizeMat := function(mat, sep)
   local row, first;
   Print("[ ");
   first := true;
   mat := VisualMatStructure(mat);
   for row in mat do
      if first then first := false;
      else Print(" \n  "); fi;
      Print("[", PrintMatRow(row, sep, Print), " ]");
   od;
   Print(" ]\n");
end;

ReallyPrintMatEl := (el, filename) -> AppendTo(filename, When(el=0, " ", el));

ReallyPrintMatRow := function(row, filename)
   DoForAllButLast(row, e -> AppendTo(filename, ReallyPrintMatEl(e, filename)));
   if Length(row) > 0 then AppendTo(filename, Last(row)); fi;
end;

RVM := function(funclist)
   local row, first, filename, mat, l, matlist, i;
   filename := "reallyvis/spiral.mat";

   matlist := List(funclist, i->spiral.spl.MatSPL(i));

   if not IsList(matlist) then
     matlist := [ matlist ];
   fi;

   PrintTo(filename, Length(matlist), "\n");

   i := 1;
   for mat in matlist do

      #if not IsList(mat) then
      #  mat := MatSPL(mat);
      #fi;

      l := Length(mat[1]);
      #AppendTo(filename, "[ ", Length(mat), ", ", l, " ]\n");
      AppendTo(filename, Length(mat), "\n");
      AppendTo(filename, l, "\n");
      AppendTo(filename, funclist[i].printlatex(), "\n");

      AppendTo(filename, "[ ");
      first := true;
      mat := VisualMatStructure(mat);
      for row in mat do
         if first then first := false;
         else AppendTo(filename, " \n  "); fi;
         AppendTo(filename, "[");
         ReallyPrintMatRow(row, filename);
         AppendTo(filename, " ]");
      od;
      AppendTo(filename, " ]\n");
      i := i+1;
   od;

  IntExec("cd reallyvis; . do");
end;


_vecPrintMatRow := function(row, sep, print_el, vlen, vsep)
   DoForAll([1..Length(row)-1], i -> Print(PrintMatEl(row[i]), When((i mod vlen)=0, vsep, sep)));
   if Length(row) > 0 then print_el(Last(row)); fi;
end;

#F vecVisualizeMat(<mat>, <sep>, <vlen>, <vsep>)
#F   mat -- matrix to visualize
#F   sep -- separator character between elements of a row
#F   vlen -- integer vector length, can be a pair of integers [v, h] (vertical, horiz vlen)
#F   vsep -- separator character between vectors (horizontally, vertically a newline is used)
#F
vecVisualizeMat := function(mat, sep, vlen, vsep)
   local i, row, first, hvlen, vvlen;
   Print("[ ");
   first := true;
   [vvlen, hvlen] := When(IsList(vlen), vlen, [vlen, vlen]);
   mat := VisualMatStructure(mat);
   for i in [1..Length(mat)] do
      row := mat[i];
      if first then first := false;
      else Print(" \n  ");
           if ((i-1) mod vvlen) = 0 then Print("\n  "); fi;
      fi;
      Print("[", _vecPrintMatRow(row, sep, Print, hvlen, vsep), " ]");
   od;
   Print(" ]\n");
end;

_latex_elt := (i, threshold) -> let(small := AbsComplex(Complex(i)) < threshold,
    Cond(small, " .",
    i=1, "\\phantom{-}1",
    i=-1, "-1",
    IsRat(i) and i>0, "\\phantom{-}r",
    IsRat(i) and i<0, "-r",
    IsInt(i) and i>0, "\phantom{-}i",
    IsInt(i) and i<0, "-i",
    IsDouble(i) and i>0, "\phantom{-}d",
    IsDouble(i) and i<0, "-d",
    IsComplex(i) or IsCyc(i), "\\phantom{-}c"));

LatexVisualizeMat := function(mat)
   local row, first, e;
   Print("\\left( \\begin{matrix}\n");
   first := true;
   mat := MapMat(mat, i->_latex_elt(i, 1e-10));
   for row in mat do
      if first then first := false;
      else Print(" \n  "); fi;
      Print("    ", DoForAllButLast(row, e->Print(e, " & ")),
            When(row<>[], Last(row)), " \\\\");
   od;
   Print("\\end{matrix} \\right)\n");
end;

LatexVisualizeMatSmall := function(mat)
   local row, first, e;
   Print("\\left[ \\begin{smallmatrix}\n");
   first := true;
   mat := MapMat(mat, i->_latex_elt(i, 1e-10));
   for row in mat do
      if first then first := false;
      else Print(" \n  "); fi;
      Print("    ", DoForAllButLast(row, e->Print(e, " & ")),
            When(row<>[], Last(row)), " \\\\");
   od;
   Print("\\end{smallmatrix} \\right]\n");
end;


ChildChain := function(obj, lst)
    local res, i, len;
    res := obj; i := 1; len := Length(lst);
    while i <= len do
       res := Child(res, lst[i]);
       i := i + 1;
    od;
    return res;
end;


# =========================================
# Manipulating functions
# =========================================
#
#F NumArgs( <function> )
#F     Returns number of arguments function takes, or -1 if function
#F     takes variable number of arguments.
#F
DocumentVariable(NumArgs);

#F NumGenArgs( <function-or-method> )
#F     Returns number of general purpose arguments, if an argument <func>
#F     is a function it is equal to NumArgs(<func>), however if <func>
#F     is a method number of general purpose arguments is one less.
NumGenArgs := func -> Cond(IsFunc(func), NumArgs(func),
                           IsMeth(func), Subst(When(x=-1, -1, x-1), x=>NumArgs(func)),
               Error("<func> must be a function or a method"));


IsCallable := func -> IsFunc(func) or IsMeth(func) or (IsRec(func) and IsBound(func.__call__));

IsCallableN := (func, nargs) ->
    Cond(IsFunc(func) or IsMeth(func), NumGenArgs(func) = nargs or NumArgs(func) = -1,
     IsRec(func) and IsBound(func.__call__), IsCallableN(func.__call__, nargs),
     false);

#F NumLocals( <function> )
#F     Returns number of local variables function defines.
#F
DocumentVariable(NumLocals);

#F ParamsFunc( <function> )
#F     Returns a string containing the list of parameters of a function
#F     separated by a comma.
#F
ParamsFunc := function(func)
    local params;
    Constraint(IsFunc(func));
    params := Reversed(
           List(Reversed(Children(func)) {[3+NumLocals(func) ..
                                       NumArgs(func)+NumLocals(func)+2]},
           NameOf));
    return Concatenation(
           Concatenation(
           List(params{[1..Length(params)-1]}, x -> Concat(x, ", "))),
           params[Length(params)]);
end;

#F ParamsMeth( <method> )
#F     Returns a string containing the list of parameters of a function
#F     separated by a comma.
#F
ParamsMeth := function(func)
    local params;
    Constraint(IsMeth(func));
    params := Reversed( # omits first parameters which is usually 'self'
           List(Reversed(Children(func)) {[3+NumLocals(func) ..
                                       NumArgs(func)+NumLocals(func)+1]},
           NameOf));
    return Concatenation(
           Concatenation(
           List(params{[1..Length(params)-1]}, x -> Concat(x, ", "))),
           params[Length(params)]);
end;

#F PermuteParamsFunc( <function>, <perm> )
#F     Returns new function with arguments in the permuted order.
#F     For example  PermuteParamsFunc( (x,y,z) -> x + 2*y +z, (2,3))
#F     returns  (x,z,y) -> x + 2*y + z
#F
PermuteParamsFunc := function(func, perm)
    local res, nargs, nloc, args, i, start;
    Constraint(IsFunc(func));
    Constraint(NumArgs(func) <> -1);
    nargs := NumArgs(func);
    nloc  := NumLocals(func);
    if nargs = 1 then return func;
    else
    if not IsBound(CopyFunc) then Error("GAP is not patched to copy functions"); fi;
    res := CopyFunc(func);
    if res = func then Error("GAP can't copy functions correctly"); fi;
    args := Reversed(
        Reversed(Children(func)){[2 + nloc .. nargs + nloc + 1]});
    args := Permuted(args, perm);
    start := Length(Children(res)) - nargs - nloc - 2;
    for i in [1..Length(args)] do
        SetChild(res, start + i, args[i]);
    od;
    return res;
    fi;
end;

#F ReverseParamsFunc( <function> )
#F     Returns new function with arguments in the reverse order.
#F     For example  (x,y) -> x+2*y  will become (y,x) -> x+2*y.
#F     This can be used for functional transposition.
#F
ReverseParamsFunc := function(func)
    Constraint(IsFunc(func));
    Constraint(NumArgs(func) <> -1);
    return PermuteParamsFunc(func, PermList(Reversed([1..NumArgs(func)])));
end;

# =========================================
# Current date and time
# =========================================
#

#F Date()
#F     Returns current date as a list [year, month, day, hour, min, sec]
#F
Date := function()
    local year, month, day, hr, min, secs;
	
    secs := TimeInSecs() + TimezoneOffset();
    year := 1970;
    day := Int(secs / (3600*24)) + 1; # ceil
    while day > DaysInYear(year) do
        day := day - DaysInYear(year);
    secs := secs - DaysInYear(year) * 24 * 3600;
        year := year + 1;
    od;
    month := 1;
    while day > DaysInMonth(month, year) do
        day := day - DaysInMonth(month, year);
    secs := secs - DaysInMonth(month, year) * 24 * 3600;
        month := month + 1;
    od;
    secs := secs - (day-1) * 24 * 3600;
    hr := Int(secs / 3600); # floor
    min := Int((secs - (hr * 3600)) / 60); # floor
    secs := secs - hr*3600 - min*60;
    return [year, NameMonth[month], day, hr, min, secs];
end;

#F Time()
#F     Prints current time.
#F
Time := function()
    local date;
    date := Date();
    Print(date[4], ":", date[5], ":", date[6], "\n");
end;

#F TimedAction(<function-call>)
#F  Executes function call and returns a tuple
#F  [exec_time, result], where exec_time is execution time in seconds,
#F  and result is the return value of the function-call.
#F
#F  Example:
#F    [result, search_time] := TimedAction(DP(DFT(8)));
#F
TimedAction := UnevalArgs(function(action)
    local t, res;
    t := TimeInSecs();
    res := Eval(action);
    t := TimeInSecs() - t;
    return [res, t];
end
);

#F UTimedAction(<function-call>) 
#F    Same as TimedAction but uses micro-second resolution timer 
#F
UTimedAction := UnevalArgs(function(action)
    local t, res;
    t := TimeInMicroSecs();
    res := Eval(action);
    t := TimeInMicroSecs() - t;
    return [res, t];
end
);


Methods := rec(
    True := arg >> true,
    False := arg >> false,
    Zero := arg >> 0,
    One := arg >> 1,
    Self := arg >> arg[1]
);

Max2 := (x,y) -> When(x>y, x, y);
Min2 := (x,y) -> When(x<y, x, y);

Max3 := (x,y,z) -> When(x>y, When(x>z, x, z), When(y>z, y, z));
Min3 := (x,y,z) -> When(x<y, When(x<z, x, z), When(y<z, y, z));

Class(Counter, rec(
    __call__ := (self, start) >> WithBases(self, rec(operations := PrintOps, n := start)),
    next := meth(self)
        local n;
    n := self.n;
        self.n := n+1;
    return n;
    end,

    print := self >> Print(self.__name__, "(", self.n, ")")
));

#F PrintEval(<format string>, <arg1>, <arg2>, ...)
#F
#F Format string is printed out, with '$N' being expanded into argN (1 <= N)
#F N could contain multiple digits.
#F Use $$ to print a '$'
#F
#F For example:
#F   spiral> PrintEval("$1 + $2\n", 3, 4);
#F   3 + 4
#F
PrintEval := function(arg)
    local i, invar, var, seps, ch, chst, value, st, pars;
    st := arg[1];
    invar := false;
    var := "";
    chst := " ";
    for ch in st do
        chst[1] := ch;
        if invar then
            if (ch >= '0' and ch <= '9') then
		Add(var, ch);
            else
		IsString(var);
		if var<>"" then
		    value := arg[1+IntString(var)];
		    Print(value);
		fi;
		if (ch<>'$' or var="") then Print(chst); invar := false; fi;
		var := "";
            fi;
	else
            if ch='$' then invar := true; var := "";
            else Print(chst);
            fi;
	fi;
    od;
    if invar then
        if IsString(var) and var<>"" then
            value := arg[1+IntString(var)];
            Print(value); 
        fi;
    fi;
end;

#F PrintEvalF(<format string>, <arg1>, <arg2>, ...)
#F
#F Format string is printed out, with '$N' being expanded into argN (1 <= N).
#F N could contain multiple digits.
#F Use $$ to print a '$'
#F
#F Same as PrintEval, but if argN happens to be a function it is invoked with no arguments
#F as argN();
#F
#F For example:
#F   spiral> PrintEvalF("$1 + $2\n", 3, ()->getNum());   # assume getNum returns 4234
#F   3 + 4234
#F
PrintEvalF := function(arg)
    local i, invar, var, seps, ch, chst, value, st;
    st := arg[1];
    invar := false;
    var := "";
    chst := " ";
    for ch in st do
        chst[1] := ch;
        if invar then
            if (ch >= '0' and ch <= '9') then
                Add(var, ch);
            else
                IsString(var);
                if var<>"" then
                    value := arg[1+IntString(var)];
                    Print( When( IsFunc(value), value(), value) );
                fi;
                if (ch<>'$' or var="") then Print(chst); invar := false; fi;
                var := "";
            fi;
        else
            if ch='$' then invar := true; var := "";
            else Print(chst);
            fi;
        fi;
    od;
    if invar then
        if IsString(var) and var<>"" then
            value := arg[1+IntString(var)];
            Print( When( IsFunc(value), value(), value) );
        fi;
    fi;
end;

# Checks if string ends with substring
EndsWith := (str,suffix) -> TakeLast(str,Length(suffix)) = suffix;

# Checks if string starts with substring
StartsWith := (str,prefix) -> Take(str,Length(prefix)) = prefix;

# ASCII codes for printable characters
INT_CHAR := tab(
    ("") := 0,
    ("\n") := 10,

    \ := 32,    \!:= 33,    \":= 34,    \#:= 35,    \$:= 36,
    \%:= 37,    \&:= 38,    \':= 39,    \(:= 40,    \):= 41,
    \*:= 42,    \+:= 43,    \,:= 44,    \-:= 45,    \.:= 46,    \/:= 47,

    0 := 48,    1 := 49,    2 := 50,    3 := 51,    4 := 52,
    5 := 53,    6 := 54,    7 := 55,    8 := 56,    9 := 57,

    \::= 58,    \;:= 59,    \<:= 60,    \=:= 61,    \>:= 62,
    \?:= 63,    \@:= 64,

    A := 65,    B := 66,    C := 67,    D := 68,    E := 69,
    F := 70,    G := 71,    H := 72,    I := 73,    J := 74,
    K := 75,    L := 76,    M := 77,    N := 78,    O := 79,
    P := 80,    Q := 81,    R := 82,    S := 83,    T := 84,
    U := 85,    V := 86,    W := 87,    X := 88,    Y := 89,    Z := 90,

    \[:= 91,    \\:= 92,    \]:= 93,    \^:= 94,    _ := 95,    \`:= 96,

    a := 97,    b := 98,    c := 99,    d := 100,   e := 101,
    f := 102,   g := 103,   h := 104,   i := 105,   j := 106,
    k := 107,   l := 108,   m := 109,   n := 110,   o := 111,
    p := 112,   q := 113,   r := 114,   s := 115,   t := 116,
    u := 117,   v := 118,   w := 119,   x := 120,   y := 121,   z := 122,

    \{:= 123,   \|:= 124,   \}:= 125,   \~:= 126,
); #"

# Convert string to an int (may be useful for hashing)
IntString := function( s )
    local res, i, base, ofs, negative;
    negative := false;
    if s[1] = '-' then
      negative := true;
      s := RemoveList(s, '-');
    fi;

    [res, base, ofs] := [0, 10, 48];
    for i in List(s,c->INT_CHAR.([c])) do
        if (i < ofs) or (i > ofs + 9) then
            [base, ofs] := [256, 0];
        fi;
        res := res * base + (i - ofs);
    od;
    if negative = false then
        return res;
    else
        return (0-res);
    fi;

end;

# Make a string from args using arg[1] as separator (a la perl's join).
StringJoin := arg -> let(
    sep := arg[1],
    MkString(Drop(Flat(
            List(Flat(Drop(arg,1)),e->[sep,e])),1)));

# Picks a random element r in the list. Returns [r, list\r]
PickRandomList := list -> let(
    L := Length(list),
    When( L > 0, let(
        i := RandomList([1..L]),
        [list[i],Concat(list{[1..i-1]},list{[i+1..L]})]),
    [,[]]));

# fPrint is a Print statement that is also a function that returns 0. This
# is useful to debug let statement like that:
# let(x:=foo(),discard:=fPrint(["x =", x]),bar(x))
fPrint:=function(l)
local a;
   for a in l do
      Print(a);
   od;
return 0;
end;

RandomHexStr := function(n) 
    local hex;
    hex := ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'];
    return List([1..n], i->Random(hex));
end;

GenGUID := () -> "{" :: RandomHexStr(8) :: "-" :: RandomHexStr(4) :: "-" :: 
                        RandomHexStr(4) :: "-" :: RandomHexStr(4) :: "-" ::
                        RandomHexStr(12) :: "}";



##  Utility functions to get information from memory manager

#F  GetMemMgrTrace()
#F      Turn on and display tracing information for one iteration of garbage collection
#F
GetMemMgrTrace := function()
    local currTrace, currMsg;
    currTrace := GASMAN("traceSTAT");
    currMsg   := GASMAN("messageSTAT");
    if currMsg   > 0 then GASMAN("message"); fi;
    if currTrace = 0 then GASMAN("traceON"); fi;
    GASMAN("collect");
    if currMsg   > 0 then GASMAN("message"); fi;
    if currTrace = 0 then GASMAN("traceOFF"); fi;
end;

#F  PrintResetRuntimeStats()
#F      Print the runtime (timing) information for Spiral and reset it
#F
PrintResetRuntimeStats := function()
    GetRuntimeStats();
    ResetRuntimeStats();
end;


#F CheckFileExists (file, folder)
#F     If <folder> is not "" look in <folder> under "spiral_dir" for <file>
#F     otherwise, look for a file named <file>
#F     Return True is found, otherwise, False

CheckFileExists := function(file, folder)
    local path, res, sep;
    res := false;
    
    if file = "" then
	PrintLine("Usage: CheckFileExists (<file>, <folder>); file name required");
	return res;
    fi;
    if folder <> "" then
	path := Conf("spiral_dir");
        sep  := Conf("path_sep"); 
        path := Concat(path, sep, folder, sep, file);
    else
	path := file;
    fi;

    if sys_exists(path) <> 0 then res := true; else res := false; fi;
    return res;
end;

#F IntFromHexString(<str>)
#F     Accepts a hexadecimal string and returns the Integer value.
#F     The string may optionally have a leading "0x" and both upper and lower
#F     case letters are supported, i.e., 0XF00D1 = 0xf00d1

IntFromHexString := function(hexstr)
    local digs, vals, res, i, hstr, posch;
    
    digs := "0123456789abcdefABCDEF";	    # hex digits, cater for upper/lower case alpha
    vals := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15 ];

    if StartsWith(hexstr, "0x") or StartsWith(hexstr, "0X") then
        hstr := SubString(hexstr, 3);	    # skip over leading "0x"
    else
	hstr := hexstr;
    fi;

    res := 0;
    for i in [1..Length(hstr)] do
    	posch := Position(digs, hstr[i]);
	if posch = false then
	    Error("usage: string must consist of decimal digits and letter a-f (or A-F) only");
	    return 0;
	fi;
	res := res * 16 + vals[posch];
    od;

    return res;
end;


#F ToUpperHexStr(hexstr [, sup])
#F     Accepts a hexadecimal string and returns the uppercase version of the string
#F     The string may optionally have a leading "0x"
#F     If <sup> = true leading zero digits (but not "0x") are suppressed

ToUpperHexStr := function( arg )
    local res, pos, ihd, ohd, lhstr, i, posch, hexstr, sup;

    hexstr := arg[1];
    if Length(arg) = 2 then sup := arg[2]; else sup := false; fi;
    
    ihd := "0123456789abcdefABCDEF";	    # hex digits, cater for upper/lower case alpha
    ohd := "0123456789ABCDEFABCDEF";

    res := [];
    pos := 0;
    lhstr := hexstr;
    
    if StartsWith(hexstr, "0x") or StartsWith(hexstr, "0X") then
        res := "0X";
	pos := 2;
	lhstr := SubString(hexstr, 3);
    fi;

    if sup then
        while lhstr[1] = '0' do
	    lhstr := SubString(lhstr, 2);
	od;
    fi;
    
    for i in [1..Length(lhstr)] do
    	posch := Position(ihd, lhstr[i]);
	if posch = false then
	    Error("usage: string must consist of decimal digits and letter a-f (or A-F) only");
	    return "";
	fi;
    	res[pos + i] := ohd[posch];
    od;

    return res;
end;


#F HexStringFromInt( <int> [, <pre> [, <case>]] )
#F     Accepts a positive integer and returns the corresponding hexadecimal string
#F     If <pre> is true the resulting string is prefixed with "0X"
#F     If <case> is true then the output string uses Upper case letters, otherwise, lowercase

HexStringFromInt := function( arg )
    local hexl, hexu, lis, lisr, dig, res, val, pre, case;

    val := arg[1];
    if Length(arg) >= 2 then pre  := arg[2]; else pre  := false; fi;
    if Length(arg)  = 3 then case := arg[3]; else case := false; fi;

    hexl := "0123456789abcdef";
    hexu := "0123456789ABCDEF";
    lis := []; lisr := []; res := "";
    
    while val > 0 do
    	dig := val mod 16;
	Add(lis, dig);
	val := (val - dig) / 16;
##	val := val / 16;
    od;
    lisr := Reversed(lis);

    if case = true then		    # return uppercase letters in string
        if pre = true then	    # prepend result with "0X"
	    res := "0X";
	fi;
	Append (res, List(lisr, x->hexu[x+1] ) );
    else			    # return lowercase letters in string
        if pre = true then
	    res := "0x";
	fi;
	Append (res, List(lisr, x->hexl[x+1] ) );
    fi;
    return res;
end;


#F TimeStamp()
#F     Return current time formatted as hh:mm:ss

TimeStamp := function()
    local start, chrs, itim, res;

    chrs  := "0123456789:";
    itim  := [ 0, 0, 10, 0, 0, 10, 0, 0 ];
    start := Date();
    res := "";

    if start[4] >= 10 then itim[1] := Int(start[4] / 10); fi;
    itim[2] := start[4] mod 10;
    if start[5] >= 10 then itim[4] := Int(start[5] / 10); fi;
    itim[5] := start[5] mod 10;
    if start[6] >= 10 then itim[7] := Int(start[6] / 10); fi;
    itim[8] := start[6] mod 10;

    Append ( res, List ( itim, x->chrs[x+1] ) );
    return res;
end;

#F ElapsedTime( begin_tim, end_tim )
#F     Return the delta in seconds between two timestamps: end_tim - begin_tim
#F     Timestamps are assumed to be formatted as hh:mm:ss
#F     If beg_tim > end_time then 24 hours (86,400 sec) is added to get a +ve result

ElapsedTime := function(begt, endt)
    local bt, et, delta, subs;

    subs := begt{ [1, 2] };
    bt   := IntString(subs) * 3600;
    subs := begt{ [4, 5] };
    bt   := bt + IntString(subs) * 60;
    subs := begt{ [7, 8] };
    bt   := bt + IntString(subs);

    subs := endt{ [1, 2] };
    et   := IntString(subs) * 3600;
    subs := endt{ [4, 5] };
    et   := et + IntString(subs) * 60;
    subs := endt{ [7, 8] };
    et   := et + IntString(subs);

    delta := et - bt;
    if delta < 0 then delta := delta + 86400; fi;
    return delta;
end;
