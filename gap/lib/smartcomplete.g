
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

#F SmartComplete(a, str)
#F  Get all the fields from Record or Namespace a and print only
#F  those that start with the string str.
#F  This is meant to be used with SmartCompletion (CTR-W)
#F  on records and namespaces.
#F
#F  Example:
#F    SmartComplete(SpiralOptions,"");
#F

_smartCompleteLastA := "";
_smartCompleteLastB := "";
_smartCompleteFull := false;

_smartCompleteRec := (obj) -> Cond( not IsRec(obj),             [],
                                    not IsBound(obj.__bases__), RecFields(obj),
                                    Set(RecFields(obj) :: ConcatList(obj.__bases__, e -> _smartCompleteRec(e))));

SmartComplete:=function(a,b)
    local x, dirtybit, list;

    Constraint(IsString(b));
    
    dirtybit:=false;
    if (IsRec(a)) then
        if b<>"" # suggest everything what is bound
            or (    _smartCompleteLastA = a 
                and _smartCompleteLastB = b
                and _smartCompleteFull) 
        then
            list := _smartCompleteRec(a);
            _smartCompleteFull := false;
        else
            list := RecFields(a);
            _smartCompleteFull := true;
        fi;
        _smartCompleteLastA := a;
        _smartCompleteLastB := b;
    else
        _smartCompleteLastA := "";
        _smartCompleteLastB := "";
        if (IsNamespace(a)) then
	    list := Dir(a);
        else 
	    return "    identifier is neither a record nor a namespace\n";
        fi;
    fi;
    Sort(list);
    for x in list do
        if b=SubString(x,1,LengthString(b)) then
            Print("    " :: x :: "\n");
            dirtybit := true;
        fi;
    od;
    if (dirtybit=false) then
        Print("    identifier has no completions\n");
    fi;
end;
