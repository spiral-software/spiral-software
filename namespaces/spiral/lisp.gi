
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details

# Functions to export variousl GAP/SPL data structures as LISP expressions

vec2lisp := function(x)
    if Length(x) = 0 then
        return "";
    else
        if Length(x) = 1 then
            return String(x[1]);
        else
            return ConcatenationString(String(x[1])," ",
                           vec2lisp(Sublist(x,[2..Length(x)])));
        fi;
    fi;
end;


CatLines := function ( arg )
    local   res,  str;
    res := "";
    for str  in arg  do
        Append( res, str );
        Append( res, "\n");
    od;
    IsString( res );
    return res;
end;

Class(LISPExport, rec(
       toLISP := self >> Error("toLISP() not implemented for " :: self.__bases__[1].name)
));

