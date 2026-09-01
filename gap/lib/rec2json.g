
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(spiral.rec2json);

Declare(Rec2JSON);
Declare(List2JSON);

_wrapJSONstr := function(str)
    return "\""::String(str)::"\"";
end;


_val2jstr := function(val)
        if IsString(val) then
            return(_wrapJSONstr(val));
        elif IsRec(val) then
            return(Rec2JSON(val));
        elif IsList(val) then
            return(List2JSON(val));
        elif TYPE(val) in ["method", "function"] then
            return(_wrapJSONstr(PrintToString(val)));
        else
            return(String(val));
        fi;
end;


List2JSON := function(list)
    local jstr, val, add_comma;
    
    if not IsList(list) then
        return "";
    fi;    
    
    add_comma := false;
    jstr := "[";
    for val in list do
        if add_comma then
            Append(jstr, ",");
        fi;
        Append(jstr, _val2jstr(val));
        add_comma := true;
    od;
    Append(jstr, "]");
    
    return jstr;
end;


Rec2JSON := function(record)
    local jstr, f, val, add_comma;
    
    if not IsRec(record) then
        return "";
    fi;    
    
    add_comma := false;
    jstr := "{";
    for f in UserRecFields(record) do
        if add_comma then
            Append(jstr, ",");
        fi;
        Append(jstr, _wrapJSONstr(f)::":");
        val := record.(f);
        Append(jstr, _val2jstr(val));
        add_comma := true;
    od;
    Append(jstr, "}");
    
    return jstr;
end;


