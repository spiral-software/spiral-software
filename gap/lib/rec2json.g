
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(spiral.rec2json);

Declare(Rec2JSON);

_wrapJSONstr := function(str)
    return "\""::String(str)::"\"";
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
        if IsString(val) then
            Append(jstr, _wrapJSONstr(val));
        elif IsRec(val) then
            Append(jstr, Rec2JSON(val));
        elif IsList(val) then
            Append(jstr, List2JSON(val));
        else
            Append(jstr, String(val));
        fi;
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
        if IsString(val) then
            Append(jstr, _wrapJSONstr(val));
        elif IsRec(val) then
            Append(jstr, Rec2JSON(val));
        elif IsList(val) then
            Append(jstr, List2JSON(val));
        else
            Append(jstr, String(val));
        fi;
        add_comma := true;
    od;
    Append(jstr, "}");
    
    return jstr;
end;