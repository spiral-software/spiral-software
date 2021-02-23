
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#   FF: needed to merge integer tables due to SAR grds. not sure where to put yet...
MergeIntData := function(s, opts)
    local datas, first, eqdata, subst;
    datas := Filtered(Set(Collect(s, @(1, var, e->IsBound(e.value) and not IsLoopIndex(e)))), i->i.t.t=TInt); 

    while datas <> [] do    
        first := datas[1];
        datas := Drop(datas, 1);
        eqdata := Filtered(datas, i->i.value=first.value);
        if Length(eqdata) > 0 then
            SubtractSet(datas, Set(eqdata));
            for subst in eqdata do
                SubstVars(s, rec((subst.id) := first));
            od;
        fi;
    od;
    return s;
end;


