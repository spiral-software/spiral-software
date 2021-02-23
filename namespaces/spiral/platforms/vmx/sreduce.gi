
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


SIMD_VMX.CellFixProblems := function(c)
    local s;
    s := UnionSet(Set(List(Collect(c, [@(0,[vload_2h_4x32f, vload_2l_4x32f]), @1.cond(e->ObjId(e)=var), @2]), i->i.args[1])),
         Set(List(Collect(c, [@(0,[vload1_8x16i, vload2_8x16i, vload4_8x16i, vloadu_8x16i]), @1.cond(e->ObjId(e)=var), @2, @(3)]), i->i.args[1])));

    if Length(s) > 0 then
        return decl(List(s), c);
    else
        return c;
    fi;
end;
