
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


IsCyclicPerm := x -> IsSPL(x) and IsBound(x.isCyclic) and x.isCyclic;

CyclicPerms := function(e)
    local beg, mid, endd, len, list, i;
    beg := I(Rows(e)); mid := I(Rows(e)); endd := I(Cols(e));
    list := e.children();
    len := Length(list);
    i := 1;
    while i <= Length(list) and IsCyclicPerm(list[i])     do
        beg:=beg*list[i]; i:=i+1; 
    od;
    while i <= Length(list) and not IsCyclicPerm(list[i]) do mid:=mid*list[i]; i:=i+1; od;
    while i <= Length(list) and IsCyclicPerm(list[i])     do endd:=endd*list[i]; i:=i+1; od;
    return [beg, mid, endd];
end;
    
PullOutCyclicPerms := tensor ->
    SubstTopDownNR(tensor, [Tensor, @(1,Compose), @(2,Compose)],
	e -> let(p1 := CyclicPerms(@(1).val), 
	         p2 := CyclicPerms(@(2).val),
		 Tensor(p1[1], p2[1]) * Tensor(p1[2], p2[2]) * Tensor(p1[3], p2[3])));

