
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


VarFrontier := (e, depth) -> Cond(
    IsBound(e.isExpComposite) and e.isExpComposite, 
        Union(List(VarArgsExp(e), x->VarFrontier(x, depth))),
    IsVar(e), 
        When(depth=0, 
            Set([e]),  
            Union([e], Union(List(PredLoc(e), x->VarFrontier(x, depth-1))))), 
    Set([])
);

IsXRef := exp -> Intersection([var.table.X,var.table.X1,var.table.XY1], VarFrontier(exp, -1)) <> [];

IsYRef := exp -> Intersection([var.table.Y,var.table.Y1,var.table.XY1], VarFrontier(exp, -1)) <> []; 

CollectLoads := cmd -> Collect(cmd, [assign, @, @(1, [deref, nth], IsXRef)]);

CollectStores := cmd -> Collect(cmd, [assign, @(1, [deref, nth], IsYRef), @]);
