
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


@1 := @(1); 
@2 := @(2);
zero   := @.cond(e -> e=0 or  (IsValue(e) and (e.v=0  or e.v=0.0)));
one    := @.cond(e -> e=1 or  (IsValue(e) and (e.v=1  or e.v=1.0)));
negone := @.cond(e -> e=-1 or (IsValue(e) and (e.v=-1 or e.v=-1.0)));

IS := objid -> When(IsList(objid), 
    o -> ObjId(o) in objid,
    o -> Same(ObjId(o), objid)
);

IS_A := (o, objid) -> Same(ObjId(o), objid);

CHILDREN_ARE := (objid) -> (o -> ForAll(o.children(), IS(objid)));

NOT_ALL_CHILDREN_ARE := (objid) -> (o -> not ForAll(o.children(), IS(objid)));

NOT_ALL_ARGS_ARE := (objid) -> (o -> not ForAll(o.args, IS(objid)));

countRealAdds := c -> Sum(Collect(c, @(1, [add,sub], x->IsRealT(x.t))), op -> Length(op.args)-1); 
countRealMuls := c -> Sum(Collect(c, @(1, mul, x->IsRealT(x.t))), op -> Length(op.args)-1);

countVectMuls := c -> Sum(Collect(c, @(1, mul, x->IsVecT(x.t))), op -> Length(op.args)-1); 
countVectAdds := c -> Sum(Collect(c, @(1, [add,sub], x->IsVecT(x.t))), op -> Length(op.args)-1); 
