
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# NOTE: the code below checks if BF_NO_COPY flag is set. This is a bad way to check 
#        whether something is a class, also hardcoding 128 will break sooner or later
_isclass := x->BinAnd(BagFlags(x),32768)=32768 or not IsRec(x);

_oid := x -> Cond(_isclass(x), x, ObjId(x));

Class(RewritableObjectOps, PrintOps, rec(
    \= := (s1,s2) -> Cond(
        _isclass(s1) and _isclass(s2), BagAddr(s1)=BagAddr(s2), 
	_oid(s1) = _oid(s2) and s1.rChildren() = s2.rChildren()),

    \< := (s1,s2) -> Cond(
	_isclass(s1) and _isclass(s2), BagAddr(s1) < BagAddr(s2), 
        _oid(s1) <> _oid(s2),        _oid(s1) < _oid(s2),
        s1.rChildren() < s2.rChildren())
));

#F RewritableObject - convenient base class for objects to be used
#F                    with rewriting.
#F
#F It provides
#F      __call__
#F      print
#F      rSetChild
#F      rChildren
#F      lessThan
#F      equal
#F
#F Constructor __call__ takes variable number of arguments and saves all
#F of them into .params field of the constructed instance.
#F 
#F To do error checking and validation on .params, redefine .updateParams
#F in subclasses. It is also called after updates in .rSetChild
#F

Class(RewritableObject, rec(
    __call__ := meth(arg)
        local self, params, res;
        self := arg[1];
        params := Drop(arg, 1);
        res := WithBases(self, 
            rec(params := params, operations := RewritableObjectOps));
        res.updateParams();
        return res;
    end,
    
    updateParams := self >> self,

    equals := (self, o) >>
        ObjId(self) = ObjId(o) and self.rChildren() = o.rChildren(),

    lessThan := (self, o) >> Cond(
        ObjId(self) <> ObjId(o), ObjId(self) < ObjId(o), 
        [ ObjId(self), self.rChildren() ] < [ ObjId(o), o.rChildren() ]
    ),

    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), rch),
    rChildren := self >> self.params,
    rSetChild := meth(self, n, newC)
        self.params[n] := newC;
        self.updateParams();
    end,

    # Compatibility interface for Print, supports standard print() and
    # print(s, i, is).
    # NOTE: doesn't propagate it to children
    print := meth(arg)
        local self;
        self := arg[1];
        Print(self.__name__, "(", PrintCS(self.rChildren()), ")");
    end,
));

#F ConstClass - base class for constants.
#F
 
Class(ConstClass);

