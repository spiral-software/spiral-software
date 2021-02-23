
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Higher-order functions support
#

# Functional expression base class
Class(FuncExp, Exp, rec(
    isFuncExp := true,

    at := arg >> let(
	self := arg[1],
	Cond(Length(arg)=2 and IsList(arg[2]), 
	     ApplyFunc(fcall, [self] :: arg[2]),
	     ApplyFunc(fcall, arg))),

    mkVars := self >> 
        List(DropLast(self.computeType().params, 1), x->var.fresh_t("q", x)),

    getRankType := (self, r) >> let(
	t := self.computeType(),
	rank := Length(t.params)-2,
	Cond(r > rank,
	         TDummy, 
	     t.params[ Length(t.params) - 1 - r ])
    )
));

Function.mkVars      := FuncExp.mkVars;
Function.getRankType := FuncExp.getRankType;

Symbolic.mkVars      := FuncExp.mkVars;
Symbolic.getRankType := FuncExp.getRankType;

IsFuncExp := o -> IsRec(o) and 
    ((IsBound(o.isFuncExp) and o.isFuncExp) or (IsBound(o.t) and ObjId(o.t)=TFunc));

# Same as IsFuncExp(<o>) but returns false for param's with t = TFunc 
# (these are considered 'trivial')
IsNonTrivFuncExp := o -> IsRec(o) and (IsBound(o.isFuncExp) and o.isFuncExp);


# !!! Below is no longer needed/used, autolib handles this via Parametrizer
# These are hacks for dealing with types which are integers like 2
# these really mean the range of an integer variable
# _convType converts 2 to TInt not to confuse other parts of the system
#

# !!! Below is no longer needed/used, autolib handles this via Parametrizer
#_convType := t -> Cond(IsSymbolic(t) or IsInt(t) or IsValue(t), TInt,                        
#    SubstBottomUp(Copy(t), TFunc, 
#        e -> ApplyFunc(TFunc, List(e.params, p -> When(IsInt(p) or IsSymbolic(p) or IsValue(p), TInt, p)))));
_convType := t->t; 

# !!! Below is no longer needed/used, autolib handles this via Parametrizer
#_convTypeExp := t -> 
#    SubstBottomUp(Copy(t), TFunc, 
#        e -> ApplyFunc(TFunc, List(e.params, p -> When(IsInt(p) or IsSymbolic(p) or IsValue(p), TInt, p))));
_convTypeExp := t->t;


#F fcurry(<func>, <pos>, <arg>) - symbolic representation of function currying
#F     Given a function with n arguments returns a functino with n-1 arguments
#F     where <arg> is plugged into position <pos> 
#F
#F Example: p := param(TFunc(TInt, 4, TComplex), "p");  # ie. rank-1 4-elt diagonal func
#F          i1 := Ind(4);
#F          fcurry(p, 1, i1).t; 
#F            => TFunc(4, TComplex);
#F
Class(fcurry, FuncExp, rec(
    computeType := self >> let(
        n := self.args[2],
        ct := self.args[1].computeType(), 
        Checked(IsValue(n), ObjId(ct)=TFunc, n.v < Length(ct.params),
            ApplyFunc(TFunc, ListWithout(ct.params, n.v)))),

    eval := self >> let(
        torig := self.args[1].t.params, 
        tcurried := self.t.params,
        n := self.args[2].v,
        vars := List([1..Length(tcurried)-1], i -> var.fresh_t("q", tcurried[i])),
        plugin := List([1..Length(torig)-1], i -> Cond(i<n, vars[i], i=n, self.args[3],vars[i-1])),
        Lambda(vars, ApplyFunc(fcall, Concatenation([self.args[1].eval()], plugin)))),

    at := (self, i) >> self.eval().at(i)
));

#F flift(<func>, <t>) - symbolic representation of function "lifting"
#F
#F  Given a function with n arguments returns a function with n+1 arguments,
#F  extra argument is of type <t> and is the "one but last" argument of new function.
#F  It is the ignored when computing the value of the function.
#F
#F  The purpose of flift is to increase the rank of functions. It implicitly creates
#F  a new inner loop, which loop variable is ignored. This is needed for rewrite rules
#F  like GT(T, ..) * Gath(f) -> GT(T*Gath(f), ...), where f is pulled into the loop.
#F
Class(flift, FuncExp, rec(
    computeType := self >> let(ct := self.args[1].computeType(), Checked(
        ObjId(ct)=TFunc, IsType(self.args[2]),
        let(tt := ct.params, n := Length(tt), 
        ApplyFunc(TFunc, Concatenation(tt{[1..n-2]}, [self.args[2]], tt{[n-1..n]}))))), 

    # NOTE: define what exactly .lambda() does in these cases
    lambda := self >> self.eval(),

    eval := self >> let(vars := List(DropLast(self.t.params,1), x->var.fresh_t("q", x)), n := Length(vars),
        Lambda(vars, ApplyFunc(fcall, Concatenation([self.args[1].eval()], ListWithout(vars, n-1)))))
));

#F fsplit(<func>, <loopid>, <inner_its>, <outer_its>)
Class(fsplit, FuncExp, rec(
    computeType := self >> let(ct := self.args[1].computeType(), Checked(
        ObjId(ct)=TFunc, IsValue(self.args[2]), self.args[2].t=TInt, 
        let(tt := ct.params, loopid := self.args[2].v, pos := Length(tt)-1-loopid,
            ApplyFunc(TFunc, Concatenation(tt{[1..pos-1]}, [tt[pos], tt[pos]], tt{[pos+1..Length(tt)]}))))),

    # NOTE: define what exactly .lambda() does in these cases
    lambda := self >> self.eval(),

    eval := self >> let(
        vars      := List(DropLast(self.computeType().params, 1), x->var.fresh_t("q", x)),
        loopid    := self.args[2].v,
        inner_its := self.args[3], 
        pos       := Length(vars)-loopid,
        callargs  := vars{[1..pos-2]} :: 
	             Cond(vars[pos].t=TDummy, [0], [inner_its * vars[pos-1] + vars[pos]]) ::
                     vars{[pos+1..Length(vars)]},
        Lambda(vars, ApplyFunc(fcall, Concatenation([self.args[1].eval()], callargs))))
));

#F frotate(<func>, <n>)
#F   switch the order of loops (=ranks), by making <n>-th loop innermost
Class(frotate, FuncExp, rec(
    computeType := self >> let(ct := self.args[1].computeType(), nn := self.args[2], Checked(
        ObjId(ct)=TFunc, IsValue(nn), nn.t=TInt, 
	let(tt := ct.params, rank := Length(tt)-2, n := nn.v, pos := rank+1-n, 
            Cond(rank=0 or (rank=1 and n=1), ct,
		 n > rank,  
		            ApplyFunc(TFunc, tt{[1..rank]} :: [TDummy] :: [tt[rank+1], tt[rank+2]]),
		 # else 
                            ApplyFunc(TFunc, tt{[1..pos-1]} :: tt{[pos+1..rank]} :: [tt[pos], tt[rank+1], tt[rank+2]]))))),

    # NOTE: define what exactly .lambda() does in these cases
    lambda := self >> self.eval(),

    eval := self >> let(f := self.args[1], rank := f.rank(), n := self.args[2].v, pos := rank+1-n, 
	Cond(rank=0 or (rank=1 and n=1), f,
	     n > rank, 
             let(vars := List(DropLast(self.computeType().params, 1), x->var.fresh_t("q", x)),
		 Lambda(vars, ApplyFunc(fcall, [f] :: vars{[Length(vars)-1-rank..Length(vars)-2]} :: [Last(vars)]))),
	     # else
             let(vars := List(DropLast(self.computeType().params, 1), x->var.fresh_t("q", x)),
		 Lambda(vars, ApplyFunc(fcall, [f] :: vars{[1..pos-1]} :: [vars[rank]] :: vars{[pos..rank-1]} :: [vars[rank+1]])))
	))
));


## Ranked functions support
## Ranked functions == functions with implicit dependencies on loop variables
##
## These are used in paradigms.common.GT and autolib.*
##
## NOTE: get rid of _, these functions are not private, but public exports
##
_rankManip := (obj, newfunc) >> 
    SubstTopDownNR(Copy(obj), @.cond(e->IsFunction(e) or IsFuncExp(e)), e -> newfunc(e));

_rank     := o -> Cond(
    IsList(o), Maximum0(List(o, _rank)),
    not IsRec(o) or not IsBound(o.rank), 0, o.rank());
_upRank   := o -> Cond(
    IsList(o), List(o, _upRank),
    not IsRec(o) or not IsBound(o.upRank), o, o.upRank());
_upRankNeq0   := o -> Cond(
    IsList(o), List(o, _upRankNeq0),
    not IsRec(o) or not IsBound(o.upRank) or o.rank()=0, o, o.upRank());
_upRankBy := (o,n) -> Cond(IsList(o), # NOTE: =0 ??
    List(o, x->_upRankBy(x,n)),
    not IsRec(o) or not IsBound(o.upRankBy) or o.rank()=0, o, o.upRankBy(n));
_downRank := (o,loopid,ind) -> Cond(
    IsList(o), List(o, x->_downRank(x, loopid, ind)),
    not IsRec(o) or not IsBound(o.downRank), o, o.downRank(loopid, ind));
_downRankFull := (o,inds) -> Cond(
    IsList(o), List(o, x->_downRankFull(x, inds)),
    not IsRec(o) or not IsBound(o.downRankFull), o, o.downRankFull(inds));
_split := (o, loopid, iits, oits) -> Cond(
    IsList(o), List(o, x->_split(x, loopid, iits, oits)), 
    not IsRec(o) or not IsBound(o.split), o, o.split(loopid, iits, oits));
_rotate := (o, n) -> Cond(
    IsList(o), List(o, x->_rotate(x, n)), 
    not IsRec(o) or not IsBound(o.rotate), o, o.rotate(n)); 


_rch_rank      := self >> Maximum0(List(self.rChildren(), _rank));
_rch_upRank    := self >> self.from_rChildren(List(self.rChildren(), _upRank));
_rch_upRankBy  := (self, n) >>  self.from_rChildren(List(self.rChildren(), c->_upRankBy(c,n)));
_rch_downRank  := (self, loopid, ind) >> 
    self.from_rChildren(List(self.rChildren(), c->_downRank(c,loopid,ind)));
_rch_downRankFull := (self, inds) >> 
    self.from_rChildren(List(self.rChildren(), c->_downRankFull(c,inds)));
_rch_split     := (self, loopid, iits, oits) >>  
    self.from_rChildren(List(self.rChildren(), c->_split(c,loopid, iits, oits)));
_rch_rotate    := (self, n) >> 
    self.from_rChildren(List(self.rChildren(), c->_rotate(c, n))); 


Symbolic.domain := self >> Checked(ObjId(self.t)=TFunc, Length(self.t.params) > 1, 
    self.computeType().params[Length(self.t.params)-1]);

Symbolic.range := self >> Checked(ObjId(self.t)=TFunc, Length(self.t.params) > 1,
    self.computeType().params[Length(self.t.params)]);
    
Symbolic.rank := self >> Cond(ObjId(self.t)=TFunc and Length(self.t.params) > 1,
    Length(self.t.params) - 2, 
    _rch_rank(self));

Symbolic.at := (self, vars) >> Checked(ObjId(self.t)=TFunc, self.lambda().at(vars));

Symbolic.lambda := self >> Checked(ObjId(self.t)=TFunc, let(
    selft := self.computeType(), 
    vars := List([1..self.rank()+1], x -> let(t:=selft.params[x], 
            Cond(IsType(t), var.fresh_t("w", t), var.fresh("w", TInt, t)))),
    Lambda(vars, ApplyFunc(fcall, Concatenation([self], vars)))));

Symbolic.upRank := self >> Cond(ObjId(self.t)<>TFunc, _rch_upRank(self), flift(self, TDummy));

Symbolic.upRankBy := (self, n) >> Cond(
    ObjId(self.t)<>TFunc, _rch_upRankBy(self,n), 
    Checked(IsPosInt0(n), FoldL([1..n], (f, i) -> f.upRank(), self)));

Symbolic.downRankFull := (self, inds) >> Cond(
    ObjId(self.t)<>TFunc, _rch_downRankFull(self, inds), 
    FoldL(Reversed([1..Minimum(self.rank(), Length(inds))]), (f, i) -> f.downRank(i, inds[i]), self));

Symbolic.downRank := (self, loopid, ind) >> let(rank := self.rank(), Cond(
    loopid > rank, self, 
    ObjId(self.t)<>TFunc, _rch_downRank(self, loopid, ind),
    fcurry(self, rank+1-loopid, ind)));  # !! loop variables are ordered with decreasing rank, highest-rank (outermost) is first var

Symbolic.split := (self, loopid, inner_its, outer_its) >> Cond(
    loopid > self.rank(), self, 
    ObjId(self.t)<>TFunc, _rch_split(self, loopid, inner_its, outer_its),
    fsplit(self, loopid, inner_its, outer_its));

Symbolic.rotate := (self, n) >> Cond(
    self.rank() <= 1 and n <= 1, self, 
    ObjId(self.t)<>TFunc, _rch_rotate(self, n), 
    frotate(self, n)); 


# NOTE: below is really a hack, since the first argument might also have ranked
# things, although this is not obvious at first glance, an example of such first
# argument would be fcurry(func, lambdaWrap(rank-n function))

fcall.rank := self >> Cond(ObjId(self.t)=TFunc and Length(self.t.params) > 1,
    Length(self.t.params) - 2, 
    _rank(Drop(self.args, 1)));

fcall.upRank := self >> Cond(ObjId(self.t)<>TFunc, 
    ApplyFunc(fcall, [self.args[1]] :: _upRank(Drop(self.args, 1))), 
    flift(self, TDummy));

fcall.upRankBy := (self, n) >> Cond(ObjId(self.t)<>TFunc, 
    ApplyFunc(fcall, [self.args[1]] :: _upRankBy(Drop(self.args, 1), n)),
    Checked(IsPosInt0(n), FoldL([1..n], (f, i) -> f.upRank(), self)));

fcall.downRankFull := (self, inds) >> Cond(ObjId(self.t)<>TFunc, 
    ApplyFunc(fcall, [self.args[1]] :: _downRankFull(Drop(self.args, 1), inds)),
    FoldL(Reversed([1..Minimum(self.rank(), Length(inds))]), (f, i) -> f.downRank(i, inds[i]), self));

fcall.downRank := (self, loopid, ind) >> let(rank := self.rank(), Cond(
    loopid > rank, self, 
    ObjId(self.t) <> TFunc, 
        ApplyFunc(fcall, [self.args[1]] :: _downRank(Drop(self.args, 1), loopid, ind)), 
    # else
    fcurry(self, rank+1-loopid, ind)));  # !! loop variables are ordered with decreasing rank, highest-rank (outermost) is first var

fcall.split := (self, loopid, inner_its, outer_its) >> Cond(
    loopid > self.rank(), self, 
    ObjId(self.t) <> TFunc, 
        ApplyFunc(fcall, [self.args[1]] :: _split(Drop(self.args, 1), loopid, inner_its, outer_its)),
    fsplit(self, loopid, inner_its, outer_its));

fcall.rotate := (self, n) >> Cond(
    self.rank() <= 1 and n <= 1, self,
    ObjId(self.t) <> TFunc, 
        ApplyFunc(fcall, [self.args[1]] :: _rotate(Drop(self.args, 1), n)),
    frotate(self, n)); 


Function.rank      := _rch_rank;
Function.upRank    := _rch_upRank;
Function.upRankBy  := _rch_upRankBy;
Function.downRank  := _rch_downRank;
Function.downRankFull := _rch_downRankFull;
Function.split     := _rch_split;
Function.rotate    := _rch_rotate;
Function.computeType := self >> self.lambda().t;

Lambda.rank      := Symbolic.rank; 
Lambda.upRank    := Symbolic.upRank;
Lambda.upRankBy  := Symbolic.upRankBy;
#Lambda.downRank  := Symbolic.downRank;  downRank now defined in lambda.gi
Lambda.downRankFull := Symbolic.downRankFull;
Lambda.split     := Symbolic.split;
Lambda.rotate    := Symbolic.rotate;

# ind(<range>, <n>) - "nameless" reference to a loop index of n-th inner most loop
#            (eg. ind(1) is inner most, ind(n) is outermost in n-loop nest)
#            loop counter runs from 0..range-1
Class(ind, Loc, rec(
    __call__ := (self, range, n) >> Checked(IsInt(n),
	WithBases(self, rec(operations := ExpOps, range:=range, n:=n))),
    print := self >> Print(self.name, "(", self.range, ", ", self.n, ")"),
    rChildren := self >> [self.range, self.n],
    rSetChild := rSetChildFields("range", "n"),
    t := TInt,
    eval := self >> self,
    can_fold := False,
    
    upRank := self >> ObjId(self)(self.range, self.n+1),

    split := (self, loopid, inner_its, outer_its) >> 
        Cond( loopid > self.n, self, 
              loopid < self.n, ObjId(self)(self.range, self.n+1), 
              ObjId(self)(inner_its, self.n) + inner_its*ObjId(self)(outer_its, self.n+1)),
    rotate := (self, n) >> 
        Cond( n < self.n, self, 
              n > self.n, ObjId(self)(self.range, self.n+1), 
              ObjId(self)(self.range, 1)),

));


# NOTE: ind.downRank might be a hack
ind.downRankFull := (self, inds) >> inds[self.n];
ind.downRank := (self, loopid, ind) >> Cond(loopid=self.n, ind, self);
ind.rank := self >> self.n;


_hofnew := true;
# ExpMarkActiveRank(<s>)
#   performs a recursive walk over <s> and sets ._expMarkActiveRank attribute to the "active rank"
#   of each subexpression.
#
#   Active rank of an expression denotes the maximum implicit loop id that the expression refers to.
#   Implicit loop id's are introduced by objects such as GT and Lambda.
#
ExpMarkActiveRank := function(s)
    local c, rch, rank;
    rch := Cond(IsRec(s) and IsBound(s.rChildren), s.rChildren(), IsList(s) and not IsString(s), s, []);
    if ObjId(s) = ind then
        rank := s.n;
    else
	if _hofnew then
	    rank := _rank(s); 
	    DoForAll(rch, ExpMarkActiveRank);
	else
# this was invalid (fcurry, fcall, etc) -> 
	    rank := Maximum(_rank(s), Maximum0(List(rch, ExpMarkActiveRank))); 
	fi;
    fi;

    if IsRec(s) and IsSymbolic(s) then s._expMarkActiveRank := rank; fi;
    return rank;
end;

_ExpMarkPassiveRank := function(s, parent_rank) 
    local c, rch, rank, my_rank;
    rch := Cond(IsRec(s) and IsBound(s.rChildren), s.rChildren(), IsList(s) and not IsString(s), s, []);
    my_rank := _rank(s);

    # NOTE: this is a terrible hack! WHAT ABOUT Lambda?
    if spiral.paradigms.common.IsGT(s)        then rank := my_rank + parent_rank;
    elif ObjId(s)=ind then rank := Maximum(parent_rank, s.n);
    else                   rank := Maximum(parent_rank, my_rank);
    fi;
    if IsRec(s) and (IsSymbolic(s) or IsFuncExp(s)) then s._expMarkPassiveRank := rank; fi;

    DoForAll(rch, x -> _ExpMarkPassiveRank(x, rank)); 
    return rank;
end;
# ExpMarkPassiveRank(<s>)
#   performs a recursive walk over <s> and sets ._expMarkPassiveRank attribute to the "passive rank"
#   of each subexpression.
#
#   Passive rank of an expression denotes the maximum implicit loop id that is defined in the expression.
#   Regardless of whether expression refers to it or now. In contracst, an "active rank" is the loop id
#   that is actually referred to.
#
#   Implicit loop id's are introduced by objects such as GT and Lambda.
#
ExpMarkPassiveRank := s -> _ExpMarkPassiveRank(s, 0);


Class(lambdaWrap, Exp, rec(
    computeType := self >> Checked(ObjId(self.args[1].t) = TFunc, Last(self.args[1].t.params))
));
