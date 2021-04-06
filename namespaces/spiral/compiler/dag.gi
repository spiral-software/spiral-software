
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ArgsExp(<exp>)
#F   Collects all location leaves in an expression.
#F   This is equivalent to Collect(exp, @.cond(IsLoc)), but has less overhead.
#F
ArgsExp := exp -> Cond(  
    not IsRec(exp) or not IsSymbolic(exp), [],
    IsLoc(exp), Concatenation([exp], ConcatList(exp.rChildren(), ArgsExp)), 
    IsValue(exp) or not IsBound(exp.args), [],
    Concatenation(List(exp.args, ArgsExp)));

assign.op_in  := self >> ConcatList(self.loc.rChildren(), ArgsExp) :: ArgsExp(self.exp);
assign.op_out := self >> [self.loc];
assign.op_inout := self >> [];

IsCmdOp := x -> IsCommand(x) and IsBound(x.op_in) and IsBound(x.op_out) and IsBound(x.op_inout);

VarArgsExp := exp -> Cond(  
    IsVar(exp), [exp], 
    IsValue(exp) or not IsRec(exp), [],
    ConcatList(exp.rChildren(), VarArgsExp));

#
# Def/Succ/Pred
#

MarkDefLoc := function(loc, d)
    loc.def := d;
    if not IsBound(loc.succ) then loc.succ := Set([]); fi;
end;

MarkPredLoc := function(loc, pred)
    pred := Set(ShallowCopy(pred));
    if not IsBound(loc.pred) then loc.pred := pred;
    # NOTE: below statement crashes with "<loc.pred> is not a set" error 
    #        if UniteSet(loc.pred, pred) is used. This is an indication of 
    #        broken < or = operations. Union() fixes the problem, but is
    #        essentially a hack
    else loc.pred := Union(loc.pred, pred); 
    fi;

    if IsBound(loc.def) then
	if not IsBound(loc.def.pred) then loc.def.pred := pred;
	else loc.def.pred := Union(loc.def.pred, pred);
	fi;
    fi;
end;

# succesors is NOT a set to allow multiplicity (2 uses from same statement, like a = b*b, succ(b)=[a, a])
# this aids compiler from preventing propagation of b in this case in CopyPropagate.propagate()
#
MarkSuccLoc := function(loc, cmd_loc)
    if IsVar(loc) then
        if not IsBound(loc.succ) then loc.succ := [cmd_loc];
        else                          Add(loc.succ, cmd_loc); fi;

        if IsBound(loc.def) then
            if not IsBound(loc.def.succ) then loc.def.succ := [cmd_loc];
            else Add(loc.def.succ, cmd_loc);
            fi;
        fi;
    fi;
end;

DefLoc   := loc -> When(IsBound(loc.def),  loc.def,  false);
PredLoc  := loc -> When(IsBound(loc.pred), loc.pred, Set([]));
SuccLoc  := loc -> When(IsBound(loc.succ), loc.succ, ([]));

PredCmd  := cmd -> When(IsBound(cmd.pred), cmd.pred, Set([]));
SuccCmd  := cmd -> When(IsBound(cmd.succ), cmd.succ, ([]));

Declare(MarkDefUse, MarkPreds, ClearDefUse);

_ChainMarkDefUse := function(code)
    local c, u, pred;
    for c in code.cmds do
        if IsCmdOp(c) then 
	    DoForAll(c.op_out(), x->MarkDefLoc(x, c));
	    pred := c.op_in() :: c.op_inout();
	    DoForAll(c.op_out(), x->MarkPredLoc(x, pred));
	    for u in pred do
	        DoForAll(c.op_out(), x->MarkSuccLoc(u, x));
	    od;
	else
	    DoForAll(c.rChildren(), MarkDefUse);
	fi;
    od;
    return code;
end;

_MarkDefUse := function(code)
    if not IsCommand(code) then return;
    elif IsChain(code) then _ChainMarkDefUse(code);
    else DoForAll(code.rChildren(), _MarkDefUse);
    fi;
    return code;
end;

MarkDefUse := function(code)
    code := ClearDefUse(code);
    return _MarkDefUse(code);
end;

_ChainMarkPreds := function(code)
    local c, u, pred;
    for c in code.cmds do
        if IsAssign(c) then 
	    pred := c.op_in() :: c.op_inout();
	    MarkPredLoc(c.loc, pred);
	else
	    DoForAll(c.rChildren(), MarkPreds);
	fi;
    od;
    return code;
end;

_MarkPreds := function(code)
    if not IsCommand(code) then return;
    elif IsChain(code) then _ChainMarkPreds(code);
    else DoForAll(code.rChildren(), _MarkPreds);
    fi;
    return code;
end;

MarkPreds := function(code)
    code := ClearDefUse(code);
    return _MarkPreds(code);
end;


ClearDefUseLoc := function(loc)
   if IsBound(loc.def) then
       Unbind(loc.def.pred);
       Unbind(loc.def.succ);
   fi;
   Unbind(loc.def);
   Unbind(loc.succ);
   Unbind(loc.pred);
   return loc;
end;

depthLoc := x -> When(IsBound(x.depth), x.depth, 0);
rdepthLoc := (x, max) -> When(IsBound(x.rdepth), x.rdepth, max);

ComputeDepthsChain := function(code)
    local cmd, depth, rdepth, maxdepth, succs, dd, preds, succs;
    Constraint(ObjId(code)=chain);
    for cmd in code.cmds do
        Constraint(IsAssign(cmd));
	preds := PredCmd(cmd);
	dd := List(preds, depthLoc);
        depth := When(dd=[], 0, 1 + Maximum(dd));
	cmd.allpreds := Union(Filtered(preds,x->IsVar(x) and IsBound(x.def)),
	    Union(List(preds, x->When(IsVar(x) and IsBound(x.def) and 
			              IsBound(x.def.allpreds), x.def.allpreds, []))));
        cmd.earliest := depth; 
	cmd.loc.depth := depth;
    od;
    maxdepth := Length(code.cmds);
    for cmd in Reversed(code.cmds) do
	succs := SuccCmd(cmd);
        dd := List(succs, s -> rdepthLoc(s, maxdepth));
        rdepth := When(dd=[], maxdepth,  -1 + Minimum(dd));
	cmd.allsuccs := Union(succs, 
	    Union(List(succs, x->When(IsBound(x.def) and 
			              IsBound(x.def.allsuccs), x.def.allsuccs, []))));
	
        cmd.latest := rdepth; 
	cmd.loc.rdepth := rdepth;
    
    od;
end;

#F ClearDefUse(<code)> - clears attributes set by MarkDefUse
#F
ClearDefUse := code -> Chain(DoForAll(Collect(code, var), ClearDefUseLoc), code);


DFSChain := function(code)
    local c, cmds, W, next, p, added, schedule;
    cmds := code.cmds; W := []; schedule := [];
    for c in cmds do
       if not IsBound(c.loc.generated) then
       Add(schedule, c); c.loc.generated := true;
       if IsBound(c.loc.succ) then 
           Append(W, c.loc.succ);
       fi;
       fi;
       while Length(W)<>0 do
           next := Last(W);
       added := false;
       if IsBound(next.pred) then 
           for p in next.pred do 
               if not IsBound(p.generated) then Add(W, p); added := true; fi;
           od;
       fi;
       if not added then 
           if IsBound(next.def) then Add(schedule, next.def); fi;
           next.generated := true;
           RemoveLast(W, 1);
       fi;
       od;
    od;

    for c in cmds do Unbind(c.loc.generated); od;
    return chain(schedule);
end;

InputsChain := code -> List(Filtered(code.cmds, x->not ForAny(x.loc.pred,p->IsBound(p.def))),x->x.loc);
OutputsChain := code -> List(Filtered(code.cmds, x->not IsBound(x.loc.succ) or x.loc.succ=[]), x->x.loc);

_ClearRedBlue := function(v) Unbind(v.blue); Unbind(v.red); end;

ClearRedBlue := code -> DoForAll(Collect(code, var), _ClearRedBlue);

_HSplitChain := function(c,inp,out)
    local i,o,x,inpsucc,outpred, mid,done,red,blue,rcmds,bcmds;
    mid := Set([]);
    blue := Set(inp); bcmds:=List(blue, x->x.def);
    red := Set(out); rcmds:=List(red, x->x.def);

    done := (inp=[]) and (out=[]);
    while not done do
        inp := Difference(Union(List(inp, x->x.succ)),blue); # unvisited successors
    out := Difference(Union(List(out, x->x.pred)),red); # unvisited predecessors

    for x in inp do 
        AddSet(blue, x);  
        if not (x in red) then Add(bcmds,x.def); fi; 
    od;

    for x in out do
        AddSet(red, x); 
        if not (x in blue) then Add(rcmds,x.def); fi;
    od;

    inp := Difference(inp, red);
    out := Difference(out, blue);

    done := (inp=[]) and (out=[]);
    od;
    return [Intersection(red,blue), bcmds, Reversed(rcmds)];
end;

HSplitChain := function(c)
    local inp, out, mid;
    inp := Set(InputsChain(c));
    out := Set(OutputsChain(c));
    mid := _HSplitChain(c, inp, out);
    return mid;
end;

scheduled := code -> Inherit(code, rec(cmd := DFSChain(code.cmd)));
