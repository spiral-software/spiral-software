
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# determine ldepth by object type
_ldepth := o -> let(
    x := When(ObjId(o)=tcast, o.args[2], o), 

    # cond functions like a switch statement.
    Cond(
        IsBound(x.ldepth),
            x.ldepth,
        not IsVar(x), 
            2^20,
        let(
            f := DefFrontier(x, 5),
            l := Length(f), 
            last := Last(f),

            Cond(
                l = 1, 
                    0,
                Length(last)=1 and IsLoopIndex(last[1]), 
                    last[1].ldepth,
                2^20
            )
        )
    )
);
    
# added a high value for constants so that they sort at one end
# when the variables are sorted by ldepth
_ldepth2 := (o, params_first) -> let(
    x := When(ObjId(o)=tcast, o.args[2], o), 

    # cond functions like a switch statement.
    Cond(
        IsBound(x.ldepth),
            x.ldepth,
        IsValue(x),
            (2^20)+1,
        ObjId(x) in [param, in_param], 100,
#            (2^20) + When(params_first, -2-(InternalHash(x.id) mod 1024), 
#                                        +2+(InternalHash(x.id) mod 1024)),

        not IsVar(x), 
            2^20,
        let(
            f := DefFrontier(x, 5),
            l := Length(f), 
            last := Last(f),

            Cond(
                l = 1, 
                    0,
                Length(last)=1 and IsLoopIndex(last[1]), 
                    last[1].ldepth,
                2^21
            )
        )
    )
);

Class(BinSplit, rec(
    
    canSplit := (self, exp) >> ((IsExpCommand(exp) and not IsBound(exp.isExpComposite)) or 
        (IsBound(exp.isExpComposite) and exp.isExpComposite 
         and ForAny(exp.rChildren(), e->IsBound(e.isExpComposite)))) 
         and not (exp _is self._splitDisabled),
                       
    expSplit := meth(self, exp)
        local c, v, composites, res, ch, args, i, prev, newexp;
        res := [];
        if self.canSplit(exp) then
            ch := exp.rChildren();
            composites := SplitBy([1..Length(ch)], i->IsBound(ch[i].isExpComposite))[1];
            for c in composites do
                v := var.fresh_t("a", ch[c].t);
                Append(res, self.assignSplit(assign(v, ch[c])));
                exp.rSetChild(c, v);
            od;
        fi;

        # check for non-binary add/mult
	# NOTE: use a list of assoc expr? use Exp property!!
        if (ObjId(exp) in [add, mul, xor, bin_xor, bin_and]) and Length(exp.args) > 2 then
            args := ShallowCopy(exp.args);
            for v in args do v.ldepth := _ldepth2(v, ObjId(exp)=mul); od;
            Sort(args, (a,b) -> a.ldepth < b.ldepth);

            prev := args[1];
            for i in [2..Length(args)-1] do
	        newexp := ObjId(exp)(prev, args[i]);
                v := var.fresh_t("b", newexp.t);
                Add(res, assign(v, newexp));
                prev := v;
            od;
            exp.args := [prev, Last(args)];
        fi;
	return [res, exp];
    end,

    assignSplit := meth(self, cmd)
        local loc, exp, cmds_loc, cmds_exp;
        [loc, exp] := [cmd.loc, cmd.exp];
	[cmds_exp, exp] := self.expSplit(exp);
	[cmds_loc, loc] := self.expSplit(loc);

        if ObjId(exp) in [div,mul,add,xor, stickyNeg] then
            # sets kdepth correctly for stuff like a+(b+c)
            loc.ldepth:=Minimum(List(exp.args,x->_ldepth2(x, false)));
            # Sets ldepth correctly for stuff like a2= 2*i3 => a2.ldepth=_ldepth(i3)
            if (loc.ldepth>=2^18) then 
                loc.ldepth:=Minimum0(Filtered(List(exp.args,x->_ldepth2(x, false)), x->x<2^18));
            fi;
        fi;
        return cmds_exp :: cmds_loc :: [cmd];
    end,

    expCommandSplit := (self, cmd) >> self.expSplit(cmd),
#    expCommandSplit := meth(self, cmd)
#        local a, args, res, cmds; 
#	res := [];
#	for a in cmd.args do
#	    [cmds, a] := self.expSplit(a);
#	    Append(res, cmds);
#	od;
#        return res :: [cmd];
#    end,

    # BinSplit(code[, opts])

    __call__ := meth(arg)
        local self, code, opts;
        [self, code, opts] := [arg[1], arg[2], Cond(Length(arg)>2, arg[3], rec())];
        if IsBound(opts.binsplitOnIndices) and not opts.binsplitOnIndices then
            self := CopyFields(self, rec( _splitDisabled := opts.simpIndicesInside ));
        else
            self := CopyFields(self, rec( _splitDisabled := [] ));
        fi;
        return self.apply(code);
    end,

    apply := meth(self, code)
        local chains, blk, i, cmd, aux, exp, split;
        chains := Collect(code, chain);
	for blk in chains do
	    for i in [1..Length(blk.cmds)] do
	        cmd := blk.cmds[i];
	        if IsAssign(cmd) then 
                    # cmd could have parts aliased with different command
                    # since we do inplace modifications of cmd inside assignSplit, we copy it
                    split := self.assignSplit(Copy(cmd));
                    blk.cmds[i] := When(Length(split)=1, split[1], chain(split));
		elif IsExpCommand(cmd) then
                    split := self.expCommandSplit(Copy(cmd));
                    blk.cmds[i] := When(Length(split)=1, split[1], chain(split));
                fi;
            od;
        od;
        return FlattenCode(code);
    end,
));

