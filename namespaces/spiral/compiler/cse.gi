
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# A variable is volatile if its value can change invisibly.
# For scalar variables this can never happen, because we keep track of all reads/writes, and also convert
# program to SSA on the fly.
#
# For arrays, however, this CAN happen, if an array is not scalarized -- because we do 
# not keep track of reads/writes, and we do not convert to array SSA.
#
# Knowing the volatility allows us to block CSE from eliminating loads to the same location,
# which would be unsafe.
#
# The error due to lack of volatility checking occurs in the simple case of unrolling the
# following piece of code, without scalarizing T:
#
# loop i
#     T[0] = X[i]
#     Y[i] = T[0]
# 
# unrolled:
#   T[0] = X[0]
#   Y[0] = T[0]
#
#   T[0] = X[1]
#   Y[1] = T[0]
#
#   T[0] = X[2]
#   Y[2] = T[0]
#
# CSE without considering T to be volatile:
# 
#   T[0] = X[0]
#   s1 = T[0]
#   Y[0] = s1
#
#   T[0] = X[1]
#   Y[1] = s1
#
#   T[0] = X[2]
#   Y[2] = s1
#
#
var.isVolatile := self >> IsArrayT(self.t) and (self in Compile.doNotScalarize);

IsVolatileExp := x -> Cond(
    IsRec(x) and IsBound(x.isVolatile), x.isVolatile(),
    IsRec(x) and IsBound(x.rChildren),  
        ForAny(x.rChildren(), IsVolatileExp),
    false
);

Class(CSE, rec(
    _op := (oid, args) -> oid.__name__ 
        :: Cond(IsBound(args[1]) and IsRec(args[1]) and IsBound(args[1].id), args[1].id, "")
        :: Cond(IsBound(args[2]) and IsRec(args[2]) and IsBound(args[2].id), args[2].id, ""),

    # returns a pair [lst, index]
    _lookup := meth(self, exp)
        local op, pos, args, lst;
	if not IsRec(exp) or not IsBound(exp.rChildren) or IsParam(exp) then 
	    return false; 
	fi;
	args := exp.rChildren();
	op   := self._op(ObjId(exp), args); 
        if not IsBound(self.csetab.(op)) then 
	    return false;
	else 
	    lst := self.csetab.(op);
	    pos := PositionProperty(lst, ent -> ent.args=args);
	    return When(pos=false, false, [lst, pos]);
	fi;
    end,

    _add := meth(self, loc, oid, args)
        local op, ent;
	op := self._op(oid, args); 
	ent := rec(loc := loc, args := args);
        if not IsBound(self.csetab.(op)) then 
            self.csetab.(op) := [ent];
	else 
            Add(self.csetab.(op), ent);
	fi;
    end,

    cseLookup := meth(self, exp) 
        local lkup, lst, idx;
	lkup := self._lookup(exp);
	if lkup = false then return false;
	else 
	    [lst, idx] := lkup;
	    return lst[idx].loc;
	fi;
    end,

    cseInvalidate := meth(self, exp)
        local lkup, lst, idx;
	lkup := self._lookup(exp);
	if lkup = false then return false;
	else 
	    [lst, idx] := lkup;
	    Unbind(lst[idx]); # this introduces holes to the list, which gap allows
	fi;
    end,

    cseAdd := meth(self, loc, exp)
        local ent, v, co, args, a, b;
	if not IsBound(exp.isExpComposite) then return; fi;
	args := exp.rChildren(); 
	if ForAny(args, IsVolatileExp) then return; fi;

	self._add(loc, ObjId(exp), args);

	if ObjId(exp) = mul and exp.t.isSigned() and Length(exp.args)=2 and IsValue(exp.args[1]) then
	    v := Copy(exp.args[1]);
	    v.v := -v.v; # negate
	    self._add(neg(loc), mul, [v, exp.args[2]]);

        elif ObjId(exp) in [add,sub] and Length(exp.args)=2 and not IsValue(exp.args[1]) and not IsValue(exp.args[2]) then
            ent := loc;
            [a, b] := exp.args;

            if ObjId(exp)=sub then
                co := self.cseLookup(add(a, b));
                if co=false then
                    co  := self.cseLookup(add(b, a));
                fi;
                [ent,co] := [co, ent];
            else
                co := self.cseLookup(sub(a, b));
                if co=false then
                   co := self.cseLookup(sub(b, a));
                   [b, a] := exp.args;
                fi;
            fi;

            if ent<>false and co<>false then 
		self._add( 2 * a, add, [ent, co]);
		self._add( 2 * a, add, [co, ent]);
		self._add( 2 * b, sub, [ent, co]);
		self._add(-2 * b, sub, [co, ent]);
            fi;

        fi;
    end,

    cseCmd := meth(self, cmd)
        local lkup;
	lkup := self.cseLookup(cmd.exp); 
	if lkup <> false then
	    cmd.exp := lkup;
	else
	    self.cseAdd(cmd.loc, cmd.exp);
	fi;
    end,

    init :=  self >> WithBases(self, rec(csetab := tab())),

    __call__ := meth(self, code)
        self.csetab := tab();
	DoForAll( Collect(code, assign), 
	          cmd -> self.cseCmd(cmd));
	return code;
    end
));
