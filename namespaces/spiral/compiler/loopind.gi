
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Eventually, PowerOpt should grow into a general 
# loop induction variable optimization pass

ModMultOpt := code -> SubstBottomUp(code, 
    [mul, @(1,Value), [imod, @r, @N]], 
    e -> let(k := @(1).val.v, imod(k * @r.val, k * @N.val)));

PowerOpt := function(code)
    local powers, p, pvar, pmap;
    powers := Set(Collect(code, powmod));
    #loops := TabList( Flat(List(Collect(code, loop), l->[l.var.id, l]) ));
    #loopvars := NSFields(loops);
    for p in powers do
	if IsValue(p.args[2]) and IsVar(p.args[3]) and IsLoopIndex(p.args[3]) then
	    #Print("---", p, "---\n");
	    pvar := var.fresh("p", TInt, p.args[4].v);
	    code := SubstBottomUp(code, [powmod, 
		    @(1).cond(e->e=p.args[1]), 
		    @(2,Value,e->e.v=p.args[2].v), 
		    @(3,var,  e->Same(e,p.args[3])),
		    @(4,Value,e->e.v=p.args[4].v)], e->pvar);

	    code := SubstBottomUp(code, @(1,loop,e->Same(e.var,p.args[3])), 
		e -> decl([pvar], 
		    chain(
			assign(pvar, p.args[1]),
			loop(@(1).val.var, @(1).val.range, 
			    chain(@(1).val.cmd,
				assign(pvar, imod(mul(pvar, p.args[2]), p.args[4])))))));
	fi;
    od;
    return code;
end;

RangeProp := c->SubstBottomUp(c, [assign, @(1, var), @(2, var, e->IsBound(e.range))], 
    e -> assign(@(1).val.setRange(@(2).val.range), @(2).val));
