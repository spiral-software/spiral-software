
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

##########
T := e -> When(IsBound(e.t), e.t, "U");
Exp.print := self >> Print(T(self), ":", self.name, "(", PrintCS(self.args), ")");
nth.print := self >> Print(T(self), ":", self.name, "(", self.loc, ", ", self.idx, ")");
Value.print := self >> Print(self.t, ":", self.v);

ComputeExpType := function(e)
    local t;
    t := UnifyTypes(List(e.args, x->x.t));
    e.t := t;
    return e;
end;

PropagateTypes := c -> SubstBottomUp(c,
    @.cond(e->IsExp(e) and not IsBound(e.t)), 
    ComputeExpType);
