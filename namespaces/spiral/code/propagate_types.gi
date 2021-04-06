
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_computeExpType := function(e)
    local t;
    if IsBound(e.computeType) then t := e.computeType();
#    elif Length(e.args)=0 then t := TUnknown;
    else t := UnifyTypes(List(e.args, x->x.t));
    fi;
    e.t := t;
    return e;
end;

PropagateTypes := c -> SubstBottomUp(c,
    @.cond(e->IsExp(e) and not IsBound(e.t)), 
    _computeExpType);
