
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F FreeVars(<o>) 
#F   list of free (unbound) variables in an object
#F   Expects a .free() method in <o>.
FreeVars := o -> Cond(
    IsRec(o) and IsBound(o.free), o.free(), 
    IsList(o), Set(Union(List(o, FreeVars))),
    Set([])
);

