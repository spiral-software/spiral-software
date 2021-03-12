
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details




#
# example:
#
#   d := DataInd(T_Int(8), [0..1]);
#   SMAP(Lambda([d], (2.0 * imod(d, 2) - 1.0)))
#


Class(SMAP, SumsBase, Sym, rec(
    abbrevs := [
        (lambda)  -> Checked(IsLambda(lambda) and Length(Set(List(lambda.vars, e->e.t)))=1, [lambda])
    ],
    def := (lambda) -> Gath(fBase(Length(lambda.vars), 0)),
    dmn := self >> [ TArray(self.params[1].vars[1].t, Length(self.params[1].vars)) ],
    rng := self >> [ TArray(self.params[1].range(),   1) ],
    at  := (arg) >> arg[1].params[1].at(StripList(Drop(arg, 1))),
));
