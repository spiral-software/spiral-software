
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

NewRulesFor(TRCDiag, rec(
    TRCDiag_VRCLR := rec(
        applicable := (self, t) >> t.hasTags() and ObjId(t.firstTag()) = AVecReg,
        apply := (self, t, C, Nonterms) >> let(isa := t.firstTag().isa, v := isa.v, n := Rows(t), 
            FormatPrm(fTensor(fId(n/v), L(2*v, v))) * 
            VRCLR(Diag(t.params[1]), v) * 
            FormatPrm(fTensor(fId(n/v), L(2*v, 2)))
        )
    )
));
