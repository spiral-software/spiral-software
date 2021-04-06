
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################################
#   vector constructs
########################################################################

IsSIMD_ISA := s -> IsRec(s) and IsBound(s.isSIMD_ISA) and s.isSIMD_ISA;
IsISA      := x -> IsRec(x) and IsBound(x.isISA) and x.isISA;

Class(AVecReg, AGenericTag, rec(
    isReg := true,
    isRegCx := false,
    isVec := true,
    updateParams := meth(self)
        Checked(IsSIMD_ISA(self.params[1]));
        Checked(Length(self.params)=1);
        self.v := self.params[1].v;
        self.isa := self.params[1];
    end,
    container := (self, spl) >> paradigms.vector.sigmaspl.VContainer(spl, self.isa)
));


Class(AVecRegCx, AVecReg, rec(
    updateParams := meth(self)
        Checked(IsSIMD_ISA(self.params[1]));
        Checked(Length(self.params)=1);
        self.v := self.params[1].v/2;
        self.isa := self.params[1];
    end,
    container := (self, spl) >> paradigms.vector.sigmaspl.VContainer(spl, self.isa.cplx()),
    isRegCx := true
));

# AMultiVec - list of ISAs, must be list of AVecReg tags in the future
#

Class(AMultiVec, AGenericTag, rec(
    isVec := true,
    updateParams := meth(self)
        Checked(ForAll(self.params, IsSIMD_ISA));
        Checked(Length(self.params)>=1);
    end,
));

Class(AISA, AGenericTag, rec(
    updateParams := meth(self)
        Checked(IsISA(self.params[1]));
        Checked(Length(self.params)=1);
        self.isa := self.params[1];
    end,
    # it's not a vectorized code, maybe defferent kind of containers?
    # containers do not go along well with OL though
    container := (self, spl) >> paradigms.vector.sigmaspl.VContainer(spl, self.isa)
));

