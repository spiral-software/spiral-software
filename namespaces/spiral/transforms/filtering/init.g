
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(approx, code, spl, formgen);
Import(paradigms.common); # tags
Import(paradigms.vector, paradigms.vector.sigmaspl);
Import(paradigms.smp);

Declare(Circulant, Toeplitz);
Class(DataNonTerminalMixin, rec(
    setData := meth(self, new_data)
        self.params[1] := new_data; 
        return self;
    end
));

Include(extensions);
Include(auxil);
Include(lifting);
Include(wavelet_lib);
Include(filt); # NOTE: requires paradigms.vector :(
Include(toeplitz);
Include(circulant); # NOTE: toFilt rule commented out
Include(ds_circulant);
Include(dwt_per);
Include(dwt); # NOTE: depends on filt

#Filt.print := (self,i,is) >> Print("Filt", When(IsBound(self.params), Print("(",
#    PrintCS([self.params[1], self.params[2].domain(), self.params[3], self.params[4]]), ")")), 
#    When(self.hasTags(), Print(".withTags(", self.getTags(), ")"), ""));

#Toeplitz.print := (self,i,is) >> When(IsBound(self.params), self.hprint(), 
#    Print(self.name));

#Circulant.print := (self,i,is) >> Print("Circulant", When(IsBound(self.params), Print("(",
#    PrintCS([self.params[1], self.params[2].domain(), self.params[3]]), ")")));

#DSCirculant.print := (self,i,is) >> Print("DSCirculant", When(IsBound(self.params), Print("(", 
#    PrintCS([self.params[1], self.params[2].domain(), self.params[3], self.params[4], self.params[5]]), ")")));

# execute Import(FiltRuleShortcuts) to be able to use these
FiltRuleShortcuts := tab(
#   ova    := Filt_OverlapAdd,
#   ovs    := Filt_OverlapSave,
#   karat  := Filt_KaratsubaSimple,
    cblock := Circulant_Blocking,
    tb     := Toeplitz_Blocking,
    tbd    := Toeplitz_BlockingDense,
    tbase  := Toeplitz_Base,
    tsbase := Toeplitz_SmartBase
);

