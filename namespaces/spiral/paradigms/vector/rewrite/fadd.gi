
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(Rules_fAdd, RuleSet);
RewriteRules(Rules_fAdd, rec(
    fTensor := Rule([@(1, fTensor), @(2, fAdd, e->ForAll(e.params, i->IsInt(i) or IsValue(i))), @(3, fId, e-> IsInt(e.params[1]) or IsValue(e.params[1]))], 
        e-> let(fadd := @(2).val, fid := @(3).val.domain(), fAdd(fadd.params[1]*fid, fadd.params[2]*fid, fadd.params[3]*fid))),

    fTensorVScatVGath :=ARule(fTensor, [ @@(2, fAdd, (e,cx)->(IsBound(cx.VScat) and Length(cx.VScat)>=1) or (IsBound(cx.VGath) and Length(cx.VGath)>=1)), @(3, fId) ], 
        e-> let(fadd := @@(2).val, fid := @(3).val.domain(), [ fAdd(fadd.params[1]*fid, fadd.params[2]*fid, fadd.params[3]*fid) ] )),
        
    fAddFBase := ARule(fCompose, [@@(2, fAdd, (e,cx)->(IsBound(cx.VScat) and Length(cx.VScat)>=1) or (IsBound(cx.VGath) and Length(cx.VGath)>=1)), @(3, fBase)], 
        e-> let(fadd := @(2).val, fbase := @(3).val, [ fAdd(fadd.params[1], 1, fadd.params[3]+fbase.params[2]) ] ))
        
));
