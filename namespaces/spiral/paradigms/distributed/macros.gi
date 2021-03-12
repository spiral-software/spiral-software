
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Declare(compareMatrices);
Declare(applyCellRules);
Declare(doCellWht);

testfunc := function(m1)
   m1.isRight := true;
   return(m1);
end;

subtractDFTMatrices := function(m1, m2)
   local mdiff, verified;

    m1 := Cond(
            ObjId(m1) = ListClass,
            m1,
            IsBound(m1.isSPL) and m1.isSPL, 
            MatSPL(m1),
            Error("Don't know what m1 is"));
    m2 := Cond(
            ObjId(m2) = ListClass,
            m2,
            IsBound(m2.isSPL) and m2.isSPL, 
            MatSPL(m2),
            Error("Don't know what m2 is"));

return Maximum(Flat(MapMat(m1-m2, i->AbsComplex(ComplexAny(i)))));

end;



compareMatrices := function(m1, m2)
   local mdiff, verified;

    m1 := Cond(
            ObjId(m1) = ListClass,
            m1,
            IsBound(m1.isSPL) and m1.isSPL, 
            MatSPL(m1),
            Error("Don't know what m1 is"));
    m2 := Cond(
            ObjId(m2) = ListClass,
            m2,
            IsBound(m2.isSPL) and m2.isSPL, 
            MatSPL(m2),
            Error("Don't know what m2 is"));

    mdiff := m1-m2;


    #return(Cond(Minimum(Flat(mdiff)) = Maximum(Flat(mdiff)), true, false));
    verified := InfinityNormMat(mdiff);
    When(verified <= 0.001, Print(gap.colors.Green("Ok."), "\n"), gap.colors.Red("Matrices are DIFFERENT!\n"));
    return(verified);
end;

# n := 4; spus := 2; pkSize := 1;
doCellWht := function(n, spus, pkSize, opts)
    local t, r, sums, sumsorig, c, m, mdiff, measured, verified, compare;

    opts.spus := spus;

    t       := WHT(n).withTags([ ParCell(spus, pkSize) ]);
    r       := RandomRuleTree(t, opts);
    sums    := SumsRuleTreeOpts(r, opts);
    #sumsorig:= Copy(sums);
    #sums    := applyCellRules(sums, opts);
    #Print("Comparing sums after application of rules...");
    #compareMatrices(sums, sumsorig);
    if compareMatrices(sums, t) = false then
      return([r, sums, ""]);
    fi;

    c := CodeSumsOpts(sums, opts);
    measured := CMeasure(c, opts);
    Print(measured, " [ cycles ]\n");
    m := CMatrix(c, opts);
    Print("Comparing via CMatrix verification...");
    compareMatrices(m, t);

    return([r, sums, m]);

end;
