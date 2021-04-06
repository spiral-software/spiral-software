
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(search);

#   These functions merge multiple hashtables into a single one
AllHashEntries := function(table)
    local e, i, res;
    res := [];
    for e in Filtered(table.entries, IsList) do
        for i in e do
            if i.data <> [] then Add(res, i); fi;
        od;
    od;
    return res;
end;


AddMergedHashTables := function(target, newhashs)
    local e, v, h, i;
    for e in Flat(List(newhashs, i->AllHashEntries(i))) do
        v := [];
        for h in newhashs do
            Add(v, HashLookup(h, e.key));
        od;
        v := Filtered(v, i -> i <> [] and i <> false);
        if v <> [] then
            Sort(v, (j,k) -> j[1].measured < k[1].measured);
            HashAdd(target, e.key, v[1]);
        fi;
    od;
end;

#   Build all base cases for a given SIMD ISA.
#   Currently only L(2v, 2), L(2v, v), L(v^2, v)
#
SIMD_ISA_DB.buildBases := meth(self, isa)
    local rebind, t, v, tags, cxtags, t1, t2, rt, brules, common, rset1, rset2, tab1, tab2, e, h;

    if self.verbose then Print("Building bases for ", isa, "\n"); fi;
    self.hash_rebuilt := true;
    v := isa.v;
    tags := isa.getTags();
    cxtags := isa.getTagsCx();

    # == do TL base cases======================================
    # TL usually is not measured in DP
    rebind := IsBound(TL.doNotMeasure) and TL.doNotMeasure;
    if rebind then 
        TL.doNotMeasure := false; 
    fi;

    brules := paradigms.tSPL_Globals.getDPOpts().breakdownRules;
    common := [ SIMD_ISA_Bases1, SIMD_ISA_Bases2, IxLxI_kmn_n, IxLxI_kmn_km, paradigms.vector.breakdown.IxLxI_vtensor ];

    rset1 := CopyFields(brules, rec(TL := Concat(common, [IxLxI_IxLxI_up])));
    rset2 := CopyFields(brules, rec(TL := Concat(common, [IxLxI_IxLxI_down])));

    tab1 := HashTableDP();
    for t in SIMD_ISA_DB.getBases(isa) do
#    [ TL(2*v,v,1,1).withTags(tags), TL(2*v,2,1,1).withTags(tags), TL(v*v,v,1,1).withTags(tags), TL(v*v/4,v/2,1,2).withTags(tags) ]
        t1 := DP(t, rec(measureFunction := VCost, verbosity := 0, hashTable := tab1, globalUnrolling := true),
                CopyFields(isa.splopts, rec(breakdownRules := rset1, dataType := "no default", baseHashes := [], globalUnrolling := 10000)));
    od;

    tab2 := HashTableDP();
    for t in SIMD_ISA_DB.getBases(isa) do
#    [ TL(2*v,v,1,1).withTags(tags), TL(2*v,2,1,1).withTags(tags), TL(v*v,v,1,1).withTags(tags), TL(v*v/4,v/2,1,2).withTags(tags) ]
        t2 := DP(t, rec(measureFunction := VCost, verbosity := 0, hashTable := tab2, globalUnrolling := false),
                CopyFields(isa.splopts, rec(breakdownRules := rset2, dataType := "no default", baseHashes := [], globalUnrolling := 10000)));
    od;
    AddMergedHashTables(self.hash, [tab1, tab2]);
    # restore TL
    if rebind then
        TL.doNotMeasure:=true;
    fi;

    # == do other base cases===================================
    # do other base cases
    #
    # none here yet :(
    #

    # == final check - did all work? ==========================
    for e in self.getBases(isa) do
        h := HashLookup(self.hash, e);
        if  h=false or h=[] then Print(e, " could not be built\n"); fi;
    od;
end;
