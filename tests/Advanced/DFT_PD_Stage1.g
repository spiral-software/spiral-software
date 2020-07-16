comment("");
comment("DFT PD Stage 1 Test");

##  if CheckBasicProfilerTest() ==> profiler tests passed

if not CheckBasicProfilerTest() then
    PrintLine("Basic Profiler test NOT PASSED, skipping test");
    if FileExists(".") then
        TestSkipExit();
    else
        TestFailExit();
    fi;
fi;

Import(realdft);
Import(filtering);
Import(paradigms.smp);

NewRulesFor(DFT, rec(

    DFT_PD_loop := rec(
        forTransposition := false,
        minSize := 7,
        maxSize := 13,
    
        TabPerm := r -> let(i:=Ind(r.domain()), f := FData(r.tolist()).at(i), Lambda(i, f).setRange(r.range())),
    
        applicable     := (self, nt) >> nt.params[1] > 2 and nt.params[1] in [self.minSize..self.maxSize] and IsPrime(nt.params[1]) and not nt.hasTags(),
    
        apply := (self, nt, C, cnt) >> let(
            N := nt.params[1], 
            k := nt.params[2], 
            root := PrimitiveRootMod(N),
            M:=(N-1)/2,
            i := Ind(M),
            j := Ind(M),
            k1 := Ind(M+1),
            m := MatSPL(DFT_PD.core(N, k, root, false) * DFT_PD.A(N)),
            m1 := Map(m{[2..(N+1)/2]}, r -> r{[2..(N+1)/2]}),
            m2 := Map(m{[(N+3)/2..N]}, r -> r{[(N+3)/2..N]}),
            d := Flat(m1)::Map(Flat(m2), c->im(ComplexAny(c)).v),
            fd := FData(d),
            
            s := Scat(self.TabPerm(RR(N, 1, root))),
            g := Gath(self.TabPerm(RR(N, 1, 1/root mod N))),

            gg := Gath(fCompose(fAdd(N, N-1, 1), fTensor(fId(2), fBase(M, k1-1)))),
            f2 := F(2),
            u := Ind(2),
            uf := Lambda(u, fd.at(u*M*M + M*i+(k1-1))),
            bb := DirectSum(Blk1(uf.at(0)), Scale(ImaginaryUnit(), Blk1(uf.at(1)))),
            
            krn0 := Mat([[1],[0]]) * Gath(fAdd(N, 1, 0)),
            krnn := bb * f2 * gg,
            krn := ScatAcc(fId(2)) * f2 * COND(eq(k1,0), krn0, krnn),
            
            q1 := L(2*M, 2) * IterVStack(i, ISumAcc(k1, krn)),
            #q2 := RowVec(fConst(13,V(1.0))),
            _i := Ind(N),
            q2 := ISumAcc(_i, ScatAcc(fAdd(1,1,0)) * Blk([[1]]) * COND(eq(_i, 0), Gath(fAdd(13,1,0)), Gath(fAdd(13,1,_i)))), 
            q3 := VStack(q2, q1),
            qq := s * q3 * g,            
            qq
        )
    )
));

RulesFor(PRDFT, rec(
   PRDFT_PD_loop := rec(
	forTransposition := false,
    minSize := 7,
	maxSize          := 13,
	isApplicable     := (self, P) >> P[1] > 2 and P[1] in [self.minSize..self.maxSize] and IsPrime(P[1]),
	
	rule := (self,P,C) >> let(N:=P[1], n:=N-1, k:=P[2], root:=PrimitiveRootMod(N),
        M:=n/2,
        i := Ind(M),
        j := Ind(M),
        k1 := Ind(M+1),
        m := MatSPL(DFT_PD.core(N, k, root, false) * DFT_PD.A(N)),
        m1 := Map(m{[2..(N+1)/2]}, r -> r{[2..(N+1)/2]}),
        m2 := Map(m{[(N+3)/2..N]}, r -> r{[(N+3)/2..N]}),
        d := Flat(m1)::Map(Flat(m2), c->im(ComplexAny(c)).v),
        fd := FData(d),
        #lfd1 := Lambda(j, fd.at(M*i+j)),
        #lfd1 := Lambda(k1, cond(eq(k1, V(0)), V(1.0), fd.at(M*i+(k1-V(1))))),
        #lfd2 := Lambda(j, fd.at(M*M+M*i+j)),
        
        gf := DFT_PD_loop.TabPerm(RR(N, 1, root)),
        g := Gath(gf),
        
        #kk1 := DirectSum(RowVec(lfd1), RowVec(lfd2)) * DirectSum(I(1), Tensor(F(2), I(M)) * OS(n, -1)),
        #q1 :=  IterVStack(i, BB(kk1)),
        
        #lbd := OS(12, -1).lambda(),
        _j := Ind(n),
        lbd := Lambda(_j, imod(V(n)-_j, V(n))),
        gg := Gath(fCompose(fAdd(N, N-1, 1), fCompose(lbd, fTensor(fId(2), fBase(M, k1-1))))),

        f2 := F(2),
        u := Ind(2),
        uf := Lambda(u, fd.at(u*M*M + M*i+(k1-1))),
        bb := DirectSum(Blk1(uf.at(0)), Blk1(uf.at(1))),

        krn0 := Mat([[1],[0]]) * Gath(fAdd(N, 1, 0)),
        krnn := bb * f2 * gg,
        krn := Grp(ScatAcc(fId(2))) * COND(eq(k1,0), krn0, krnn),

        q1 := IterVStack(i, ISumAcc(k1, krn)),
        q2a := RowVec(fConst(13,V(1.0))),
#        _i := Ind(N),
#        q2a := ISumAcc(_i, ScatAcc(fAdd(1,1,0)) * Blk([[1]]) * COND(eq(_i, 0), Gath(fAdd(13,1,0)), Gath(fAdd(13,1,_i)))), 
        q2b := RowVec(fConst(13,V(0.0))),
        q3 := VStack(BB(VStack(q2a, q2b)), q1),
        sf := DFT_PD_loop.TabPerm(Refl((N+1)/2, N, (N+1)/2, RR(N,1,root))),
        s := Scat(fTensor(sf, fId(2))),
        
        u1 := Ind(N+1),
        df := Lambda(u1, cond(logic_and(eq(V(1), bin_and(u1, 1)), geq(gf.at(idiv(u1, 2)), (N+1)/2)), V(-1), V(1))),
        dfl := df.tolist(),
        m1s := Filtered([0..N], _i->dfl[_i+1] = V(-1)),
        m1eq := List(m1s, _i->eq(u1, V(_i))),
        cnd := ApplyFunc(logic_or, m1eq),
        lbdcnd := Lambda(u1, cond(cnd, V(-1), V(1))),
        diag := Diag(lbdcnd),
        sct := s * diag,
        qq := sct * q3 * g,
        qq
    )
)));


PRDFT1_Base2.forTransposition := false;
IPRDFT1_Base2.forTransposition := false;
PRDFT2_Base2.forTransposition := false;
IPRDFT2_Base2.forTransposition := false;
PRDFT3_Base2.forTransposition := false;
IPRDFT3_Base2.forTransposition := false;
PRDFT1_PF.forTransposition := false;
PRDFT1_CT.forTransposition := false;
PRDFT1_CT.forTransposition := false;
PRDFT_PD.forTransposition := false;

PrunedPRDFT_base.maxSize := 5;
DFT_PD.maxSize := 7;
PRDFT_PD.maxSize := 7;

# rule set configuration
opts := SpiralDefaults;
opts.breakdownRules.PRDFT := [ PRDFT1_Base2, PRDFT_PD, PRDFT_PD_loop]; #PRDFT1_Base1, PRDFT1_CT, PRDFT1_PF, PRDFT_Rader
opts.breakdownRules.IPRDFT := [ IPRDFT1_Base1, IPRDFT1_Base2, IPRDFT1_CT, IPRDFT_PD, IPRDFT_Rader];
opts.breakdownRules.IPRDFT2 := [ IPRDFT2_Base1, IPRDFT2_Base2, IPRDFT2_CT];
opts.breakdownRules.PRDFT3 := [ ]; # PRDFT3_Base1, PRDFT3_Base2, PRDFT3_CT, PRDFT3_OddToPRDFT1;
opts.breakdownRules.URDFT := [ URDFT1_Base1, URDFT1_Base2, URDFT1_Base4, URDFT1_CT ];
opts.breakdownRules.DFT := [ DFT_Base, DFT_PD,  DFT_PD_loop];
opts.breakdownRules.PrunedPRDFT := [ PrunedPRDFT_base, PrunedPRDFT_CT_rec_block ];
opts.useDeref := false;
opts.globalUnrolling := 10;

RewriteRules(RulesRC, rec(
    RC_ISumAcc := Rule([RC, @(1, ISumAcc)], e -> ISumAcc(@(1).val.var, @(1).val.domain, RC(@(1).val.child(1)))),
    RC_COND := Rule([RC, @(1, COND)], e -> ApplyFunc(COND, [ @(1).val.cond ]::List(@(1).val._children, i->RC(i)))),
));

RewriteRules(RulesSums, rec(
    Scat_ISumAcc := ARule(Compose,
       [ @(1, Scat), @(2, ISumAcc) ],
        e -> [ CopyFields(@(2).val, 
        rec(_children :=  List(@(2).val._children, c -> ScatAcc(@(1).val.func) * c), dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ]),
    PropagateBB := Rule([@(1,BB), @(2,Compose, e->ForAny(e.children(), i->ObjId(i)=COND))],
        e -> Compose(List(@(2).val.children(), i->Cond(ObjId(i) = COND, ApplyFunc(COND, [i.cond]::List(i.children(), j->BB(j))), BB(i))))),
    MergeBB := ARule(Compose, [@(1, BB), @(2, BB)], e-> [ BB(@(1).val.child(1) * @(2).val.child(1)) ]),
    COND_true := Rule(@(1, COND, e->e.cond = V(true)), e -> @(1).val.child(1)),
    eq_false := Rule([@(1, eq), [add, @(3, var, e->e.t=TInt), @(2, Value, e->e.v >0) ], @(0, Value, e->e.v=0)], e-> V(false)),
    COND_false := Rule(@(1, COND, e->e.cond = V(false)), e -> @(1).val.child(2)),
    DiagISumAccLeft := ARule( Compose, [ @(1, ISumAcc, canReorder), @(2, RightPull) ],
        e -> [ ISumAcc(@1.val.var, @1.val.domain, @1.val.child(1) * @2.val).attrs(@(1).val) ]),
    COND_Diag :=  ARule( Compose, [ @(1, COND), @(2, [Diag, RCDiag]) ],
        e -> [ CopyFields(@(1).val, rec(_children := List(@(1).val._children, c -> c * @(2).val),
                dimensions := [Rows(@(1).val), Cols(@(2).val)] )) ])
));


RewriteRules(RulesStrengthReduce, rec(
    mul_cond := Rule([@(1, mul), [cond, @(2), @(3, Value), @(4, Value)], @(5)],
        e->cond(@(2).val, @(5).val*@(3).val, @(5).val*@(4).val)),
    eq_add := Rule([eq, [add, @(1,Value), @(2)], @(3,Value)], e->eq(@(2).val, @(3).val-@(1).val))     
));


# set up codegen for CUDA
#opts.codegen := CopyFields(opts.codegen, SMPCodegenMixin);

fixISumAccPDLoop := function(s, opts)
    local ss, vr, scat, dim, krnl, lvar, it0, itn, sct, isum, its, sx, scand, scats;
 
    while Length(Collect(s, ISumAcc)) >0  do
        scand := Collect(s, ISumAcc);
        ss := scand[1];
        vr := ss.var;
        scats := Collect(ss, ScatAcc);
        if Length(scats) > 1 then Error("There should be only one ScatAcc in an ISumAcc..."); fi;
        scat := scats[1];
        dim := Cols(scat);

        krnl := ss.child(1);
        lvar := Ind(vr.range-1);

        it0 := RulesSums(SubstTopDown(SubstVars(Copy(krnl), rec((vr.id) := V(0))), ScatAcc, e-> Scat(fId(dim))));
        itn := RulesSums(RulesSums(SubstTopDown(SubstVars(Copy(krnl), rec((vr.id) := lvar+1)), ScatAcc, e-> ScatAcc(fAdd(dim, dim, 0)))));
        sct := Scat(Collect(krnl, ScatAcc)[1].func);
        isum := ISum(lvar, itn);
        isum.doNotMarkBB := true;
        its := BB(sct) * SUM(it0, isum);

        s := SubstTopDown(s, ss, e->its);
        s := MergedRuleSet(RulesSums, RulesDiag, RulesDiagStandalone, RulesStrengthReduce, RulesRC)(s);
    od;
 
    return s;
end;


fixInitFunc := function(c, opts)
    local cis, ci, ci2, fc, freei, offending, myd, tags;
   
    tags := rec();
    if IsBound(c.dimensions) then tags.dimensions  := c.dimensions; fi;
    if IsBound(c.ruletree) then tags.ruletree  := c.ruletree; fi;
    
    cis := Collect(c, @(1, func, e->e.id="init"));
    for ci in cis do
        fc := c.free();
        freei := ci.free();
        offending := Filtered(fc, i->i in freei);
        for myd in offending do
           ci2 := func(ci.ret, ci.id, ci.params,
               data(myd, myd.value, ci.cmd));
           c := SubstTopDown(c, @(1, func, e->e = ci), e->ci2);    
        od;
    od;    
    
    c := CopyFields(tags, c);
    return c;    
end;

fixReplicatedData := function(c, opts)
    local datas, d, replicas, cc, r, tags;

    tags := rec();
    if IsBound(c.dimensions) then tags.dimensions  := c.dimensions; fi;
    if IsBound(c.ruletree) then tags.ruletree  := c.ruletree; fi;

    datas := Collect(c, data);
    for d in datas do
        replicas := Collect(d.cmd, @(1, data, e->e.value = d.value));
        for r in replicas do
            cc := r.cmd;
            cc := SubstVars(cc, rec((r.var.id) := d.var));
            c := SubstTopDown(c, @(1, data, e->e.var = r.var), e->cc);
        od;    
    od;
    c := CopyFields(tags, c);
    return c;
end;


scalarizeAccumulator := function(c, opts)
    local scand, svr, svars, tags;

    tags := rec();
    if IsBound(c.dimensions) then tags.dimensions  := c.dimensions; fi;
    if IsBound(c.ruletree) then tags.ruletree  := c.ruletree; fi;

    scand := Filtered(Set(Flat(List(Collect(c, decl), i->i.vars))), e->IsArray(e.t) and ForAll(Collect(c, @(1, nth, f-> f.loc = e)), g->IsValue(g.idx)));
    for svr in scand do
        svars := List([1..svr.t.size], e->var.fresh_t("q", svr.t.t));
        c := SubstTopDown(c, @(1, decl, e->svr in e.vars), e-> decl(svars :: Filtered(@(1).val.vars, i-> i <>svr), @(1).val.cmd));
        c := SubstTopDown(c, @(1, nth, e->e.loc = svr), e->svars[e.idx.v+1]);
    od;
    c := CopyFields(tags, c);
    return c;
end;


fixScatter := function(s, opts)
    local tags, scts, sct, f, fl, vars, fll, rng, it, itspace, vals, vby2, vby2d, vby2d2, accf2, flnew, sctnew, srec;
    
    tags := rec();
    if IsBound(s.ruletree) then tags.ruletree  := s.ruletree; fi;
    
    # very brittle and danger of infinite loop
    scts := Collect(s, @(1, Scat, e->Cols(e) = 4 and Length(Collect(e, BH)) > 0 and ForAny(e.free(), i->i.t <> TInt))); 
    while Length(scts) > 0  do
        sct := scts[1];
    
        f := sct.func;
        fl := f.lambda();
        vars := Filtered(fl.free(), i->i.t = TInt);
        fll := fl.tolist();
        rng := List(vars, i->i.range);
        itspace := Cartesian(List(rng, i->[1..i]));
    
        vals := [];
        for it in itspace do
            srec := rec();
            for i in [1..Length(vars)] do
               srec := CopyFields(srec, rec((vars[i].id):=V(it[i]-1)));
            od;
            vals := vals :: List(Copy(fll), e->EvalScalar(SubstVars(e, srec)));
        od;
       
        vby2 := List([1..Length(vals)/2], i->vals[2*i-1]/2);
        vby2d := fTensor(FData(List(vby2, i->V(i))), fId(2));
    
        vby2d2 := FData(List(vby2, i->V(i)));
        accf2 := fTensor(List(vars, fBase) :: [fId(fl.vars[1].range/2)]);
        flnew := fTensor(fCompose(vby2d2, accf2).lambda().setRange(Rows(sct)/2), fId(2));
    
        sctnew := Scat(flnew);
        s := SubstTopDownNR(s, sct, e->sctnew);
        # very brittle and danger of infinite loop
        scts := Collect(s, @(1, Scat, e->Cols(e) = 4 and Length(Collect(e, BH)) > 0 and ForAny(e.free(), i->i.t <> TInt))); 
    od;
    s := CopyFields(tags, s);
    return s;
end;


opts.codegen._Formula := Copy(opts.codegen.Formula);
opts.codegen.Formula := meth ( self, o, y, x, opts )
    local prog, prog1, s;
    s := fixISumAccPDLoop(Copy(o), opts);
    s := fixScatter(s, opts);

    s := SubstTopDown(s, [@(1,RCDiag, e->e.dims()=[2,2]), 
           @(2, FDataOfs, e->e.var.value = V([ V(1.0), V(0.0), V(1.0), V(0.0) ])), @(3, I)], e->I(2));
           
    s := SubstTopDown(s, [@(1,RCDiag, e->e.dims()=[4,4]), @(2, FDataOfs,
            e->ForAll([1..e.var.value.t.size/4], i->e.var.value.v[4*i-3] = V(1.0)) and ForAll([1..e.var.value.t.size/4],
            i->e.var.value.v[4*i- 2] = V(0.0))), @(3, I)],
        e->let(vals := V(Flat(List([1..@(2).val.var.value.t.size/4], i->[@(2).val.var.value.v[4*i-1], @(2).val.var.value.v[4*i]]))),
            dt := Dat(vals.t).setValue(vals),
            fdo := FDataOfs(dt, 2, V(2) * @(2).val.ofs.free()[1]),
            rcdnew  := RCDiag(fdo, I(2)),
            SumsSPL(DirectSum(I(2), rcdnew), opts))
    );
   
    prog := opts.codegen._Formula(s, y, x, opts);
    while Length(prog.free()) > 0 do
        prog := fixInitFunc(prog, opts);
    od;
   
    prog1 := skip();
    while prog1 <> prog do 
        prog1 := prog;
        prog := fixReplicatedData(prog, opts);
    od;
   
    prog := scalarizeAccumulator(prog, opts);
   
    return prog;
end;


opts.formulaStrategies.postProcess := opts.formulaStrategies.postProcess #:: [ fixISumAccPDLoop ] 
    :: opts.formulaStrategies.sigmaSpl;

opts.generateInitFunc := false;
 
#========================================================================================


# problem size
ns := 33;
n := 130;
blk := Gcd(n, ns);
scatpat := [0..ns/blk-1];

#============================================
# Stage 1
t1 := PrunedPRDFT(n, -1, blk, scatpat);
tm1 := MatSPL(t1);;


rt1 := RuleTreeMid(t1, opts);

c1 := CodeRuleTree(rt1, opts);

if not IsBound(c1) then
    Print("DFT_PD_Stage1: CodeRuleTree failed\n");
    TestFailExit();
fi;

cm1 := CMatrix(c1, opts);

if not IsBound(cm1) then
    Print("DFT_PD_Stage1: CMatrix failed\n");
    TestFailExit();
fi;

inorm := 2;
inorm := InfinityNormMat(cm1 - tm1);

if inorm > 1 then
    Print("DFT_PD_Stage1: InfinityNormMat failed\n");
    TestExitFail();
fi;

PrintLine("InfinityNormMat: ", inorm);



