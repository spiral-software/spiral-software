
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#
IDFUNC := (N,n,ofs) -> #Cond(
#    (N mod n) = 0 and (ofs mod n) = 0 and n<>1, fTensor(fBase(N/n, ofs/n), fId(n)),
#    IsInt(n) and IsInt(N) and IsInt(ofs) and Gcd(n, N, ofs) > 1, let(g := Gcd(n,N, ofs), fTensor(fAdd(N/g, n/g, ofs/g), fId(g))),
    H(N,n,ofs,1);
#);
#    fAdd(N,n,ofs));

RID := (N, n, ofs) -> Gath(IDFUNC(N,n,ofs));
WID := (N, n, ofs) -> Scat(IDFUNC(N,n,ofs));


###########################################################################################
# SumsSPL(spl, opts)
SumsSPL := function(spl, opts) 
    local s;
    if IsBound(opts.sumsgen) then
        trace_log.beginStage("SPL","Sigma-SPL", spl);    
        s := opts.sumsgen(spl, opts);
        trace_log.endStage("SPL","Sigma-SPL", s);    
        return s;
    else
        Error("opts.sumsgen must be defined!");
     fi;
end;

SumsSPLOpts := function(spl,opts)
  local sums;
  sums:=SumsSPL(spl,opts);
  return ApplyStrategy(sums,opts.formulaStrategies.postProcess,UntilDone, opts);
end;

LiftDataSums := function(sums)
    local p, datas, sums;
    p := Pull(sums, Data, e->e.child(1), e->[e.var, e.value]);
    sums := p[2];
    datas := tab();
    DoForAll(p[1], function(vv) datas.(vv[1].id) := vv[2]; end);
    sums.datas := datas;
    return sums;
end;

FinalizeSums := s->s;
sums -> SubstBottomUp(
    sums,
    [Compose, @(1).target(Scat), @(2).target(Gath)],
    e -> let(n := Rows(@(2).val),
         j := Ind(n),
         fbase := fBase(n,j),
         ISum(j, n,
         Scat(fCompose(@(1).val.func, fbase)) *
         Gath(fCompose(@(2).val.func, fbase)))));

###########################################################################################
DoNotReorder := function(sums) sums.doNotReorder := true; return sums; end;
###########################################################################################
NonTerminal.sums := self >> self;

O.sums := self >> self;

PermClass.sums := self >> Prm(self);

Rot.sums := self >> ExpandRotationsSPL(self).sums();

Sym.sums := self >> #When(self.isPermutation(),
#    SumsSPL(Perm(foldPerm(self.getObj()), Rows(self))),
    self.getObj().sums();

I.sums := self >> Prm(fId(self.params[1]));

# Perm is given by a list of numbers, and is only used for unrolled code,
# so I convert it to Gath, which is simpler. Using Prm would require
# something invertible, and a list of numbers is not invertible when
# using FList (inversion not implemented).
#
Perm.sums := self >> let(
    N := self.size,
    plist_r := Permuted([1..N], self.element^-1) - 1, # we want 0-based perms
    #plist_w := Permuted([1..N], self.element) - 1, # we want 0-based perms
    Gath(FList(N, plist_r).setRange(N))
);

Diag._sums := self >> let(
    N  :=  Rows(self),  ind  :=  Ind(N),  d  :=  Dat(TUnknown),
    DoNotReorder(
    ISum(ind, N,
        Data(d, self.element.lambda(),
        Compose(Scat(fBase(N, ind)),
                Blk1( nth(d, ind) ),     # SS( $.. $..)
                Gath(fBase(N, ind)))) ))
);

Diag.sums := self >> self;
RCDiag.sums := self >> self;

Mat.sums := self >> Blk(self.element);

Sparse.sums := self >> Blk(MatSparseSPL(self));

# Operators

Scale.sums := self >> Inherit(self, rec(_children := [self.child(1).sums()]));

#Compose(
#    Diag(LambdaList([1..Rows(self)], Subst(i -> $(self.scalar)))), self.child(1)).sums();

Conjugate.sums := self >> let(
    O := self.child(1), perm := self.child(2), Tperm := perm.transpose(),
    Compose(Tperm, O, perm).sums()
);

Compose.sums := self >> Cond(self.isPermutation(),
    Prm(ApplyFunc(fCompose, List(Reversed(self.children()), c->c.sums().func))),
    Compose(Map(self.children(), c->c.sums())));

_directSumSums := (obj, roverlap, coverlap) ->
    let(cspans  :=  BaseOverlap.spans(roverlap, List(obj.children(), Cols)),
    rspans  :=  BaseOverlap.spans(coverlap, List(obj.children(), Rows)),
    nblocks  :=  obj.numChildren(),
    C  :=  Cols(obj),
    R  :=  Rows(obj),
    OP  :=  When(coverlap > 0, SUMAcc, SUM),
    OP( Map([1..nblocks],
        i -> Compose( #W(N, bksize_cols, fTensor(fBase(nblocks, ind), fId(bksize)))
                      WID(R, 1 + rspans[i][2] - rspans[i][1], rspans[i][1]-1),
                      obj.child(i).sums(),
                  RID(C, 1 + cspans[i][2] - cspans[i][1], cspans[i][1]-1) ))));

DirectSum.sums    := self >> Cond(self.isPermutation(),
    Prm(ApplyFunc(fDirsum, List(self.children(), c->c.sums().func))),
    _directSumSums(self, 0, 0));

ColDirectSum.sums := self >> _directSumSums(self, 0,            self.overlap);

RowTensor.sums := self >>
    let( A := self.child(1),
     i := Ind(self.isize),
     ISum(i, i.range,
         Scat(fTensor(fBase(self.isize, i), fId(Rows(A)))) *
         A.sums() *
         Gath(fAdd(Cols(self), Cols(A), i * (Cols(A) - self.overlap)))) );

ColTensor.sums := self >>
    let( A := self.child(1),
     i := Ind(self.isize),
     ISumAcc(i, i.range,
         Scat(fAdd(Rows(self), Rows(A), i * (Rows(A) - self.overlap))) *
         A.sums() *
         Gath(fTensor(fBase(self.isize, i), fId(Cols(A))))) );

#Tensor.sums := self >>
#    Cond(self.isPermutation(),
#       Prm(
#       ApplyFunc(fTensor, List(self.children(), c->c.sums().func))),

#    self.numChildren() > 2,
#       self.rightBinary().sums(),

#    not IsIdentitySPL(self.child(1)),
#       self.parallelForm().sums(),

#    let(nblocks := Cols(self.child(1)),
#        ind := Ind(nblocks),
#        bkcols := Cols(self.child(2)),
#        bkrows := Rows(self.child(2)),
#        cols := Cols(self), rows := Rows(self),
#        ISum(ind, nblocks,
#        Compose(Scat(fTensor(fBase(nblocks, ind), fId(bkrows))),
#                self.child(2).sums(),
#            Gath(fTensor(fBase(nblocks, ind), fId(bkcols)))))
#    ));

Tensor.gathTensor := (self,i) >> fTensor;
Tensor.scatTensor := (self,i) >> fTensor;
Tensor.fTensor := fTensor;
Tensor.sums := meth(self)
    local ch, col_prods, row_prods, col_prods_rev, row_prods_rev, i, j1, j2, j1b, j2b, bkcols, bkrows, prod, term;
    if self.isPermutation() then return
    Prm(ApplyFunc(self.fTensor, List(self.children(), c->c.sums().func)));
    elif ForAll(self.children(), x->ObjId(x)=Diag) then return
    Diag(ApplyFunc(diagTensor, List(self.children(), c->c.element)));
    fi;

    ch := self.children();
    col_prods := ScanL(ch, (x,y)->x*Cols(y), 1);
    col_prods_rev := Drop(ScanR(ch, (x,y)->x*Cols(y), 1), 1);
    #row_prods := ScanL(ch, (x,y)->x*Rows(y), 1);
    #row_prods_rev := DropLast(ScanR(ch, (x,y)->x*Rows(y), 1), 1);

    prod := [];
    for i in [1..Length(ch)] do
        if not IsIdentitySPL(ch[i]) then
        bkcols := Cols(ch[i]);
        bkrows := Rows(ch[i]);
        j1 := Ind(col_prods[i]);
        j2 := Ind(col_prods_rev[i]);
        j1b := When(j1.range = 1, 0, fBase(j1));
        j2b := When(j2.range = 1, 0, fBase(j2));

        term := Scat(self.scatTensor(i)(Filtered(
            [j1b, fId(bkrows), j2b], x->x<>0))) *
                ch[i].sums() *
            Gath(self.gathTensor(i)(Filtered(
            [j1b, fId(bkcols), j2b], x->x<>0)));

        if j2.range <> 1 then
        term := ISum(j2, j2.range, term); fi;
        if j1.range <> 1 then
        term := ISum(j1, j1.range, term); fi;

        Add(prod, term);
    fi;
    od;
    return Compose(prod);
end;

Class(GammaTensor, Tensor, rec(
    gathTensor := (self,i) >> When(i=self.numChildren(), gammaTensor, fTensor),
    scatTensor := (self,i) >> When(i=1, gammaTensor, fTensor),
    #fTensor := gammaTensor,
    toAMat := self >> self.sums().toAMat()
));

_partSum := l -> List([1..Length(l)+1], i -> Sum(l{[1..i-1]}));

#-----------------------------------------------------------------------
IterDirectSum.sums := self >>
    let(s := fTensor(fBase(self.domain, self.var), fId(Rows(self.child(1)))),
        g := fTensor(fBase(self.domain, self.var), fId(Cols(self.child(1)))),
        ISum(self.var, self.domain,
             Scat(s) * self.child(1).sums() * Gath(g)));


_sbs := function(s, e)
    local cs;
    cs := Set(Copy(s));
    SubtractSet(cs, Set([e]));
    return cs;
end;

#-----------------------------------------------------------------------
IRowDirSum.sums := meth(self)
    local idx, i, ii, bkrows2, ivars,k, nblocks, q, overlap, s, cd1, rows, bkrows, vars, lambda, bs, ps, fps, gcd, rbase, len, rdom, rows_data, rows_by_gcd, rows_by_one, cols, bkcols, cdom, cbase, g,
        j, pj, rlambda;

    nblocks := self.domain;
    q := self.var;
    overlap := self.overlap;
    cd1 := spiral.code.RulesMergedStrengthReduce(self.child(1).dims()[1]);
    vars := _sbs(Filtered(cd1.free(), IsLoopIndex), q);
    if IsInt(cd1) then
        rows := Rows(self);
        bkrows := Rows(self.child(1));
        s := fTensor(fBase(nblocks, self.var), fId(bkrows));
#   FF: NOTE!! temporary hack for SAR
    elif IsExp(cd1) and Length(vars) = 0 then
        bkrows := _evInt(List(self.unrolledChildren(), i->spiral.code.RulesStrengthReduce(Rows(i))));
        rows := Rows(self);
        ps := _partSum(bkrows);
        gcd := Gcd(List(Concat(ps, [rows]), _unwrapV));
        rbase := FData(List(ps, i->TInt.value(i/gcd)));
        rdom := rbase.at(q+1)-rbase.at(q);
        s := fTensor(fAdd(rows/gcd, rdom, rbase.at(q)), fId(gcd));
    elif IsExp(cd1) and ObjId(cd1)=mul and IsValue(cd1.args[1]) then
        if Length(vars)=1 then
            lambda := Lambda(vars, Lambda(q, cd1));
            bs := Product(List(vars, i->i.range));
            bkrows := List(lambda.tolist(), i->i.tolist());
            ps := List(bkrows, _partSum);
            fps := Flat(ps);
            gcd := Gcd(List(Concat(fps, [cd1.args[1].v]), _unwrapV));
            rbase := FData(List(Flat(ps), i->TInt.value(i/gcd)));
            len := Length(ps[1]);
            rdom := rbase.at(vars[1]*len+q+1)-rbase.at(vars[1]*len+q);
            rows_data := idiv(Rows(self), gcd);
            rows_by_gcd := FData(Lambda(vars, rows_data).tolist()).at(vars[1]);
            s := fTensor(fAdd(rows_by_gcd, rdom, rbase.at(vars[1]*len+q)), fId(gcd));
        elif Length(vars) = 2 and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
            pj := Filtered(vars, IsParallelLoopIndex)[1];
            j := Filtered(vars, i -> not IsParallelLoopIndex(i))[1];
            lambda := Lambda(pj, Lambda(j, Lambda(q, cd1)));
            bs := Product(List(vars, i->i.range));
            bkrows := List(Flat(List(lambda.tolist(), i->i.tolist())), j->spiral.code.RulesMergedStrengthReduce(j.tolist()));
            ps := List(bkrows, _partSum);
            fps := Flat(ps);
            gcd := Gcd(List(Concat(fps, [cd1.args[1].v]), _unwrapV));
            rbase := FData(List(Flat(ps), i->TInt.value(i/gcd)));
            len := Length(ps[1]);
            rdom := rbase.at((pj * j.range +j)*len+q+1)-rbase.at((pj * j.range +j)*len+q);
            rows_data := idiv(Rows(self), gcd);
            rlambda := Lambda(pj, Lambda(j, rows_data));
            rows_by_gcd := FData(Flat(List(rlambda.tolist(), j->j.tolist() ))).at(pj * j.range +j);
            s := fTensor(fAdd(rows_by_gcd, rdom, rbase.at((pj * j.range +j)*len+q)), fId(gcd));
        elif  Length(vars) = 3 and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
            pj := Filtered(vars, IsParallelLoopIndex)[1];
            ivars := Filtered(vars, i -> not IsParallelLoopIndex(i));
            j := ivars[2];
            k := ivars[1];
	    k.isSWPLoopIndex := true;
            lambda := Lambda(k, Lambda(pj, Lambda(j, Lambda(q, cd1))));
            bs := Product(List(vars, i->i.range));
            bkrows := List(Flat(List(lambda.tolist(), i->i.tolist())), j-> (List(j.tolist(),kk->Flat(spiral.code.RulesMergedStrengthReduce(kk.tolist())))));
            bkrows2 :=[]; 
            for i in [1..Length(bkrows)] do
               for ii in [1..Length(bkrows[i])] do
                       Add(bkrows2, bkrows[i][ii]);
               od; 
                  
            od;    
            ps := List(bkrows2, _partSum);
            fps := Flat(ps);
            gcd := Gcd(List(Concat(fps, [cd1.args[1].v]), _unwrapV));
            rbase := FData(List(Flat(ps), i->TInt.value(i/gcd)));
            len := Length(ps[1]);
            idx := k * (pj.range * j.range) + (pj * j.range) + j;
            rdom := rbase.at(idx*len+q+1)-rbase.at(idx*len+q);
            rows_data := idiv(Rows(self), gcd);
            rlambda := Lambda(k, Lambda(pj, Lambda(j, rows_data)));
            rows_by_gcd := FData(Flat(List(rlambda.tolist(), j->List(j.tolist(),i->i.tolist())))).at(idx);
            s := fTensor(fAdd(rows_by_gcd, rdom, rbase.at(idx*len+q)), fId(gcd));
        else
            Error("Unsupported case -- please fix IRowDirSum.sums in spl2sums.gi:308");
        fi;
    elif IsExp(cd1) then
        vars := _sbs(Filtered(cd1.free(), IsLoopIndex), q);
        if Length(vars)=1 then
            lambda := Lambda(vars, Lambda(q, cd1));
            bs := Product(List(vars, i->i.range));
            bkrows := List(lambda.tolist(), i->i.tolist());
            ps := List(bkrows, _partSum);
            fps := Flat(ps);
            rbase := FData(List(Flat(ps), i->TInt.value(i)));
            len := Length(ps[1]);
            rdom := rbase.at(vars[1]*len+q+1)-rbase.at(vars[1]*len+q);
            rows_data := Rows(self);
            rows_by_one := FData(Lambda(vars, rows_data).tolist()).at(vars[1]);
            s := fAdd(rows_by_one, rdom, rbase.at(vars[1]*len+q));
        elif Length(vars) = 2 and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
            pj := Filtered(vars, IsParallelLoopIndex)[1];
            j := Filtered(vars, i -> not IsParallelLoopIndex(i))[1];
            lambda := Lambda(pj, Lambda(j, Lambda(q, cd1)));
            bs := Product(List(vars, i->i.range));

            bkrows := List(Flat(List(lambda.tolist(), i->i.tolist())), j->spiral.code.RulesMergedStrengthReduce(j.tolist()));
            ps := List(bkrows, _partSum);
            fps := Flat(ps);
            rbase := FData(List(Flat(ps), i->TInt.value(i)));
            len := Length(ps[1]);
            rdom := rbase.at((pj * j.range +j)*len+q+1)-rbase.at((pj * j.range +j)*len+q);

            rows_data := Rows(self);
            rlambda := Lambda(pj, Lambda(j, rows_data));
            rows_by_one := FData(Flat(List(rlambda.tolist(), j->j.tolist() ))).at(pj * j.range +j);
            s := fAdd(rows_by_one, rdom, rbase.at((pj * j.range +j)*len+q));
         elif Length(vars) = 3 and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
            pj := Filtered(vars, IsParallelLoopIndex)[1];
            ivars := Filtered(vars, i -> not IsParallelLoopIndex(i));
            j := ivars[2];
            k := ivars[1];
            k.isSWPLoopIndex := true;
            lambda := Lambda(k, Lambda(pj, Lambda(j, Lambda(q, cd1))));
            bs := Product(List(vars, i->i.range));
            bkrows := List(Flat(List(lambda.tolist(), i->i.tolist())), j-> (List(j.tolist(),kk->Flat(spiral.code.RulesMergedStrengthReduce(kk.tolist())))));
            bkrows2 :=[];
            for i in [1..Length(bkrows)] do
               for ii in [1..Length(bkrows[i])] do
                       Add(bkrows2, bkrows[i][ii]);
               od;

            od;
            ps := List(bkrows2, _partSum);
            fps := Flat(ps);
            rbase := FData(List(Flat(ps), i->TInt.value(i)));
            len := Length(ps[1]);
            idx := k * (pj.range * j.range) + (pj * j.range) + j;
            rdom := rbase.at(idx*len+q+1)-rbase.at(idx*len+q);
            rows_data := Rows(self);
            rlambda := Lambda(k, Lambda(pj, Lambda(j, rows_data)));
            rows_by_one := FData(Flat(List(rlambda.tolist(), j->List(j.tolist(),kk->kk.tolist())))).at(idx);
            s := fAdd(rows_by_one, rdom, rbase.at(idx*len+q));
        else
            Error("Unsupported case -- please fix IRowDirSum.sums in spl2sums.gi:344");
        fi;
    else
        Error("Unsupported case -- please fix IRowDirSum.sums in spl2sums.gi:347");
    fi;

    if IsInt(self.child(1).dims()[2]) then
        cols := Cols(self);
        bkcols := Cols(self.child(1));
        cdom :=bkcols-overlap;
        gcd := Gcd(List([cols, bkcols, cdom], _unwrapV));
        g := fTensor(fAdd(cols/gcd, bkcols/gcd, q*(cdom/gcd)), fId(gcd));
    else
        bkcols := _evInt(List(self.unrolledChildren(), i->Cols(i)));
        cols := Cols(self);
        ps := _partSum(bkcols);
        gcd := Gcd(List(Concat(ps, [cols]), _unwrapV));
        cbase := FData(List(ps, i->TInt.value(i/gcd)));
        cdom := cbase.at(q+1)-cbase.at(q);
        g := fTensor(fAdd(cols/gcd, cdom, cbase.at(q)), fId(gcd));
    fi;
    return
        ISum(self.var, self.domain,
            Compose(
                Scat(s),
                self.child(1).sums(),
                Gath(g)
            )
        );
end;


#NOTE: FF to YSV: Injecting H by default is BAD!!!
RowDirectSum.sums := meth(self)
    local c, nblocks, cols, rows, bkrows, bkcols, overlap, psr, psc, gcd, blks, bkrows_by_gcd, bkrows_by_gcd_data, vars, psr_data, psr_by_gcd,
        pj, j, k, lambda, ivars, llist, lambda2, llist2;
    overlap := self.overlap;
    nblocks  :=  self.numChildren();
    c := self.children();
    cols  :=  EvalScalar(Cols(self));
    rows := EvalScalar(Rows(self));
    bkcols := List(c, i->EvalScalar(Cols(i)));
#   FF: NOTE!! temporary hack for SAR
    bkrows := List(c, i->spiral.code.RulesStrengthReduce(Rows(i)));
    if ForAny(bkrows, IsExp) then
        bkrows := List(bkrows, i->spiral.code.RulesMergedStrengthReduce(i));
        if ForAny(bkrows, i->not (IsRec(i) and ObjId(i)=mul and IsValue(i.args[1]) and i.args[1].t=TInt)) then
            vars := Set(Flat(List(bkrows, i->Filtered(i.free(), IsLoopIndex))));
            gcd := 1;
            if Length(vars) = 1 then
                bkrows_by_gcd := List(bkrows, i->FData(Lambda(vars, i).tolist()).at(vars[1]));
                psr_data := DropLast(_partSum(bkrows), 1);
                psr_by_gcd := List(psr_data, i->When(IsExp(i), FData(Lambda(vars, i).tolist()).at(vars[1]), i));
            elif Length(vars) = 2 and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
                pj := Filtered(vars, IsParallelLoopIndex)[1];
                j := Filtered(vars, i -> not IsParallelLoopIndex(i))[1];
                lambda := List(bkrows, i->Lambda(pj, Lambda(j, i)));
                llist := List([1..Length(bkrows)], k->Flat(List(lambda[k].tolist(), i->i.tolist())));
                bkrows_by_gcd := List(llist, i->fCompose(FData(i), fTensor(fBase(pj), fId(j.range))).at(j));
                psr_data := List(DropLast(_partSum(bkrows), 1), i ->spiral.code.RulesMergedStrengthReduce(i));
                lambda2 := List(psr_data, i->Lambda(pj, Lambda(j, i)));
                llist2 := List([1..Length(bkrows)], k->Flat(List(lambda2[k].tolist(), i->i.tolist())));
                psr_by_gcd := List(llist2, i->fCompose(FData(i), fTensor(fBase(pj), fId(j.range))).at(j));
            elif Length(vars) = 3  and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
                pj := Filtered(vars, IsParallelLoopIndex)[1];
                ivars := Filtered(vars, i -> not IsParallelLoopIndex(i));
                j := ivars[2];
                k := ivars[1];
                k.isSWPLoopIndex := true;
                lambda := List(bkrows, i->Lambda(k, Lambda(pj, Lambda(j, i))));
                llist := List([1..Length(bkrows)], k->Flat(List(lambda[k].tolist(),i->List(i.tolist(),j->j.tolist()))));
                bkrows_by_gcd := List(llist, i->fCompose(FData(i), fTensor(fBase(k), fBase(pj), fId(j.range))).at(j));
                psr_data := List(DropLast(_partSum(bkrows), 1), i ->spiral.code.RulesMergedStrengthReduce(i));
                lambda2 := List(psr_data, i->Lambda(k, Lambda(pj, Lambda(j, i))));
                llist2 := List([1..Length(bkrows)], k->Flat(List(lambda2[k].tolist(), i->List(i.tolist(),kk->kk.tolist()))));
                psr_by_gcd := List(llist2, i->fCompose(FData(i), fTensor(fBase(k), fBase(pj), fId(j.range))).at(j));
            else
                Error("Unsupported case -- please fix RowDirSum.sums in spl2sums.gi:409");
            fi;
        else
            vars := Set(Flat(List(bkrows, i->Filtered(i.free(), IsLoopIndex))));
            blks := List(bkrows, i->i.args[1].v);
            gcd := Gcd(blks);
            if Length(vars) = 1 then
                bkrows_by_gcd_data := List(bkrows, i->ApplyFunc(mul, Concat([i.args[1].v/gcd], Drop(i.args, 1))));
                bkrows_by_gcd := List(bkrows_by_gcd_data, i->FData(Lambda(vars, i).tolist()).at(vars[1]));
                psr_data := List(DropLast(_partSum(bkrows), 1), i ->idiv(i,gcd));
                psr_by_gcd := List(psr_data, i->When(IsExp(i), FData(Lambda(vars, i).tolist()).at(vars[1]), i));
            elif Length(vars) = 2 and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
                bkrows_by_gcd_data := List(bkrows, i->spiral.code.RulesMergedStrengthReduce(ApplyFunc(mul, Concat([i.args[1].v/gcd], Drop(i.args, 1)))));
                pj := Filtered(vars, IsParallelLoopIndex)[1];
                j := Filtered(vars, i -> not IsParallelLoopIndex(i))[1];
                lambda := List(bkrows_by_gcd_data, i->Lambda(pj, Lambda(j, i)));
                llist := List([1..Length(bkrows)], k->Flat(List(lambda[k].tolist(), i->i.tolist())));
                bkrows_by_gcd := List(llist, i->fCompose(FData(i), fTensor(fBase(pj), fId(j.range))).at(j));
                psr_data := List(DropLast(_partSum(bkrows), 1), i ->spiral.code.RulesMergedStrengthReduce(idiv(i,gcd)));
                lambda2 := List(psr_data, i->Lambda(pj, Lambda(j, i)));
                llist2 := List([1..Length(bkrows)], k->Flat(List(lambda2[k].tolist(), i->i.tolist())));
                psr_by_gcd := List(llist2, i->fCompose(FData(i), fTensor(fBase(pj), fId(j.range))).at(j));
            elif Length(vars) = 3 and Length(Filtered(vars, IsParallelLoopIndex)) = 1 then
                bkrows_by_gcd_data := List(bkrows, i->spiral.code.RulesMergedStrengthReduce(ApplyFunc(mul, Concat([i.args[1].v/gcd], Drop(i.args, 1)))));
                pj := Filtered(vars, IsParallelLoopIndex)[1];
                ivars := Filtered(vars, i -> not IsParallelLoopIndex(i));
                j := ivars[2];
                k := ivars[1]; 
                k.isSWPLoopIndex := true;
                lambda := List(bkrows_by_gcd_data, i->Lambda(k, Lambda(pj, Lambda(j, i))));
                llist := List([1..Length(bkrows)], k->Flat(List(lambda[k].tolist(),i->List(i.tolist(),j->j.tolist()))));
                bkrows_by_gcd := List(llist, i->fCompose(FData(i), fTensor(fBase(k), fBase(pj), fId(j.range))).at(j));
                psr_data := List(DropLast(_partSum(bkrows), 1), i ->spiral.code.RulesMergedStrengthReduce(idiv(i,gcd)));
                lambda2 := List(psr_data, i->Lambda(k, Lambda(pj, Lambda(j, i))));
                llist2 := List([1..Length(bkrows)], k->Flat(List(lambda2[k].tolist(), i->List(i.tolist(),kk->kk.tolist()))));
                psr_by_gcd := List(llist2, i->fCompose(FData(i), fTensor(fBase(k), fBase(pj), fId(j.range))).at(j));
 
            else
                Error("Unsupported case -- please fix RowDirSum.sums in spl2sums.gi:432");
            fi;
        fi;
    else
        bkrows := List(bkrows, EvalScalar);
        gcd := Gcd(List(Concat([rows, cols, overlap], bkrows, bkcols), _unwrapV));
        bkrows_by_gcd := List(bkrows, i->i/gcd);
        psr_by_gcd := List(DropLast(_partSum(bkrows), 1), i ->idiv(i,gcd));
    fi;
    psc := DropLast(Concat([0], List(Drop(_partSum(bkcols), 1), i->i-overlap)), 1);
    return SUM( Map([1..nblocks],
        i -> Compose(
                    Scat(fTensor(fAdd(rows/gcd, bkrows_by_gcd[i], Cond(i=1, 0, i=2, bkrows_by_gcd[i-1], psr_by_gcd[i])), fId(gcd))),
                    self.child(i).sums(),
                    Gath(fTensor(fAdd(cols/gcd, bkcols[i]/gcd, psc[i]/gcd), fId(gcd))))
            ));
end;

#----------------------------------------------------------------------
IterHStack.sums := self >> let(
    bkcols := Cols(self.child(1)),
    bkrows := Rows(self.child(1)),
    nblocks := self.domain,
    cols := Cols(self), rows := Rows(self),
    ISumAcc(self.var, self.domain,
        Scat(fId(bkrows)) *
        self.child(1).sums() *
        Gath(fTensor(fBase(nblocks, self.var), fId(bkcols)))));

#-----------------------------------------------------------------------
IterVStack.sums := self >> let(
    bkcols := Cols(self.child(1)),
    bkrows := Rows(self.child(1)),
    nblocks := self.domain,
    cols := Cols(self), rows := Rows(self),
    ISum(self.var, self.domain,
        Scat(fTensor(fBase(nblocks, self.var), fId(bkrows))) *
        self.child(1).sums() *
        Gath(fId(bkcols))));

#-----------------------------------------------------------------------
