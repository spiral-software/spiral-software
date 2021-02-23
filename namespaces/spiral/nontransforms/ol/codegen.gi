
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


TempArrayOL:=function(y,x,child)
  local y1,x1,mytype,accu;
  return StripList(List(child.dmn(),l->TempVec(l)));
end;

#Class(OLCodegen,VectorCodegen, rec(
Class(OLCodegen, RecCodegenMixin, SMPCodegenMixin, VectorCodegen, rec(

    LinkIO := (self, o, y, x, opts) >> self(o.params[1], y, x, opts),

    Compose := meth(self, o,y,x,opts)
        local ch, numch, vecs, allow, i, j, indices, output, chcomp, veclen, rveclen, dmnoffset, rngoffset, outvar, cc;

    ch := Filtered(o.children(), i-> not i.name in ["DMPGath","DMPScat"]);
    numch := Length(ch);
    vecs := [y];
    allow := (x<>y);

    for i in [1..numch-1] do
        if allow and ObjId(ch[i])=Inplace then 
            vecs[i+1] := vecs[i];
        else 
            vecs[i+1] := TempArrayOL(y,x,ch[i]);
        fi;
    od;
    vecs[numch+1] := x;
    for i in Reversed([1..numch]) do
            if allow and ObjId(ch[i])=Inplace
        then vecs[i] := vecs[i+1]; fi;
    od;

        # everything was inplace, make it go from x -> y as expected
    if vecs[1] = vecs[numch+1] then vecs[1] := y; fi;
    if vecs[1] = vecs[numch+1] then vecs[numch+1] := x; fi;

    #Error("Codegn-Compose -> Wire");
    #Start from the most right and handle Wire

    i := numch;
    cc:=ch[i];
    if ObjId(cc)=BB then
	cc:=cc._children[1];
    fi;

    while (i>1 and (ObjId(cc)=Wire or ObjId(cc) = Cross)) do
    
        if ObjId(cc) = Wire then
        #Variable length
            if not IsList(vecs[i+1]) then
                veclen := 1;
                indices := Compacted([1..veclen]);
                output := List(cc.params[2]*indices,x->vecs[i+1]);
            else
                veclen := Length(vecs[i+1]);
                indices := Compacted([1..veclen]);
                output := List(cc.params[2]*indices,x->vecs[i+1][x]);
	fi;
	vecs[i] := output;
    else
    # Wires in Cross
        dmnoffset := 0;
        rngoffset := 0;
        for chcomp in cc._children do
	    if ObjId(cc)=BB then
	       cc:=cc._children[1];
	    fi;

        if (ObjId(chcomp) = Wire) then
            if not IsList(chcomp.dims()[2]) then
                veclen := 1;
            else
                veclen := Length(chcomp.dims()[2]);
            fi;
            if not IsList(chcomp.dims()[1]) then
                rveclen := 1;
            else
                rveclen := Length(chcomp.dims()[1]);
            fi;
            indices := Compacted([1..veclen]);
            output := List(chcomp.params[2]*indices,x->vecs[i+1][x+dmnoffset]);
            j:=1;
            for outvar in output do
                vecs[i][rngoffset + j] := outvar;
                j:=j+1;
            od;
            dmnoffset:=dmnoffset+veclen;
            rngoffset:=rngoffset+rveclen;
        fi;
    od;

    fi;
        i:=i-1;
        cc:=ch[i];
        if ObjId(cc)=BB then
	  cc:=cc._children[1];
	fi;
    od;



#    for i in [1..numch-1] do
#      if ObjId(ch[i])=Wire then
#          indices := Compacted([1..Length(vecs[i])]);
#          output := List(ch[i].params[2]*indices,x->vecs[i][x]);
#          vecs[i+1]:=output;
#      fi;
#    od;

    ## HACK For the A Cross I that were I is not computed
    ## Could be (advantageously) replaced by a Wire(Id) when
    ## the new system will be around

    # Disabled by Hao because it is broken in some way#
    for i in [2..numch] do
       if ObjId(ch[i])=Cross then
           for j in [1..Length(ch[i]._children)] do
              if (ObjId(ch[i]._children[j])=Prm and
                          ObjId(ch[i]._children[j].func)=fId and
                          ObjId(vecs[i][j].t)=TArray)or
                      (ObjId(ch[i]._children[j])=BB and
                          ObjId(ch[i]._children[j]._children[1])=Prm and
                          ObjId(ch[i]._children[j]._children[1].func)=fId and
                          ObjId(vecs[i][j].t)=TArray) or
                      (ObjId(ch[i]._children[j])=BB and
                          ObjId(ch[i]._children[j]._children[1])=I and
                          ObjId(vecs[i][j].t)=TArray) then
	      skip();
              #vecs[i][j]:=vecs[i+1][j];
          fi;
           od;
       fi;
    od;

    [vecs, ch] := [Reversed(vecs), Reversed(ch)];
    return decl( Difference(Flat(vecs{[2..Length(vecs)-1]}), Flat([x,y])),
        chain( List([1..numch], i -> When(vecs[i+1]=vecs[i],
            self(ch[i], vecs[i],   vecs[i], CopyFields(opts, rec(_inplace:=true))),
            self(ch[i], vecs[i+1], vecs[i], opts)))));
    end,

    ScatInit := (self, o, y, x, opts) >> let(ii := Ind(),
    condition := When(Length(o.cond)=0,V(1),FoldL1(List(o.cond,x->eq(x,V(0))),logic_and)),
    chain(
    IF(condition,
        loop(ii, o.func.domain(), assign(nth(y,o.func.at(ii)), When(ObjId(y)=var,y.t.t.zero(),y.t.zero()))),
        skip()),
    self(o._children,y,x,opts))
    ),

    Wire := (self, o, y, x, opts) >> skip(),

    ScatAcc := (self, o, y, x, opts) >>
        self._acc(self(Scat(o.func),y,x,opts),y),

    ICScatAcc := meth(self,o,y,x,opts)
      local i, func;
      i := Ind(); func := o.func.lambda();
      return loop(i, o.func.domain(), chain(assign_acc(nth(y,func.at(i)), nth(x, i)),
                        assign_acc(nth(y,add(func.at(i),1)),nth(x,add(i,1)))));
    end,


     # NOTE: handle unaligned case - IsUnalignedPtrT(y)
     VScatAcc := (self, o, y, x, opts) >>
         self._acc(self(VScat(o.func,o.v),y,x,opts),y),

     VTensor_OL := meth(self, o, y, x, opts)
       local CastToVect;
       CastToVect:=x->tcast(TPtr(TVect(x.t.t, o.vlen)), x);
       return self(o.child(1), StripList(List(Flat([y]),t->CastToVect(t))), StripList(List(Flat([x]),t->CastToVect(t))), opts);
     end,


##########
### to FIX: this Multiplication implicitely takes 2 inputs
    Multiplication:= meth(self, o, y, x, opts)
    local iterator;

    iterator:=Ind();
    return loop(iterator, [ 0 .. o.element[2]-1 ],
            assign(nth(StripList(y), iterator), mul(nth(x[1], iterator),nth(x[2], iterator))));
    end,

    ICMultiplication:=meth(self,o,y,x,opts)
    local len,iterator;

    iterator:=Ind();
    len := When(o.element[2] = 2,1,o.element[2]);
    len := When(Length(o.element) = 3,len/2,len);
    return loop(iterator, [ 0 .. (len)- 1 ],
            chain(
                assign(nth(StripList(y),mul(iterator,2)),sub(mul(nth(x[1], mul(iterator,2)),nth(x[2],mul(iterator,2))),
                                 mul(nth(x[1], add(mul(iterator,2),1)),nth(x[2], add(mul(iterator,2),1))))),
                assign(nth(StripList(y), add(mul(iterator,2),1)),add(mul(nth(x[1], mul(iterator,2)),nth(x[2], add(mul(iterator,2),1))),
                                             mul(nth(x[1], add(mul(iterator,2),1)),nth(x[2],mul(iterator,2)))))));

    end,

    Addition:= meth(self, o, y, x, opts)
        local iterator;
        iterator:=Ind();
        return loop(iterator, [ 0 .. _unwrap(o.element[2])-1 ],
            assign(nth(StripList(y), iterator), add(nth(x[1], iterator),nth(x[2], iterator))));
    end,

   Subtraction:= meth(self, o, y, x, opts)
        local iterator;
        iterator:=Ind();
        return loop(iterator, [ 0 .. o.element[2]-1 ],
            assign(nth(StripList(y), iterator), sub(nth(x[1], iterator),nth(x[2], iterator))));
    end,

    Codelet := meth(self, o, y, x, opts)
        local code,inputlist,outputlist,outputTypes,inputTypes;

    o := o.child(1);
        o := SubstBottomUp(o,BB,e->e.rChildren()[1]);
        o := OLQuickAndDirtyHackForCodelet(o);

        o:=ApplyStrategy(o,opts.formulaStrategies.postProcess, UntilDone, opts);
        o := OLRulesBufferFinalize(o);
        outputTypes := opts.OutputTypes;
        inputTypes := opts.InputTypes;
        inputlist :=[];
        outputlist :=[];
        for i in [1..DimLength(o.dims()[1])] do
          Append(outputlist, [TPtr(outputTypes[i])]);
        od;
        for i in [1..DimLength(o.dims()[2])] do
          Append(inputlist, [TPtr(inputTypes[i])]);
        od;
        Unification(o, StripList(inputlist));

        if List(outputlist,l->l.t)<>List(o.rng(),l->l.t) then
          Error("Unification of final output failed");
        fi;

    ## Generating code : main body
#        o := BlockSums(opts.libgen.basesUnrolling, o);
    code := SReduce(self(o, y, x, opts), opts);
        code := ESReduce(code, opts);
    code := RemoveAssignAcc(code);
    code := BlockUnroll(code, opts);
    code := DeclareHidden(code);
    return code;
    end,

   Formula := meth(self, o, y, x, opts)
        local code, init_code, datas, prog, params, sub, inputTypes, outputTypes,
    initsub,i,inputlist,outputlist,smp, num_threads, buffers, bufalloc, dalloc, dvars, map,ignore, codelet_codes, codelet_recs, chash;

    o := o.child(1);
        o := OLRulesBufferFinalize(o);
    o := OLRulesCode(o);


    params := Set(Collect(o, param));
    smp := Collect(o, SMPSum);
    num_threads := When(smp=[], 1, smp[1].p);
    if not ForAll(smp, x->x.p=num_threads) then Error("Non-uniform num_threads in SMPSum's"); fi;
        outputTypes := opts.OutputTypes;
        inputTypes :=  opts.InputTypes;
    inputlist :=[];
    outputlist :=[];
    for i in [1..DimLength(o.dims()[1])] do
      Append(outputlist, [TPtr(outputTypes[i])]);
    od;
    for i in [1..DimLength(o.dims()[2])] do
      Append(inputlist, [TPtr(inputTypes[i])]);
    od;
    Unification(o, StripList(inputlist));

    if List(outputlist,l->l.t)<>List(o.rng(),l->l.t) then
      Error("Unification of final output failed");
    fi;

    inputlist :=[];
    outputlist :=[];
    for i in [1..DimLength(o.dims()[1])] do
      Append(outputlist, [var(ConcatenationString("Y",String(i)), TPtr(outputTypes[i]))]);
    od;
    for i in [1..DimLength(o.dims()[2])] do
      Append(inputlist, [var(ConcatenationString("X",String(i)), TPtr(inputTypes[i]))]);
    od;

        #We're dropping y and x right away because previous stuff is backward compatibility
    y:=StripList(outputlist);
        x:= StripList(inputlist);


        ## HACK for the unroll of ROIs
        if IsBound(opts.libgen) then
            chash:=spiral.libgen.CreateCodeletHashTable();
            for i in Flat(opts.libgen.codeletTab.entries) do
                if Collect(i.data.sums,Multiplication)=[] then
                    i.data.unrolling:=300;         #this is the unrolling threshold for non kernels
                fi;
                HashAdd(chash,i.key, i.data);
            od;
            opts.libgen.codeletTab:=chash;
        elif Collect(o,Multiplication)=[] then
            o := SubstBottomUp(o,BB,e->e.rChildren()[1]);
            SubstTopDownNR(o, @(1).cond(IsMarkedBlock),
                function(e)
                    local f;
                    f:=@(1).val;
                    f.isBlock:=false;
                    return f;
                end);
            o := compiler.BlockSumsOpts(o,CopyFields(opts,rec(globalUnrolling:=32)));
        fi;

    ## Generating code : codelets
    codelet_recs := spiral.libgen.CompileCodelets(o, opts);
    codelet_codes := List(codelet_recs,
            function(clrec)
              local inputs, outputs, myf;
              outputs:=StripList(List([1..Length(Flat([clrec.sums.dims()[1]]))],x->var(Concat("Y",When(x>1,String(x),"")), TPtr(outputTypes[x]))));
              inputs:=StripList(List([1..Length(Flat([clrec.sums.dims()[2]]))],x->var(Concat("X",When(x>1,String(x),"")), TPtr(inputTypes[x]))));
              myf:=func(TVoid, clrec.name, Concatenation(Flat([outputs, inputs]), clrec.params), clrec.code);
              myf.inline:=true;
              return myf;
            end
        );

    map := tab();
    ## Generating code : main body
    datas := Collect(o, FDataOfs);
        code := self(o, y, x, opts);
    code := SReduce(code, opts);
        code := ESReduce(code, opts);
    code := RemoveAssignAcc(code);
    code := BlockUnroll(code, opts);
#   code := SimplifySingularArray(code);
        # code := PowerOpt(code);
    code := DeclareHidden(code);
    if IsBound(opts.isFixedPoint) and opts.isFixedPoint then
        code := FixedPointCode(code, opts.bits, opts.fracbits);
    fi;
    [buffers, bufalloc, code] := spiral.libgen.AllocBuffersSMP("buf", code, map, opts);

    ## Generating code : initialization
    [dvars, dalloc, ignore] := spiral.libgen.AllocBuffersSMP("dat", decl(datas, skip()), map, opts);

    # NOTE: is there a better way?
    sub := opts.subName;
    initsub := opts.subInitName;
    code := func(TVoid, sub, Concatenation(params,
                When(IsBound(opts.subParams), opts.subParams, []), [y,x]), code);

        init_code := func(TVoid, initsub, [],
            chain(bufalloc, dalloc, List(datas, x -> SReduce(x.var.init, opts))));

        prog := SubstVars(chain(init_code, code), map);

        if Length(Filtered(opts.subParams,x->x.id="num_threads"))>0 then
            prog := data(var("NUM_THREADS", TInt), V(num_threads),prog);
        fi;

        prog := program(
            codelet_codes,
            decl(Concatenation(dvars, buffers, List(datas, x->x.var)),
        prog));

        prog:= GenerateBench(o, prog,opts);

    return FlattenCode(prog);
    end,
));


DefaultCodegen.ExpDiag := (self, o, y, x, opts) >> let(
   i   := Ind(),     d := o.d,
   elt := o.element, s := Length(elt.vars),
   xx  := List([0..s-1], j -> nth(x, i*s + j)),
   loop(i, d, 
       assign(nth(y,i), elt.at(xx)))
);


