
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(TTensorI_OL);
Declare(TTensorI_OL_Vectorize_AVecLast);

#Class(RewriteBarrier,RewritableObject);
#Class(RewriteBarrier,DFT);
#RewriteBarrier.numops:=self>>1;

DropVectorTag:=function(nt)
    return nt.withoutTag(AVecReg); 
#D    return SetTag(nt,Filtered(GetTags(nt),t->ObjId(t)<>AVecReg));
end;

DropParTag:=function(nt)
   return nt.withoutTag(AParSMP);
#D    return SetTag(nt,Filtered(GetTags(nt),t->ObjId(t)<>AParSMP));
end;

Devectorize:=function(l,vlen)
  return List(l,e->
      Cond(
          ObjId(e)=TArray and e.t=TUnknown, TArray(e.t,e.size*vlen),
          ObjId(e)=TArray and ObjId(e.t)=TVect, TArray(e.t.t,e.size*e.t.size),
          ObjId(e)=TVect, TArray(e.t,e.size),
          e)
  );
end;

Class(VTensor_OL, Tensor, rec(
    new := (self, L) >> SPL(WithBases(self, rec(
        _children := [L[1]],
        vlen := L[2]))),
    print := (self,i,is) >> Print(self.name, "(",
        self.child(1).print(i+is,is), ", ", self.vlen,")"),
    sums := self >> Inherit(self, rec(_children := [self.child(1).sums()])),
    isPermutation := False,
    rng := meth(self)         #FULL HACK
       return Devectorize(self.child(1).rng(),self.vlen);
    end,
    dmn := meth(self)         #FULL HACK
       return Devectorize(self.child(1).dmn(),self.vlen);
    end,
    dims := meth(self)
       if (IsBound(self.rng)and IsBound(self.dmn)) then
          return [StripList(List(self.rng(),l->l.size)),
                  StripList(List(self.dmn(),l->l.size))];
       fi;
    end,
));

#VTensor.dmn:=  meth(self)         #FULL HACK
#  local x; 
#  x:=self.child(1).dmn()[1];
#  if ObjId(x)=TArray and x.t=TUnknown then         
#      return [TArray(TUnknown,x.size*self.vlen)];
#  else
#      return Devectorize(self.child(1).dmn());
#  fi;
#end;
#VTensor.rng:=  meth(self)         #FULL HACK
#  local x; 
#  x:=self.child(1).rng()[1];
#  if ObjId(x)=TArray and x.t=TUnknown then         
#      return [TArray(TUnknown,x.size*self.vlen)];
#  else
#      return Devectorize(self.child(1).rng());
#  fi;
#end;

BlockVPerm.dmn := meth(self)
       return [TArray(self.child(1).dmn()[1].t,self.child(1).dmn()[1].size*self.n)];
end;
BlockVPerm.rng := meth(self)
       return [TArray(self.child(1).rng()[1].t,self.child(1).rng()[1].size*self.n)];
end;

Class(ScatQuestionMark, Scat, rec());
Class(ICScatAcc,Scat,rec(codeletName:="ICSA"));

Class(VScatQuestionMark, VScat, rec());

Class(VScat_svQuestionMark, VScat_sv, rec());

Class(ScatInit, BaseMat, rec(
   new := meth(self, f,con,c)
        local res;
        res := SPL(WithBases(self, rec(func:=f,cond:=con, _children:=c)));
        return res;
   end,
   rng:=self>>self._children.rng(),
   dmn:=self>>self._children.dmn(),
   print := meth(self,i,is)
      Print(self.name, "(", self.func, ", ",self.cond,", ");
      Print("\n", Blanks(i+is));
      SPLOps.Print(self._children, i + is, is);
      Print("\n", Blanks(i),")");
      return;
   end,
   rChildren := self >> [self._children,self.func],
   rSetChild := meth ( self, n, newChild )
     if n= 1  then
         self._children :=newChild;
     elif n=2 then
         self.func := newChild;
else
        Error("<n> must be 1");
     fi;
   end

));

Class(ScatInitProbe,ScatInit,rec());
Class(ScatInitFixed,ScatInit,rec());

Class(KroneckerSymbol, BaseMat, rec(
   abbrevs := [ arg -> [Flat(arg)] ],
   new := meth(self, l)
        local res;
        res := SPL(WithBases(self, rec(element:=l)));
        return res;
   end,
   isExp:=true,
   dims:=self>>[1,1]  #avoid recursive definition for the old-school Tensor
));

## Base Vector
Class(BV, BaseMat, rec(
   new := (self, i) >> #Checked(IsVar(i),SPL(WithBases(self, rec(element:=i))))
       SPL(WithBases(self, rec(element:=i))),
   dims:=self>>self.element.dimensions #[0,0]  #avoid recursive definition for the old-school Tensor
));

## Multiplication operator
## Multiplication(1,n) is I(n)
## Multiplication(2,n) is a point-wise multiplication of two vectors of size n
 Class(Multiplication, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
        local res;
        if l[1]=1 then return Prm(fId(l[2])); fi;
        res := SPL(WithBases(self, rec(element:=l,TType:=Replicate(l[1],TUnknown))));
        return res;
    end,
    isPermutation := self >> false,
#    dmn:=self>>Replicate(self.element[1],TArray(self.TType,self.element[2])),
#HACK
#    dmn:=meth(self) local a; a:=Replicate(self.element[1],TArray(self.TType,self.element[2]));a[1]:=TArray(TReal,1);return a; end,
    dmn:=self >>List(self.TType,x->TArray(x,self.element[2])),
    rng:=self>>let(a:=Try(First(self.TType,x->ObjId(x)=TVect)),t:=Cond(a[1],a[2],self.TType[1]),[TArray(t,self.element[2])]),
    sums:= self>>self,
    numops:=self>>self.element[2]*(self.element[1]-1),
    transpose := self >>self,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ", ",self.element[2],")"); self.printA();
      return;
     end
 ));
 Class(ICMultiplication,Multiplication, rec(

 ));

  _mk_advdim_r := (e) -> List(e, x -> When(IsList(x), _mk_advdim_r(x), [x]));
  _mk_advdim := (d) -> _mk_advdim_r(When(IsList(d), d, [d]));
                        
  Class(Glue, BaseMat, rec(
    abbrevs := [(n,size) ->[n,size]],
    new := meth(self,n,size)
        local res;
        res := SPL(WithBases(self, rec(element:=[n,size],dimensions:=[size*n,Replicate(n,size)],TType:=Replicate(n,TUnknown))));
        return res;
    end,
    isPermutation := self >> false,
    dmn:=self >>List([1..(self.element[1])],x->TArray(self.TType[x],self.element[2])),
    rng:=self>>[TArray(self.TType[1],self.element[1]*self.element[2])],
    sums:= self>>self,
    transpose:=self>>Copy(self),
    numops:=self>>self.element[2]*(self.element[1]-1),
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ", ",self.element[2],")"); self.printA();
      return;
     end,
    rChildren := self >> self.element,
    rSetChild := meth( self, n, newChild ) self.element[n] := Checked(n=1 or n=2, newChild); end,
    advdims := self >> let( d := self.dims(), [ _mk_advdim(d[1]), _mk_advdim(d[2]) ]),
    normalizedArithCost := (self) >> 0,
    isReal := self >> true, # makes no sense
 ));


  Class(Split, BaseMat, rec(
    abbrevs := [(size,n) ->[size,n]],
    new := meth(self,size,n)
        local res;
        res := SPL(WithBases(self, rec(element:=[size,n],dimensions:=[Replicate(n,size / n),size],TType:=[TUnknown])));
        return res;
    end,
    isPermutation := self >> false,
    rng:=self >>List([1..self.element[2]],x->TArray(self.TType[1],self.element[1] / self.element[2])),
    dmn:=self>>[TArray(self.TType[1],self.element[1])],
    sums:= self>>self,
    transpose:= self>>Copy(self),
    numops:=self>>(self.element[1]),
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ", ",self.element[2],")"); self.printA();
      return;
     end,
    rChildren := self >> self.element,
    rSetChild := meth( self, n, newChild ) self.element[n] := Checked(n=1 or n=2, newChild); end,
    advdims := self >> let( d := self.dims(), [ _mk_advdim(d[1]), _mk_advdim(d[2]) ]),
    normalizedArithCost := (self) >> 0,
    isReal := self >> true, # makes no sense
 ));


## NoOp, used with in COND_OL()
Class(NoOp, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(1, TUnknown))));
        fi;
    end,
    isPermutation := self >> false,
    dmn:=self >> [TArray(self.TType[1], self.element[1])],
    rng:=self >> [TArray(self.TType[1], self.element[1])],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));


## Constant
Class(Const, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(1, TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> [],
# [TArray(self.TType[1], 0)],
    rng:=self >> [TArray(self.TType[1], 1)],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));

Class(Or, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(l[1], TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> List(self.TType, x->TArray(x,1)),
    rng:=self >> [ TArray(self.TType[1], 1)],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));

Class(And, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(l[1], TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> List(self.TType, x->TArray(x,1)),
    rng:=self >> [ TArray(self.TType[1], 1)],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));

Class(NotEqual, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(l[1], TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> List(self.TType, x->TArray(x,1)),
    rng:=self >> [ TArray(self.TType[1], 1)],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));

Class(Equal, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(l[1], TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> List(self.TType, x->TArray(x,1)),
    rng:=self >> [ TArray(self.TType[1], 1)],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));

Class(ExclusiveOr, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(l[1], TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> List(self.TType, x->TArray(x,1)),
    rng:=self >> [ TArray(self.TType[1], 1)],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));

## Minimums
## Minimums(n) -> Input is n-way vector
Class(Minimums, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(l[1], TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> List(self.TType, x->TArray(x,1)),
    rng:=self >> [ TArray(self.TType[1], 1)],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));

## Maximums
## Maximums(n) -> Input is n-way vector
Class(Maximums, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
       local res;
        if (Length(l) = 1) then
          return SPL(WithBases(self, rec(element:=l, TType:=Replicate(l[1], TUnknown))));
        fi;
	# Here
    end,
    isPermutation := self >> false,
    dmn:=self >> List(self.TType, x->TArray(x,1)),
    rng:=self >> [ TArray(self.TType[1], 1) ],
    sums:= self>>self,
    numops:=self>> 0,
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ")"); self.printA();
      return;
    end
));


#Addition
Class(Addition, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
        local res;
        if l[1]=1 then return Prm(fId(l[2])); fi;
        res := SPL(WithBases(self, rec(element:=l,TType:=Replicate(l[1],TUnknown))));
        return res.setDims();
    end,
    isPermutation := self >> false,
    
    dmn:=self >> List( [1..self.element[1]], i -> TArray(TUnknown, self.element[2])),
    rng:=self >> [TArray(TUnknown, self.element[2])],
    sums:= self>>self,
    numops:=self>>self.element[2]*(self.element[1]-1),
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ", ",self.element[2],")"); self.printA();
      return;
     end,
    transpose := self >> InertTranspose(self),
    area := self >> self.element[2]*(self.element[1]-1),
));

Class(Subtraction, BaseMat, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := meth(self, l)
        local res;
        if l[1]=1 then return Prm(fId(l[2])); fi;
        res := SPL(WithBases(self, rec(element:=l,TType:=Replicate(l[1],TUnknown))));
        return res;
    end,
    isPermutation := self >> false,
    dmn:=self >>List(self.TType,x->TArray(x,self.element[2])),
    rng:=self>>let(a:=Try(First(self.TType,x->ObjId(x)=TVect)),t:=Cond(a[1],a[2],self.TType[1]),[TArray(t,self.element[2])]),
    sums:= self>>self,
    numops:=self>>self.element[2]*(self.element[1]-1),
    print := meth(self,i,is)
      Print(self.name, "(", self.element[1], ", ",self.element[2],")"); self.printA();
      return;
     end
));

# Declare(Cross);
# Class(Cross, BaseOperation, rec(
#    abbrevs := [ arg -> [Flat(arg)] ],

#    new := meth(self, L)
#         if Length(L)=1 then return L[1]; fi;
#         return SPL(WithBases(self, rec(_children:=L)));
#    end,

#    codeletName:="C",
#    isPermutation := self >> false,
#    dmn:=self>>let(li:=[],Flat(List(self._children, x-> x.dmn()))),
#    rng:=self>>let(li:=[],Flat(List(self._children, x-> x.rng()))),
#    sums:= meth(self)
#    local i;
#    for i in [1..Length(self._children)] do
#       self._children[i]:=self._children[i].sums();
#    od;
#    return self;
#    end,
#    hasLeftGath:=meth(self)
#       local i;
#       for i in self._children do
# #          if (ObjId(i)=Compose and ObjId(i.rChildren()[1])=Gath) or 
# #              (ObjId(i)=Gath) then
# #              return true;
#            if (ObjId(i)=Compose and IsBound(i.rChildren()[1].func) and i.rChildren()[1].func.free()<>Set([])) or (IsBound(i.func) and (not ObjId(i)=VGath_dup) and i.func.free()<>Set([])) and not(ObjId(i)=ISum) then
#                return true;
#           fi;
#       od;
#       return false;
#    end,
#    hasRightScat:=meth(self)
#       local i;
#       for i in self._children do
# #          if (ObjId(i)=Compose and ObjId(Last(i.rChildren()))=Scat) or 
# #              (ObjId(i)=Scat) then
# #              return true;
# #          fi;
#            if (ObjId(i)=Compose and ((IsBound(Last(i.rChildren()).func) and Last(i.rChildren()).func.free()=Set([])) or (ObjId(Last(i.rChildren()))=ISum) or (ObjId(Last(i.rChildren()))=VGath_dup and Length(Last(i.rChildren()).func.free())=1))) or (IsBound(i.func) and i.func.free()=Set([])) or (ObjId(i)=VGath_dup and Length(i.func.free())=1) or ObjId(i)=ISum  then
#                return true;
#           fi;
#       od;
#       return false;
#    end,
#    splitCross:=meth(self)
#       local l,r;
#       l:=Copy(self);
#       r:=[];
#       for i in [1..Length(l._children)] do
# #          if (ObjId(l._children[i])=Compose and ObjId(Last(l._children[i].rChildren()))=Scat) then
#            if (ObjId(l._children[i])=Compose and ((IsBound(Last(l._children[i].rChildren()).func) and Last(l._children[i].rChildren()).func.free()=Set([])) or (ObjId(Last(l._children[i].rChildren()))=ISum) or (ObjId(Last(l._children[i].rChildren()))=VGath_dup and Length(Last(l._children[i].rChildren()).func.free())=1))) or (ObjId(l._children[i])=ISum) then
#               Add(r,Last(l._children[i].rChildren()));
#               l._children[i]._children:=DropLast(l._children[i]._children,1);
# #          elif (ObjId(l._children[i])=Scat) then
#            elif (IsBound(l._children[i].func) and l._children[i].func.free()=Set([])) or (ObjId(i)=VGath_dup and Length(i.func.free())=1) then
#               Add(r,l._children[i]);
#               l._children[i]:=Prm(fId(l._children[i].dims()[1]));
#           else
#               Add(r,Prm(fId(l._children[i].dims()[2])));
#           fi;
#       od;
#       return [l,Cross(r)];
#    end
#  )
# );

Class(CrossBlockTop,Cross);

Identities:=function(l)
   return Cross(List(l,x->Prm(fId(x.size))));
end;


BB.numops:= self>>self.child(1).numops();
NoPull.numops:= self>>self.child(1).numops();
NoPullRight.numops:= self>>self.child(1).numops();
NoPullLeft.numops:= self>>self.child(1).numops();

PushR.numops:= self>>self.child(1).numops();

ISum.numops:=self>>self.child(1).numops()*self.domain;

SUM.numops:=self>>Sum(List(self._children,c->c.numops()));

Cross.numops:=self>>Sum(List(self._children,c->c.numops()));

Compose.numops:=self>>Sum(List(self._children,c->c.numops()));

Scat.numops:=self>>self.dims()[2];

Gath.numops:=self>>self.dims()[1];

VGath.numops:=self>>self.dims()[1];
VGath_dup.numops:=self>>self.dims()[1]; 
VReplicate.numops:=self>>self.v;
VHAdd.numops:=self>>self.v*self.v;
VPerm.numops:=self>>self.dimensions[1];
Tensor.numops:=self>>self.dimensions[1];

ScatAcc.numops:=self>>self.dimensions[2]*2; #an add and a store
VScatAcc.numops:=self>>self.dimensions[2]*2; #an add and a store
VScat.numops:=self>>self.dimensions[2];

fBase.numops:=self>>0;
fTensor.numops:=self>>0;
fId.numops:=self>>0;
ScatInit.numops:=self>>self._children.numops();

Prm.numops:=self>>0;
I.numops:=self>>0;

VTensor_OL.numops:=self>>self._children[1].numops()*self.vlen;

Class(AOne,AGenericTag);
Class(AMul,AGenericTag, rec(
    __call__ := meth ( arg )
      local  result, self, params;
      self := arg[1];
      params := arg{[ 2 .. Length(arg) ]};
      result := WithBases(self, rec(
              params := params,
              operations := PrintOps));
      return result;
    end,
    print := self >> Print(self.__name__, "(", PrintCS(self.params), ")")
   ));

Class(VOLWrap, VWrapBase, rec(
    __call__ := (self,isa) >> Checked(IsSIMD_ISA(isa), 
        WithBases(self, rec(operations:=PrintOps, isa:=isa))),

    wrap := (self,r,t) >> let(isa := self.isa, v := isa.v,
#This is OBVIOUSLY a hack. only deals with some kind of vectorization
            nontransforms.ol.TTensorI_OL_Vectorize_AVecLast(TTensorI_OL(t, [ AOne, AVec ],[ [ 0, 2 ] ], [ 1, v ], [ AVecReg(isa) ]), r)),

    twrap := (self, t) >> let(isa := self.isa, v := isa.v, 
#This is OBVIOUSLY a hack. only deals with some kind of vectorization
           TTensorI_OL(t, [ AOne, AVec ],[ [ 0, 2 ] ], [ 1, v ], [ AVecReg(isa) ])
        ),
    
    print := self >> Print(self.name, "(", self.isa, ")")
));

Class(TTensorI_OL, Tagged_tSPL, rec(
    abbrevs := [ 
        (nt,g,s,v)  -> [nt,List([1..Length(g)],i->When(v[i]=1,AOne,g[i])),s,v] 
    ],

    dmn := self >> let(
        nt := self.params[1],
        sizes := self.params[4],
        List([1..Length(sizes)], i->
            TArray(nt.dmn()[i].t,sizes[i] * nt.dmn()[i].size)
        )),

    rng := self >> let(
        nt := self.params[1],
        s := self.params[3],
        v := self.params[4],
        List([1..Length(s)], i->
            TArray(nt.rng()[i].t, Product(List(s[i], a ->
                When(a=0,
                    nt.rng()[i].size,
                    v[a]
                )
            )))
        )
    ),

    isReal := self >> self.params[1].isReal(),
    doNotMeasure:=true,
    doNotSaveInHashtable:=true,
    decomposePerformance:=true,
    transpose := self >> Copy(self),
#D    tagpos :=5,
));

#Changed by Marek, looks OK
NewRulesFor(TTensorI_OL, rec(
    TTensorI_OL_Base :=rec(
        applicable := (self,t) >> not(t.hasTag(AParSMP)) and t.getTags() = t.params[1].getTags(),
        freedoms := nt -> let(
            g := nt.params[2],
            nbvars := Length(Filtered(g,x->x=APar or x=AVec)),
            [Arrangements([1..nbvars],nbvars)]
        ),
        child := (nt,freedoms) -> [nt.params[1],InfoNt(freedoms)],
        recompose := (nt,cnt,cperf) -> cperf[1]*Product(List(Filtered(Zip2(nt.params[2],nt.params[4]),l->l[1] in [APar,AVec]),x->x[2])),
        apply := function(nt,c,cnt)
            local g,s,v,perm,ind1,ind,ind2,gathers,scatters,result,z;
            g := nt.params[2];
            s := nt.params[3];
            v := nt.params[4];
            perm := cnt[2].params[1][1];
            ind1 := List([1..Length(g)], i ->
                Cond(g[i]=APar or g[i]=AVec, fBase(Ind(v[i])),
                    g[i]=AOne, fId(1),
                    ObjId(g[i])=AMul, g[i],
                    Error("PV not known!")
                )
            );
            ind := List(ind1, x -> When(ObjId(x)=AMul,ind1[x.params[1]],x));
            gathers := Cross(List([1..Length(g)], i -> let(
                kernelsize := fId(cnt[1].dmn()[i].size),
                When(g[i]=APar or (ObjId(g[i])=AMul and g[i].params[2]=APar),
                    Gath(fTensor(ind[i],kernelsize)),
                    Gath(fTensor(kernelsize,ind[i]))
                )
            )));
            scatters := Cross(List([1..Length(s)], i -> 
                ScatQuestionMark( fTensor(
                    List(s[i], y ->
                        When(y=0, 
                            fId(cnt[1].rng()[i].size),
                            ind[y]
                        )
                    )
                ))
            ));
            result := scatters*c[1]*gathers;
            ind2:=[];
            for i in [1..Length(g)] do
                if g[i]=APar or g[i]=AVec then
                    Add(ind2,ind[i]);
                fi;
            od;
            for i in [1..Length(ind2)] do
                result:=ISum(ind2[perm[i]].params[2], ind2[perm[i]].params[1], result);
            od;

            return result;
        end
    ),
#D        applicable :=(self,t) >> Length(Filtered(GetTags(t),x->ObjId(x)=AParSMP))=0 and GetTags(t)=GetTags(t.params[1])
#D,
#D            freedoms := nt -> let(g:=nt.params[2],
#D                nbvars:=Length(Filtered(g,x->x=APar or x=AVec)),
#D                [Arrangements([1..nbvars],nbvars)]),
#D            child := (nt,freedoms) -> [nt.params[1],InfoNt(freedoms)],
#D            recompose := (nt,cnt,cperf) -> cperf[1]*Product(List(Filtered(Zip2(nt.params[2],nt.params[4]),l->l[1] in [APar,AVec]),x->x[2])),
#D            apply := function(nt,c,cnt)
#D                local g,s,v,perm,ind1,ind,ind2,gathers,scatters,result,z;
#D                g:=nt.params[2];
#D                s:=nt.params[3];
#D                v:=nt.params[4];
#D                perm:=cnt[2].params[1][1];
#D                ind1:=List([1..Length(g)],
#D                    i->Cond(g[i]=APar or g[i]=AVec,fBase(Ind(v[i])),
#D                    g[i]=AOne,fId(1),
#D                    ObjId(g[i])=AMul,g[i],
#D                    Error("PV not known!")));
#D                ind:=List(ind1,x->When(ObjId(x)=AMul,ind1[x.params[1]],x));
#D                gathers:= Cross(List([1..Length(g)],
#D                        i->let(kernelsize:=fId(cnt[1].dmn()[i].size),
#D                            When(g[i]=APar or (ObjId(g[i])=AMul and g[i].params[2]=APar),
#D                                Gath(fTensor(ind[i],kernelsize)),
#D                                Gath(fTensor(kernelsize,ind[i]))))));
#D                scatters:=Cross(List([1..Length(s)],
#D                            i->ScatQuestionMark(fTensor(
#D                                    List(s[i],y->
#D                                        When(y=0,fId(cnt[1].rng()[i].size),
#D                                            ind[y]))))));
#D                result:=scatters*c[1]*gathers;
#D                ind2:=[];
#D                for i in [1..Length(g)] do
#D                    if g[i]=APar or g[i]=AVec then
#D                        Add(ind2,ind[i]);
#D                    fi;
#D                od;
#D                for i in [1..Length(ind2)] do
#D                        result:=ISum(ind2[perm[i]].params[2],ind2[perm[i]].params[1],result);
#D                od;
#D
#D                return result;
#D            end
#D            ),
        
#Changed by Marek, looks OK
    TTensorI_OL_Parrallelize_AParFirst := rec(
        applicable :=(self,t) >> 
            t.isTag(1, AParSMP)
            and 0 <> t.params[3][1][1]
            and APar = t.params[2][t.params[3][1][1]]
            and t.firstTag().params[1] = t.params[4][t.params[3][1][1]]
            and 1 = t.params[3][1][1], #t.params[3][1][1]=1 is a trick to prevent // the 2nd input because TensorGeneral breaks at the moment

        freedoms := nt -> [[1]],

        child := function (nt,freedoms)
            local PV,sizes;
            sizes := Copy(nt.params[4]);
            sizes[nt.params[3][1][1]] := 1;
            if Length(Filtered(sizes, t -> t<>1)) > 0 then
                PV:=Copy(nt.params[2]);
                PV[nt.params[3][1][1]] := AOne;
                return [TTensorI_OL(
                    DropParTag(nt.params[1]),
                    PV,
                    nt.params[3],
                    sizes,
                    Drop(nt.params[5],1)
                )];
            else
                return [ DropParTag(Copy(nt.params[1])) ];
            fi;
        end,

        apply := function(nt,c,cnt)
            local myCross,a,b,index;

            index := Ind(nt.firstTag().params[1]);

            a := List(c[1].dims()[2], x -> fId(x));
            a[nt.params[3][1][1]] := fTensor(fBase(index), a[nt.params[3][1][1]]);

            b:=List(a, x -> Gath(x));

            myCross:=Cross(b);

#              return SMPSum(GetFirstTag(nt).params[1],index, GetFirstTag(nt).params[1],ScatQuestionMark(fTensor(fBase(index), fId(c[1].dims()[1]))) *c[1]* Cross(Gath(fTensor(fBase(index), fId(c[1].dims()[2][1]))),Gath(fId(c[1].dims()[2][2]))));
            return SMPSum(
                nt.firstTag().params[1],
                index,
                nt.firstTag().params[1],
                ScatQuestionMark(fTensor(
                    fBase(index), 
                    fId(c[1].dims()[1])
                ))
                * c[1] * myCross
            );
        end
    ),
#D            applicable :=(self,t) >> FirstTagEq(t, AParSMP) and t.params[3][1][1]<>0 and 
#D               t.params[2][t.params[3][1][1]]=APar and 
#D               t.params[4][t.params[3][1][1]]=GetFirstTag(t).params[1] and 
#D               t.params[3][1][1]=1, #t.params[3][1][1]=1 is a trick to prevent // the 2nd input because TensorGeneral breaks at the moment
#D            freedoms := nt -> [[1]],
#D            child := function (nt,freedoms)
#D              local PV,sizes;
#D              sizes:=Copy(nt.params[4]);
#D              sizes[nt.params[3][1][1]]:=1;
#D              if Length(Filtered(sizes, t->t<>1))>0 then
#D                  PV:=Copy(nt.params[2]);
#D                  PV[nt.params[3][1][1]]:=AOne;
#D              return [TTensorI_OL(DropParTag(nt.params[1]),
#D                          PV,nt.params[3],sizes,Drop(nt.params[5],1))];
#D              else
#D                  return [DropParTag(Copy(nt.params[1]))];
#D              fi;
#D            end,
#D            apply := function(nt,c,cnt)
#D              local myCross,a,b,index;
#D              index:=Ind(GetFirstTag(nt).params[1]);
#D              a:=List(c[1].dims()[2],x->fId(x));
#D              a[nt.params[3][1][1]]:=fTensor(fBase(index),a[nt.params[3][1][1]]);
#D              b:=List(a,x->Gath(x));
#D              myCross:=Cross(b);
#D#              return SMPSum(GetFirstTag(nt).params[1],index, GetFirstTag(nt).params[1],ScatQuestionMark(fTensor(fBase(index), fId(c[1].dims()[1]))) *c[1]* Cross(Gath(fTensor(fBase(index), fId(c[1].dims()[2][1]))),Gath(fId(c[1].dims()[2][2]))));
#D              return SMPSum(GetFirstTag(nt).params[1],index, GetFirstTag(nt).params[1],ScatQuestionMark(fTensor(fBase(index), fId(c[1].dims()[1]))) *c[1]* myCross);
#D            end),

    #That code vectorizes the last guy if it is a AVec of the vector size
    #Hack, only works with one output
#Changed by Marek, looks OK
    TTensorI_OL_Vectorize_AVecLast :=rec(
        applicable := (self,t) >> 
            t.isTag(1, AVecReg)
            and Last(t.params[3][1])<>0 
            and AVec = t.params[2][Last(t.params[3][1])]
            and t.params[4][Last(t.params[3][1])] = t.firstTag().v,

        freedoms := nt -> [[1]],

        child := function(nt, freedoms)
            local PV, sizes;

            sizes := Copy(nt.params[4]);
            sizes[Last(nt.params[3][1])] := 1;

            if Length(Filtered(sizes, t->t<>1))>0 then
                PV:=Copy(nt.params[2]);
                PV[Last(nt.params[3][1])]:=AOne;
                return [TTensorI_OL(
                    DropVectorTag(nt.params[1]).setWrap(VOLWrap(nt.firstTag().isa)),
                    PV,
                    nt.params[3],
                    sizes,
                    Drop(nt.params[5],1)
                )];
            else
                return [ DropVectorTag(Copy(nt.params[1])).setWrap(VOLWrap(nt.firstTag().isa)) ];
          fi;
        end,

        apply := function(nt,c,cnt)
            local myCross, myScat, mydims, v;

            v := nt.firstTag().v;

            #little hack for KernelDup
            if ObjId(nt.params[1]).name <> "KernelMMMDuped" then
                myCross := Cross(List(nt.dims()[2], t ->
                    VGath_dup(fId(t), v)
                ));
                myCross._children[Last(nt.params[3][1])] :=
                    VPrm_x_I(
                        fId( nt.dims()[2][Last(nt.params[3][1])] / v ), 
                        nt.firstTag().v
                    );
            else 
                mydims := nt.dims()[2];
                mydims[1] := mydims[1] * v;
                myCross := Cross(List(mydims, t ->
                    VPrm_x_I( fId(t/v), v )
                ));
            fi;

            myScat := VScat(fId( nt.dims()[1]/v ), v );

            return myScat * VTensor_OL(c[1], v) * myCross;
        end
    ),
#D            applicable :=(self,t) >> FirstTagEq(t, AVecReg) and Last(t.params[3][1])<>0 and t.params[2][Last(t.params[3][1])]=AVec and t.params[4][Last(t.params[3][1])]=GetFirstTag(t).v,
#D            freedoms := nt -> [[1]],
#D            child := function (nt,freedoms)
#D              local PV,sizes;
#D              sizes:=Copy(nt.params[4]);
#D              sizes[Last(nt.params[3][1])]:=1;
#D              if Length(Filtered(sizes, t->t<>1))>0 then
#D                  PV:=Copy(nt.params[2]);
#D                  PV[Last(nt.params[3][1])]:=AOne;
#D              return [TTensorI_OL(DropVectorTag(nt.params[1]).setWrap(VOLWrap(GetFirstTag(nt).isa)),
#D                          PV,nt.params[3],sizes,Drop(nt.params[5],1))];
#D              else
#D                  return [DropVectorTag(Copy(nt.params[1])).setWrap(VOLWrap(GetFirstTag(nt).isa))];
#D              fi;
#D            end,
#D            apply := function(nt,c,cnt)
#D              local myCross,myScat,mydims;
#D#little hack for KernelDup
#D              if (ObjId(nt.params[1]).name<>"KernelMMMDuped") then
#D                  myCross:=Cross(List(nt.dims()[2],t->VGath_dup(fId(t),GetFirstTag(nt).v)));
#D                  myCross._children[Last(nt.params[3][1])]:=
#D                  VPrm_x_I(fId(nt.dims()[2][Last(nt.params[3][1])]/GetFirstTag(nt).v),GetFirstTag(nt).v);
#D              else 
#D                  mydims:=nt.dims()[2];
#D                  mydims[1]:=mydims[1]*GetFirstTag(nt).v;
#D                  myCross:=Cross(List(mydims,t->VPrm_x_I(fId(t/GetFirstTag(nt).v),GetFirstTag(nt).v)));
#D              fi;
#D              myScat:=VScat(fId(nt.dims()[1]/GetFirstTag(nt).v),GetFirstTag(nt).v);
#D              return myScat*VTensor_OL(c[1],GetFirstTag(nt).v)*myCross;
#D            end),

#Changed by Marek, looks OK
    TTensorI_OL_Vectorize_AParFirst :=rec(
        switch:=false,

        applicable := (self,t) >> 
            t.isTag(1, AVecReg)
            and 0 <> First(t.params[3][1], x -> true)
            and APar = t.params[2][First(t.params[3][1],x -> true)]
            and t.firstTag().v = t.params[4][First(t.params[3][1], x -> true)],

        freedoms := nt -> [[1]],

        child := function (nt,freedoms)
            local PV,outorder,theguy;

            PV := Copy(nt.params[2]);
            PV[First(nt.params[3][1],x->true)] := AVec;

            theguy := Copy(First(nt.params[3][1], x->true));
            outorder := Drop(Copy(nt.params[3][1]),1);
            Add(outorder,theguy);

            return [
                TTensorI_OL( nt.params[1], PV, [outorder], nt.params[4], nt.params[5]),
                TL(nt.rng()[1].size, nt.firstTag().v, 1, 1, nt.params[5]),
                TL(
                    nt.dmn()[First(nt.params[3][1], x -> true)].size, 
                    nt.dmn()[First(nt.params[3][1], x -> true)].size / nt.firstTag().v,
                    1,
                    1,
                    nt.params[5]
                )
            ];
        end,

        apply := function(nt,c,cnt)
            local i;

            i := Identities(nt.dmn());
            i._children[First(nt.params[3][1], x -> true)] := c[3];

            return c[2] * c[1] * i;
        end
    ),

#D            applicable :=(self,t) >> FirstTagEq(t, AVecReg) and First(t.params[3][1],x->true)<>0 and t.params[2][First(t.params[3][1],x->true)]=APar and t.params[4][First(t.params[3][1],x->true)]=GetFirstTag(t).v,
#D            freedoms := nt -> [[1]],
#D            child := function (nt,freedoms)
#D              local PV,outorder,theguy;
#D              PV:=Copy(nt.params[2]);
#D              PV[First(nt.params[3][1],x->true)]:=AVec;
#D              theguy:=Copy(First(nt.params[3][1],x->true));
#D              outorder:=Drop(Copy(nt.params[3][1]),1);
#D              Add(outorder,theguy);
#D              return [TTensorI_OL(nt.params[1],PV,[outorder],nt.params[4],nt.params[5]),TL(nt.rng()[1].size,GetFirstTag(nt).v,1,1,nt.params[5]),TL(nt.dmn()[First(nt.params[3][1],x->true)].size,nt.dmn()[First(nt.params[3][1],x->true)].size/GetFirstTag(nt).v,1,1,nt.params[5])];
#D            end,
#D            apply := function(nt,c,cnt)
#D              local i;
#D              i:=Identities(nt.dmn());
#D              i._children[First(nt.params[3][1],x->true)]:=c[3];
#D              return c[2]*c[1]*i;
#D            end),
));










