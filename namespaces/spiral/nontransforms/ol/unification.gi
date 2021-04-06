
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


DimLength:=function(l)
  if IsList(l) then
    return Length(l);
  else
    return 1;
  fi;
end;



Hacktype:=function(x)
  if (ObjId(x)=TPtr or ObjId(x)=TArray) then
      return x.t;
  else
      return x;
  fi;
end;

Unify:=function(type,lower)

   if lower=TUnknown then
      return type;
   else
     if type=lower or type=TUnknown then
        return lower;
     else 
        if (type=TReal and lower=TComplex) or (type=TComplex and lower=TReal) then
           return TComplex;
	elif (ObjId(lower)=TVect and lower.t=type) then
	    return lower;
	elif (ObjId(type)=TVect and type.t=lower) then
	    return type;
        else
           Error("Type Unification impossible between ",type, " and ",lower, "\n");
        fi;
     fi;
   fi;
end;

Class(Unification, Visitor, rec(
	Multiplication:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	   o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
        end,
        ICMultiplication:=meth(self,o,input)
          self.Multiplication(o,input);
        end,
        Glue:=meth(self,o,inputs)
         local i,t;
         t := List(inputs,l->Hacktype(l));
         for i in [1..Length(t)] do
          o.TType[i]:=Unify(o.TType[i],t[i]);
         od;
        end,
       Split:=meth(self,o,inputs)
          o.TType[1]:=Unify(o.TType[1],Flat([inputs])[1].t);
       end,


     Cross:=meth(self,o,inputs) 
        local counter,sl1,dim1,dimprev;
        dimprev := 0;
        for counter in [1..Length(o._children)] do
           dim1 := DimLength(o._children[counter].dims()[2]);
           sl1 := SubList(inputs,dimprev+1,dim1+dimprev);
           self(o._children[counter],sl1); 
           dimprev := dim1;
        od;
        #dim1 := DimLength(o._children[1].dims()[2]);
        #dim2 := DimLength(o._children[2].dims()[2]);
        #sl1 := SubList(inputs,1,dim1);
        #sl2 := SubList(inputs,dim1+1,dim1+dim2);      
        #self(o._children[1],sl1);
        #self(o._children[2],sl2); 
     end,
    
    Addition:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	   o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
    end,

    Subtraction:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	   o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
    end,

    #Hacks
    Table := meth(self, o, inputs)
	o.TType:=TInt;
    end,
    
    TableMRC := meth(self, o, inputs)
	o.TType:=TInt;
    end,
    
    TableSC := meth(self, o, inputs)
        o.TType:=TInt;
    end,

    SUMAcc := meth(self, o, inputs)
	self(o._children[1],inputs);
    end,

    Wire := meth(self, o, inputs)
        local i, t;
        t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(o.rng())]  do
	   o.TType[i]:=Unify(o.TType[1],TInt);
	od;
    end,

    Or:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	    o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
    end,

    And:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	    o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
    end,

    Equal:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	    o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
     end,

     NotEqual:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	    o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
     end,

    ExclusiveOr:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	    o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
     end,
    
    Minimums:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	    o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
     end,

     Maximums:=meth(self,o,inputs)
	local i,t;
	t:= List(inputs,l->Hacktype(l));
	for i in [1..Length(t)]  do
	    o.TType[i]:=Unify(o.TType[i],t[i]);
	od;
     end,

     AppendToList := (self, o, inputs) >> skip(),

     MoveToNext := (self, o, inputs) >> skip(),

     GathNeighbours := (self, o, inputs) >> skip(),

     GathOne := (self, o, inputs) >> skip(),

     AssignOneFromNine := (self, o, inputs) >> skip(),

     ConstZBitPlane := (self, o, inputs) >> skip(),

     MagPlane := (self, o, inputs) >> skip(),

     MagCube := (self, o, inputs) >> skip(),
     
     SignPlane := (self, o, inputs) >> skip(),

     CUPCode := (self, o, inputs) >> skip(),

     SetStatus := (self, o, inputs) >> skip(),

     EBCond := meth(self,o,inputs)
        self.(o.A.name)(o.A,inputs);
	self.(o.B.name)(o.B,inputs);
	self.(o.C.name)(o.C,inputs);
     end,

     EBICompose := meth(self, o, inputs)
        self.(o.C.name)(o.C, inputs);
     end,

     EBEncoded := (self, o, inputs) >> skip(),

     EBNoop := (self, o, inputs) >> skip(),

     RLCoding := (self, o, inputs) >> skip(),

     sppPosMod4 := (self, o, inputs) >> skip(),

     sppFalse := (self, o, inputs) >> skip(),

     sppPosMove2 := (self, o, inputs) >> skip(),

     EBLink := (self, o, inputs) >> skip(),

#MQ:

     CodeLPS := (self, o, inputs) >> skip(),
     
     MQbitStuffing := (self, o, inputs) >> skip(),

     MQnobitStuffing := (self, o, inputs) >> skip(),

     mqBoutClt := (self, o, inputs) >> skip(),

     mqBoutBeq := (self, o, inputs) >> skip(),

     mqBoutCand := (self, o, inputs) >> skip(),

     mqBoutBpp := (self, o, inputs) >> skip(),

     RenormalizationENC := (self, o, inputs) >> skip(),

     mqRenAlt := (self, o, inputs) >> skip(),

     mqRenCTeq := (self, o, inputs) >> skip(),

     MQCond := (self, o, inputs) >> skip(),

     MQDoWhile := (self, o, inputs) >> skip(),

     MQNoop := (self, o, inputs) >> skip(),

     MQLink := (self, o, inputs) >> skip(),

     mqMPSpre := (self, o, inputs) >> skip(),

     mqMPSf := (self, o, inputs) >> skip(),

     mqMPSt := (self, o, inputs) >> skip(),

     mqFlushRegpre := (self, o, inputs) >> skip(),

     mqFlushRegmed := (self, o, inputs) >> skip(),

     mqFlushRegpost := (self, o, inputs) >> skip(),

     MQInit := (self, o, inputs) >> skip(),

     MQBranch := (self, o, inputs) >> skip(),
     
     MQRetrieve := (self, o, inputs) >> skip(),

     MQLoop := (self, o, inputs) >> skip(),

     Const:=meth(self,o,inputs)
         local t;
         t:= List(inputs,l->Hacktype(l));
	 o.TType[1]:=Unify(o.TType[1],TInt);
     end,

     #NOTE: move to ebcot/unification.gi
     COND_OL := (self, o, inputs) >> skip(),
  
     Cross:=meth(self,o,inputs) 
        local i;
        for i in [1..Length(o._children)] do
	     self(o._children[i],inputs[i]);
        od;
     end,

     ISum:=meth(self,o,inputs) self(o._children[1],inputs); end,
     SMPSum:=meth(self,o,inputs) self(o._children[1],inputs); end,
     ISumAcc:=meth(self,o,inputs) self(o._children[1],inputs); end,
     Compose:=meth(self,o,inputs)
        local ch, numch,counter,z;
	ch := o.children();
	numch := Length(ch);
        for counter in [0..numch-1] do
           self(ch[numch-counter],inputs);
           inputs:=ch[numch-counter].rng();
#	   Print("after step ",counter," Compose is ",inputs,"\n");
        od;
     end,

     VTensor_OL:=meth(self,o,inputs)
       local inp;
       inp:=Copy(inputs);
       inp:=List(inp,x->Cond(ObjId(x)=TPtr,TPtr(TVect(x.t,o.vlen)),ObjId(x)=TArray,TArray(TVect(x.t,o.vlen),x.size/o.vlen),Error("Trying to unify a VTensor with something weird. It should work but i(Fred) just don't know how to do it right now.")));
        self(o._children[1],inp);
     end,
     VTensor:=meth(self,o,input)
       local inputs;
       inputs:=StripList(Copy(input));
       self(o._children[1],Cond(ObjId(inputs)=TPtr, [TPtr(TVect(inputs.t,o.vlen))],ObjId(inputs)=TArray,[TArray(TVect(inputs.t,o.vlen),inputs.size/o.vlen)],Error("Trying to unify a VTensor with something weird. It should work but i(Fred) just don't know how to do it right now.")));
      end,
     BlockVPerm:=meth(self,o,inputs)
            self(o._children[1],[Cond(ObjId(inputs[1])=TArray,TArray(inputs[1].t,inputs[1].size/o.n),"Not handled")]);
      end,

     ScatInit:=meth(self,o,inputs)         
         self(o._children, inputs); 
     end,

     
     VScat:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
#     VScat_sv:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     VPrm_x_I:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     VGath:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     VGath_dup:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     VScatAcc:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     ScatAcc:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     ICScatAcc :=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     Scat:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     Gath:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     Blk:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     VReplicate:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     VHAdd:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     Prm:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     VPerm:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     I:=meth(self,o,inputs) o.TType:=Unify(o.TType,Hacktype(Flat([inputs])[1])); end,
     SUM:=meth(self,o,inputs) DoForAll(o._children,i->self(i,inputs)); end,
     L:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
     BB:=meth(self,o,inputs) self(o._children[1],inputs); end,
     LinkIO:=meth(self,o,inputs) self(o.rChildren()[1],inputs); end,
     Buf:=meth(self,o,inputs) self(o._children[1],inputs); end,
     Inplace:=meth(self,o,inputs) self(o._children[1],inputs); end,
     SymSPL:=meth(self,o,inputs) self(o._children[1],inputs); end,
     NoPull:=meth(self,o,inputs) self(o._children[1],inputs); end,

     RecursStepCall:=meth(self,o,inputs) o.TType:=Unify(o.TType,Flat([inputs])[1].t); end,
 ));

