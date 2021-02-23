
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################################
# If RulesXYOffset correctly takes out base addresses out of H into RecursStep
# we can use the following hack to pass into the function only the stride
#
# H.codeletParNums := [4];
# H.signature := self >> [IntVar("s")];
# H.mkCodelet := self >> let(ss := self.signature(), H(self.params[1], self.params[2], 0, ss[1]));
#
#########################################################################

CollectRecursSteps := s -> Collect(s, RecursStep);

#F Shape(<obj>) - returns a LISP-style structural list, i.e. add(1,2) -> [add, 1, 2]
#F
Shape := x-> Cond(IsRec(x) and IsBound(x.shape), x.shape(),  x);

ClassSPL.shape := self >> Concatenation([ObjId(self)],
    List(self.rChildren(), Shape));

Exp.shape := self >> Concatenation([ObjId(self)],
    List(self.rChildren(), Shape));

RTWrap.shape := self >> self.rt.node.shape();

#F CodeletShape(<obj>) - returns a LISP-style structural list for a codelet
#F    In this case parameters that are passed into a codelet functions do not
#F    show up, i.e. Scat(H(16,2,0,1))*DFT(16)*... -> [Compose, [Scat, H], DFT, ...]
#F
CodeletShape := x-> Cond(IsRec(x) and IsBound(x.codeletShape), x.codeletShape(),  x);
ClassSPL.codeletShape := self >> Concatenation([ObjId(self)],
    List(self.rChildren(), CodeletShape));
FuncClassOper.codeletShape := ClassSPL.codeletShape;
RTWrap.codeletShape := self >> CodeletShape(self.rt.node);
NonTerminal.codeletShape := self >> Concatenation(
    [ObjId(self)],
    List(self.params, CodeletShape),
    When(self.transposed, ["T"], []));

#F CodeletSignature(<obj>) - codelet function signature
#F
CodeletSignature := x-> Cond(IsRec(x) and IsBound(x.signature), x.signature(),
    IsRec(x), Error(".signature() field missing (objid = ", ObjId(x), ")"), []);
ClassSPL.signature := self >> Concatenation(List(self.rChildren(), CodeletSignature));
FuncClassOper.signature := ClassSPL.signature;
RTWrap.signature := self >> CodeletSignature(self.rt.node);
Value.signature := self >> [];

#F CodeletParams(<obj>) - codelet function call params
#F
CodeletParams := x-> Cond(IsRec(x) and IsBound(x.codeletParams), x.codeletParams(),
    IsRec(x), Error(".codeletParams() field missing (objid = ", ObjId(x), ")"), []);
ClassSPL.codeletParams := self >> Concatenation(List(self.rChildren(), CodeletParams));
FuncClassOper.codeletParams := ClassSPL.codeletParams;
RTWrap.codeletParams := self >> CodeletParams(self.rt.node);
Value.codeletParams := self >> [];

#F MkCodelet(<obj>) - generalize <obj> by changing codelet params to be variables
#F
MkCodelet := x -> Cond(IsRec(x) and IsBound(x.mkCodelet), x.mkCodelet(),
    IsRec(x), Error(".mkCodelet() field missing (objid = ", ObjId(x), ")"), x);
#ClassSPL.codeletParams := self >> Concatenation(List(self.rChildren(), CodeletParams));
ClassSPL.mkCodelet := self >> ApplyFunc(ObjId(self),
    List(self.rChildren(), MkCodelet));
FuncClassOper.mkCodelet := BaseOperation.mkCodelet;
RTWrap.mkCodelet := self >> self;
NonTerminal.mkCodelet := self >> self;
Value.mkCodelet := self >> self;

#F CodeletName(<shape>)
#F
CodeletName := shape -> Cond(
    IsInt(shape),
        When(shape < 0, Concat("_n", String(-shape)), Concat("_",String(shape))),
    IsRat(shape),
        Concat(When(shape < 0, "_n", "_"), StringInt(AbsInt(Numerator(shape))), "d", StringInt(Denominator(shape))),
    IsValue(shape),
        CodeletName(shape.v),

    IsRec(shape), Concat(When(IsBound(shape.codeletNameNo_), "", "_"),
    When(IsBound(shape.codeletName), shape.codeletName, shape.name)),
    IsString(shape),
        shape,

    IsList(shape) and not (Length(shape) > 2 and IsRec(shape[1]) and IsBound(shape[1].codeletNameInfix)),
        When(Length(shape)=2 and not (IsRec(shape[2]) or IsList(shape[2])),
         Concat("", CodeletName(shape[1]), String(shape[2])),
         Concat("", Concatenation(List(shape, CodeletName)))),

    IsList(shape), let(cc := shape{[2..Length(shape)-1]}, op := shape[1].codeletName,
        Concat(
        ConcatList(cc, x->Concat(CodeletName(x), "_", op)), CodeletName(Last(shape)))),

    IsFunc(shape), "GAPFunc",

    Concat("_", String(shape)));

####################################################################################
Compose.codeletNameNo_ := true;
Scat.codeletNameNo_ := true;
fTensor.codeletNameNo_ := true;
Compose.codeletName := "";
fId.codeletName := "I";
fTensor.codeletName := "x";
Diag.codeletName := "D";
RCDiag.codeletName := "RD";
Gath.codeletName:= "G";
Scat.codeletName:= "S";

H.codeletShape     := self >> ObjId(self);
FList.codeletShape := self >> Concatenation([ObjId(self)], self.list);
FData.codeletShape := self >> ObjId(self);
FDataOfs.codeletShape := self >> ObjId(self);
#RCDiag.codeletShape:= self >> ObjId(self);

H.signature     := self >> List(["b", "s"], IntVar);
RM.signature    := self >> List(["N", "phi", "g"], IntVar);
FData.signature := self >> [ var.fresh_t("D", TPtr(self.var.t.t)) ];
FDataOfs.signature := self >> [ var.fresh_t("D", TPtr(self.var.t.t)) ];
FList.signature := self >> [];

FuncClass.codeletParams := self >> self.params{self.codeletParNums};
FuncClass.codeletParNums := [];
FuncClass.mkCodelet := self >> let(sig := self.signature(), ApplyFunc(ObjId(self),
    List([1..Length(self.params)], i -> let(parnums := self.codeletParNums,
    When(i in parnums, sig[Position(parnums,i)], self.params[i])))));
Sym.mkCodelet := FuncClass.mkCodelet;
Sym.codeletParNums := [];

H.codeletParNums    := [3, 4];
RM.codeletParNums   := [1, 3, 4];

FList.codeletParams := self >> [];
FData.codeletParams := self >> [self.var];
FData.mkCodelet := self >> FDataOfs(self.signature()[1], self.domain(), 0);

FDataOfs.codeletParams := self >> When(self.ofs=0, [self.var], [self.var+self.ofs]);
FDataOfs.mkCodelet := self >> FDataOfs(self.signature()[1], self.len, 0);
FList.mkCodelet := self >> self;

#
# Here we set some fields required by libgen for objects defined in transforms.*
# This system will change in the future to require less fields, and be
# more streamlined.
#
# Note: This code fragment depends on transforms. NOTE.
#
DFT.codeletShape := self >> When(self.params[2]=1, [ObjId(self), self.params[1]],
                                  [ObjId(self), self.params[1], self.params[2]]);

BHD.codeletShape := self >> ObjId(self);
BHD.mkCodelet    := self >> self; # NOTE?

Twid.signature  := self >> [ ];
Twid.codeletShape  := self >> [ObjId(self), self.params[1], self.params[2], self.params[3]];
Twid.codeletParams := self >> [];

BH.signature    := self >> List(["R", "b", "s"], IntVar);
BH.codeletShape    := self >> ObjId(self);
BH.codeletParNums  := [2, 4, 5];

RCData.signature  := self >> self.func.signature();
RCData.codeletShape  := self >> [ObjId(self), CodeletShape(self.func)];
RCData.codeletParams := self >> self.func.codeletParams();
RCData.mkCodelet := self >> ObjId(self)(self.func.mkCodelet());

FConj.signature  := self >> self.func.signature();
FConj.codeletShape  := self >> [ObjId(self), CodeletShape(self.func)];
FConj.codeletParams := self >> self.func.codeletParams();
FConj.mkCodelet := self >> ObjId(self)(self.func.mkCodelet());

Typ.signature := self >> [];
Typ.codeletShape := self >> [ObjId(self)];
Typ.codeletParams := self >> [];
Typ.mkCodelet := self >> self;
