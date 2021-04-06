
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Symbolic Scalars
# ================

#F Scalars
#F =======
#F

#F The class "Scalar" represents a symbolic scalar, that is not
#F evaluated. The main motivation is to
#F deal with trigonometric expressions (e.g., cosines) efficiently.
#F The usual numbers provided by gap will be called numbers.
#F
#F A Scalar is a GAP record with the following fields mandatory
#F for all types
#F
#F   isScalar    (=true)
#F   operations  (=ScalarOps)
#F   domain      (=Scalars)
#F   val
#F
#F where val is the actual symbolic expression represented as the delayed
#F evaluation object constructed with Delay() function.
#F

#F CanBeScalar( <obj> )
#F    This functions returns true if an object can become a Scalar record.
#F    Integers and rationals can't become a Scalar record, although they
#F    are considered to be part of Scalars field. This implies that this
#F    function is *misleading*. (NOTE: function def + description)
CanBeScalar :=  obj ->
    IsFloat(obj) or IsComplex(obj) or obj in FieldElements;

Declare(ScalarsOps, IsScalar);

_abs := x->Cond(IsComplex(x), AbsComplex(x),
                IsCyc(x), AbsComplex(Complex(x)),
        AbsFloat(x));

inf_norm := mat ->
    Maximum(List(mat, r -> Sum(r, x -> Cond(IsExp(x),   _abs(x.ev()),
                                    IsValue(x), _abs(x.v), _abs(x)))));
evmat := mat ->
    List(mat, r -> List(r, x -> Cond(IsExp(x), x.ev(), IsValue(x), x.v, x)));
absmat := mat ->
    List(mat, r -> List(r, x -> Cond(IsExp(x), _abs(x.ev()), IsValue(x), _abs(x.v), _abs(x))));
remat := mat -> MapMat(mat, x->ReComplex(Complex(x)));

col := (mat, c) -> List(mat, row->[row[c]]);

InfinityNormMat := inf_norm;

RealMatComplexMat := function(mat)
    local res, row, row1, row2, r, i, e, ee;
    res := []; # list of rows
    for row in mat do
        row1 := []; row2 := [];
    for e in row do
       ee := When(IsExp(e), e.eval(), e); r := re(ee).eval(); i := im(ee).eval();
       Append(row1, [r, -i]);
       Append(row2, [Copy(i), Copy(r)]);
    od;
    Add(res, row1); Add(res, row2);
    od;
    return res;
end;

RealVMatComplexVMat := function(vmat)
    local res, row, row1, row2, r, i, e, ee, el;
    res := []; # list of rows
    for row in vmat do
        row1 := []; row2 := [];
        for el in row do
            if IsRec(el) and IsBound(el.t) and IsArrayT(el.t) then el := el.v; fi;
            if IsList(el) then
                r := [];
                i := [];
                for e in el do
                    ee := When(IsExp(e), e.eval(), e);
                    Append(r, [re(ee).eval()]);
                    Append(i, [im(ee).eval()]);
                od;
            else
                Constraint((IsExp(el) or IsValue(el)) and IsVecT(el.t) and el.t.t<>TComplex);
                r := el;
                i := el.t.value(0);
            fi;
            Append(row1, [r, -i]);
            Append(row2, [Copy(i), Copy(r)]);
        od;
        Add(res, row1); Add(res, row2);
    od;
    return res;
end;

RCMat := RealMatComplexMat;

RCMatCyc := function ( mat )
    local  res, row, row1, row2, r, i, e, ee;
    res := [];
    for row  in mat  do
        row1 := []; row2 := [];
        for e  in row  do
            ee := When(IsExp(e), e.ev(), e);
            r := Re(ee); i := Im(ee);
            Append(row1, [ r, -i ]);
            Append(row2, [ i, r ]);
        od;
        Add(res, row1); Add(res, row2);
    od;
    return res;
end;

getReIm := function(c)
    local r, i;
    if IsCyc(c) then c:=ComplexCyc(c); fi;
    if IsInt(c) or IsFloat (c) then r:=c; i:=0;
    else if IsComplex(c) then r:=ReComplex(c); i:=ImComplex(c);
    else Error("bad matrix entry type");
    fi; fi;
    return [r,i];
end;

#############################################################################
#V  Scalars  . . . . . . . . . . . . . . . . . . . . . . field of all scalars
#V  ScalarsOps  . . . . . . . . . . . . . .  operations record for the domain
##
Class(Scalars, rec(
    isDomain   := true,
    isField    := true,
    #generators := [ ]
    zero       := V(0),
    one        := V(1),
    size       := "infinity",
    char       := 0,
    field      := 0,
    isFinite   := false,
));

Scalars.operations := Class(ScalarsOps, OpsOps, rec(
    \in := ( x, Scalars ) -> IsScalar(x) or IsRat(x) or IsCyc(x) or CanBeScalar(x),
    Field  := elms -> Scalars,
    DefaultField := elms ->Scalars
));

#F IsScalar( <obj> )
#F   tests whether <obj> is a scalar and returns true in the affirmative
#F   case. Else, false is returned.
#F
IsScalar := s -> IsRec(s) and (IsExp(s) or IsValue(s) or IsLoc(s));

#F IsScalarOrNum( <obj> )
#F
IsScalarOrNum := x -> IsScalar(x) or CanBeScalar(x);

#F Scalar( <d> )
#F   adds fields common to all scalars to a record to form a scalar
#F   from the delayed evaluation object <d>
#F
Scalar := d -> When(not IsDelay(d) and IsList(d),
    List(d, Scalar),
    toExpArg(d));

# Construction macro (allows to skip D())
SS := UnevalArgs( x -> Scalar(x) );

# we use lower case function names for optical convenience in SPLs
normPi := function(r)
    # move r into the interval [0,2)
    r := 2 * (r/2 - Int(r/2));
    if r < 0 then
    r := r + 2;
    fi;
    return r;
end;

ExportBagSPL := x->x;

TranslateGapFuncToSPL := rec(
    CosPi := cc -> Print("cos(", ExportBagSPL(cc[2]), "*pi)"),
    SinPi := cc -> Print("sin(", ExportBagSPL(cc[2]), "*pi)"),
    Sqrt  := cc -> Print("sqrt(", ExportBagSPL(cc[2]), ")")
);

ExportBagSPL := function(o)
    local t, cc;
    t := BagType(o);
    cc := Children(o);
    cc := cc{[2..Length(cc)]};

    Cond(t = T_PROD, Print("(", ExportBagSPL(cc[1]), " * ", ExportBagSPL(cc[2]), ")"),
     t = T_QUO,  Print("(", ExportBagSPL(cc[1]), " / ", ExportBagSPL(cc[2]), ")"),
     t = T_SUM,  Print("(", ExportBagSPL(cc[1]), " + ", ExportBagSPL(cc[2]), ")"),
     t = T_DIFF, Print("(", ExportBagSPL(cc[1]), " - ", ExportBagSPL(cc[2]), ")"),
     t = T_FUNCCALL, When(BagType(cc[1]) = T_VAR,
                       let(f  := NameOf(cc[1]),
			   tr := TranslateGapFuncToSPL,
                          When(IsBound(tr.(f)), tr.(f)(cc),
                   Error("Don't know how to translate '", f, "' to SPL")) ),
               Error("Can't translate unnamed function to SPL")),

     t = T_CYC,  let(c := ComplexCyc(o),
                     Print("(", String(ReComplex(c)), ",", String(ImComplex(c)), ")")),
     Print(o));
    return "";
end;

# PrintScalarToSPLNC ( <scalar> )
#   prints <scalar> in spl syntax. Used for exporting (spl.g).
#
PrintScalarToSPLNC := s -> s.printSPL();

Value.printSPL := self >> spiral.spl.PrintNumberToSPLNC(self.v);
Exp.printSPL := self >> Print(self.name, "(",
    DoForAllButLast(self.args, a->Chain(a.printSPL(), Print(", "))),
    Last(self.args).printSPL(), ")");

Loc.printSPL := self >> self.print();

fdiv.printSPL := self >> Print("(", self.args[1].printSPL(), "/", self.args[2].printSPL(), ")");
mul.printSPL := self >> Print("(", self.args[1].printSPL(), "*", self.args[2].printSPL(), ")");
add.printSPL := self >> Print("(", self.args[1].printSPL(), "+", self.args[2].printSPL(), ")");
sub.printSPL := self >> Print("(", self.args[1].printSPL(), "-", self.args[2].printSPL(), ")");
neg.printSPL := self >> Print("(-", self.args[1].printSPL(), ")");

cospi.printSPL := self >> Print("cos(", self.args[1].printSPL(), "*pi)");
sinpi.printSPL := self >> Print("sin(", self.args[1].printSPL(), "*pi)");
ScalarIsCos := x -> ObjId(x) = cospi;
ScalarIsSin := x -> ObjId(x) = sinpi;
ScalarCosArg := x -> x.args[1];


#F Functions for Scalars
#F ---------------------
#F

#F EvalScalar ( <scalar>/<number> )
#F   returns the number represented by <scalar>. If a <number>
#F   is given, then <number> is returned.
#F
EvalScalar := s -> _unwrap(
    Cond(s _is funcExp,  
	    s.eval(),
	 IsScalar(s) and IsBound(s.ev), 
	    s.ev(), 
	 Eval(s)));

#F <s1> = <s2>
#F   returns true if both <s1> and <s2> are scalars, not of
#F   type "float" or "complex", and represent the same number.
#F
#Ops.\= := function ( s1, s2 )
#    return IsScalar(s1) and IsScalar(s2) and
#           s1.ev() = s2.ev();
#end;


#Exp.domain := Scalars;
#Value.domain := Scalars;
#Loc.domain := Scalars;
