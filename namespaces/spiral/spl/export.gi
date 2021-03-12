
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ExpandRotationsSPL33( <spl> )
#F   descends <spl> and replaces rotations by a product of three
#F   matrices such that multiplication with a vector requires
#F   3 adds and 3 mults rather than 2 adds and 4 mults. More
#F   precisely:
#F      x  y        1  1  0                        1 0
#F     -y  x   -->  0 -1  1  * diag(x-y, y, x+y) * 1 1
#F                                                 0 1
#F   Exceptions are the cases y = x and y = -x (angles +-pi/4)
#F   which are replaced by (2 adds and 2 mults)
#F
#F      x * F_2 * (1,2)   # y = x
#F      x * (1,2) * F_2   # y = -x
#F
ExpandRotationsSPL33 := S -> SubstTopDownNR(S, Rot, x->x.expandWinograd()); 


#F ExpandRotationsSPL33a( <spl> )
#F   descends <spl> and replaces rotations by a product of three
#F   matrices such that multiplication with a vector requires
#F   3 adds and 3 mults rather than 2 adds and 4 mults. More
#F   precisely:
#F     x y      1 (1-x)/y   1 0   1 (1-x)/y
#F    -y x      0   1       y 1   0   1
#F   Exceptions are the cases y = x and y = -x (angles +-pi/4)
#F   which are replaced by (2 adds and 2 mults)
#F
#F      x * F_2 * (1,2)   # y = x
#F      x * (1,2) * F_2   # y = -x
#F
ExpandRotationsSPL33a := S -> SubstTopDownNR(S, Rot, x->x.expandLifting()); 


#F ExpandRotationsSPL24( <spl> )
#F   descends <spl> and replaces rotations by the corresponding matrix,
#F   which requires (the maximum of) 2 additions and 4 multiplications.
#F
ExpandRotationsSPL24 := S -> SubstTopDownNR(S, Rot, x->x.expandDef()); 

ExpandRotationsSPLRandom := S -> SubstTopDownNR(S, Rot, x->x.expandRandom()); 


#F ExpandMonomialSPL( <spl> )
#F   descends <spl> and replaces monomials by products perm * diag.
#F
ExpandMonomialSPL := 
    S -> SubstTopDownNR(S, @.cond(s->ObjId(s)=Mon),
		 s-> Perm(s.element.perm, Length(s.element.diag)) *
                     Diag(s.element.diag));


#F The Exporting Functions
#F -----------------------
#F

#F ExpandRotationsSPL( <spl> )
#F
ExpandRotationsSPL := function ( S )
    Constraint(IsSPL(S));

    # EXPAND_ROTATIONS is one of (see above)
    # - ExpandRotations33      sums * diag * sums
    # - ExpandRotations33a     3 lifting steps (fused mult/add)
    # - ExpandRotations24      by definition
    if EXPAND_ROTATIONS = 1 then  S := ExpandRotationsSPL33(S);
    elif EXPAND_ROTATIONS = 2 then S := ExpandRotationsSPL33a(S);
    elif EXPAND_ROTATIONS = 3 then S := ExpandRotationsSPL24(S);
    elif EXPAND_ROTATIONS = 4 then S := ExpandRotationsSPLRandom(S);
    else Error("invalid value for EXPAND_ROTATIONS (see Doc(EXPAND_ROTATIONS))");
    fi;
    
    return S;
end;


# PrintNumberToSPLNC( <x> )
#   prints the scalar <x> in SPL syntax.

PrintNumberToSPLNC := function ( x )
  local precision, p, a, extra, c, n;

  precision := 16;

  # scalar case
  if IsScalar(x) then
    PrintScalarToSPLNC(x);
    return;
  # integers
  elif IsInt(x) then
    if x < 0 then Print("(", x, ".0)");
    else Print(x, ".0");
    fi;
    return;
  # rationals
  elif IsRat(x) then
    if x < 0 then Print("(", x, ")");
    else Print(x);
    fi;
    return;
  # floats
  elif IsDouble(x) then
      if x < 0 then Print(StringDouble("(%.17e)", x));
      else Print(StringDouble("%.17e", x));
      fi;
      return;
  # complex numbers
  elif IsComplex(x) then
      Print(StringComplex("(%.17e, %.17e)", x)); 
      return;
  fi;

  # sqrt of a rational and real
  if x = GaloisCyc(x, -1) and IsRat(x^2) then
    if x = -Sqrt(x^2) then
      Print("(-sqrt(");
      PrintNumberToSPLNC(x^2);
      Print("))");
    else
      Print("sqrt(");
      PrintNumberToSPLNC(x^2);
      Print(")");
    fi;
    return;
  fi;

  # x = a/b * cos(c*pi/d)
  p := RecognizeCosPi(x);
  if p <> false then
    extra := false;
    if p[1] < 0 then
      Print("(");
      extra := true;
    fi;
    if p[1] = 1 then
      Print("cos(");
    elif p[1] = -1 then
      Print("-cos(");
    else
      Print(p[1], "*cos(");
    fi;

    a := Numerator(p[2]);
    if a = 1 then
      Print("pi/", Denominator(p[2]), ")");
    elif a = -1 then
      Print("-pi/", Denominator(p[2]), ")");
    else
      Print(Numerator(p[2]), "*pi/", Denominator(p[2]), ")");
    fi;

    if extra then
      Print(")");
    fi;
    return;
  fi;

  # x = sqrt(2) * a/b * cos(c*pi/d)
  # this captures sum of two cosines/sines
  # the reason to include this is that the
  # substitution rule for rotations (see ExpandRotationsSPL33 or 24)
  # creates such numbers
  p := RecognizeCosPi(x/Sqrt(2));
  if p <> false then
    extra := false;
    if p[1] < 0 then
      Print("(");
      extra := true;
    fi;

    a := Numerator(p[1]);
    if a = 1 then
      Print("sqrt(2)");
    elif a = -1 then
      Print("-sqrt(2)");
    else
      Print(a, "*sqrt(2)");
    fi;
    if not IsInt(p[1]) then
      Print("/", Denominator(p[1]));
    fi;
    Print("*cos(");

    a := Numerator(p[2]);
    if a = 1 then
      Print("pi/", Denominator(p[2]), ")");
    elif a = -1 then
      Print("-pi/", Denominator(p[2]), ")");
    else
      Print(Numerator(p[2]), "*pi/", Denominator(p[2]), ")");
    fi;

    if extra then
      Print(")");
    fi;
    return;
  fi;

  # x = b/(a* CosPi(c/d))
  p := RecognizeCosPi(1/x);
  if p <> false then
    extra := false;
    if p[1] < 0 then
      Print("(-");
      extra := true;
    fi;
    Print(Denominator(p[1]), "/(", AbsInt(Numerator(p[1])), "*cos(");

    a := Numerator(p[2]);
    if a = 1 then
      Print("pi/", Denominator(p[2]), ")");
    elif a = -1 then
      Print("-pi/", Denominator(p[2]), ")");
    else
      Print(Numerator(p[2]), "*pi/", Denominator(p[2]), ")");
    fi;
    Print(")");

    if extra then
      Print(")");
    fi;
    return;
  fi;

  # x is a root of unity
  # note that w(n) (in spl) is E(n)^(n-1)
  a := NofCyc(x);
  if x^(2*a) = 1 then
    c := CoeffsCyc(x, a);
    n := PositionProperty(c, i -> i <> 0);
    if c[n] = 1 then
      if n = 2 then # x = E(a)
        Print("w(", a, ")");
      else
        Print("w(", a, ", ", n - 1, ")");
      fi;
    else # c[n] = -1
      if n = 2 then # x = -E(a)
        Print("(-w(", a, "))");
      else
        Print("(-w(", a, ", ", n - 1, "))");
      fi;
    fi;
    return;
  fi;

  # x is real
  if x = GaloisCyc(x, -1) then
       PrintNumberToSPLNC(ReComplex(Complex(x)));
  else 
       PrintNumberToSPLNC(Complex(x));
  fi;
end;
