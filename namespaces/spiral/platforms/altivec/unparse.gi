
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Cell Unparser
# Author: schellap

@TInt := @.cond(x->x.t=TInt);
@TRealOrInt := @.cond(x->x.t=TInt or x.t=TReal);
@TReal := @.cond(x->x.t=TReal);
@TVect := @.cond(x->ObjId(x.t)=TVect);

Class(altivecUnparser, CMacroUnparserProg, rec(
    # str = format string with $n for arguments ($1 = first argument)
    # list of arguments , arguments are printed using this unparser
    printf := (self, str, args) >> ApplyFunc(PrintEvalF, Concatenation([str],
        List(args, a->(When(IsFunc(a), a, ()->self(a,0,0)))))),

    # -----------------------------
    # ISA independent constructs
    # -----------------------------
    nth :=  (self, o, i, is) >> self.printf("$1[$2]", [o.loc, o.idx]),
    fdiv := (self, o, i, is) >> self.printf("((($3)$1) / $2)", Concatenation(o.args, [self.opts.vector.isa.ctype])),
    fdiv := (self, o, i, is) >> self.printf("($1 / $2)", o.args),
    idiv := (self, o, i, is) >> self.printf("($1 / $2)", o.args),
    imod := (self, o, i, is) >> self.printf("($1 % $2)", o.args),

    # This is the type used for declarations of vector variables
    ctype := (t, isa) -> Cond(
        #t in [TDouble, TVect(TDouble, 1)], "float",
        t = TReal,              isa.ctype, #evals to "float" or "double"
        t = TVect(TReal, 2),    isa.vtype, #evals to "vector double"
        t = TVect(TReal, 4),    isa.vtype,
        t = TVect(TReal, 8),    isa.vtype,
        Error("Unparser doesn't know how to declare type: ", t)
    ),

    Value := (self, o, i, is) >> Cond(
        o.t = TReal,
           let(v := When(IsCyc(o.v), ReComplex(Complex(o.v)), Double(o.v)), When(v<0, Print("(", v, ")"), Print(v))),
        o.t = TInt,         
           When(o.v < 0, Print("(", o.v, ")"), Print(o.v)),
        ObjId(o.t)=TVect,
           Print("((", self.ctype(o.t, self.opts.vector.isa), "){", self.infix(o.v, ", "), "})"),
        #NOTE: Adding this for parallel cell, but this shouldn't be needed, right?
       IsArray(o.t),
           Print("{", self.infix(o.v, ", "), "}")
    ),

    vdup         := (self, o, i, is) >> self.printf("vec_splat( ((vector float){$1, $1, $1, $1}), 0)", [o.args[1]]),

    # Declarations
    TVect := (self, t, vars, i, is) >> let(ctype := self.ctype(t, self.opts.vector.isa), 
              Print(ctype, " ", self.infix(vars, ", "))),

    TReal := ~.TVect, 

    TInt := (self, t, vars, i, is) >> Print("int ", self.infix(vars, ", ")),

    # Arithmetic
    # --------------------------------
    # -- mul -- 
    mul := (self, o, i, is) >> CondPat(o, 
    [mul, @TReal, @TVect], 
        Cond(self.opts.vector.isa.v = 2,
            self.printf("vec_madd(((vector $1){$2,$2}), $3, (vector $1)(0))",       [self.ctype(o.args[1].t, self.opts.vector.isa), o.args[1], o.args[2]]),
        self.opts.vector.isa.v = 4,
            self.printf("vec_madd(((vector $1){$2,$2,$2,$2}), $3, (vector $1)(0))", [self.ctype(o.args[1].t, self.opts.vector.isa), o.args[1], o.args[2]]),
        Error("Don't know how to unparse vector arch of length: ", self.opts.vector.isa.v)
        ),
    [mul, @TVect,   @TVect], 
        self.printf("vec_madd($1, $2, (vector float)(0) )",              [o.args[1], o.args[2]]),
    [mul, @TRealOrInt, @TRealOrInt],  
        self.printf("($1 * $2)", o.args),
    Error("Don't know how to unparse <o>. Unrecognized type combination")),

    # -- add -- 
    add := (self, o, i, is) >> When(Length(o.args) > 2, 
       self(_computeExpType(add(o.args[1], _computeExpType(ApplyFunc(add, Drop(o.args, 1))))), i, is), 
       CondPat(o, 
           [add, @TVect,   @TVect], 
              self.printf("vec_add($1, $2)", [o.args[1], o.args[2]]),
           #[add, @TRealOrInt, @TRealOrInt],  
           [add, @, @],  
              self.printf("($1 + $2)", o.args),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
       )
    ),

    # -- sub -- 
    sub := (self, o, i, is) >> When(Length(o.args) > 2, 
       self(_computeExpType(sub(o.args[1], _computeExpType(ApplyFunc(sub, Drop(o.args, 1))))), i, is), 
       CondPat(o, 
           [sub, @TVect,   @TVect], 
              self.printf("vec_sub($1, $2)", [o.args[1], o.args[2]]),
           [sub, @TRealOrInt, @TRealOrInt],  
              self.printf("($1 + $2)", o.args),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
       )
    ),

    # -- neg -- 
    neg := (self, o, i, is) >> CondPat(o,
       [neg, @TVect],
           self(o.t.value(Replicate(o.t.size,0)) - o.args[1], i, is),
       self.printf("(-$1)", o.args)
    ),

    vpack := (self, o, i, is) >> Print(self.opts.vector.isa.vconstv, "{", self.infix(o.args, ", "), "}"),

    vunpackhi_4x32f_av := (self, o, i, is) >> self.printf("vec_mergeh($1, $2)", [o.args[1], o.args[2]]),
    vunpacklo_4x32f_av := (self, o, i, is) >> self.printf("vec_mergel($1, $2)", [o.args[1], o.args[2]]),

   vparam_av := (self, o, i, is) >> 
      Print("((vector unsigned char){", PrintCS(prep_perm_string(o.p)), "})"),

   vzero_4x32f := (self, o, i, is) >> Print("((", self.opts.vector.isa.vtype ,"){", PrintCS(List([1..self.opts.vector.isa.v], i->0)), "})"),

   # ----------------------------------------------------------------------------------
   # ISA specific : altivec_4x32f
   # ----------------------------------------------------------------------------------
   vperm_4x32f := (self, o, i, is) >>
     self.printf("vec_perm($1, $2, $3)", o.args),

   vuperm_4x32f := (self, o, i, is) >>
     self.printf("vec_perm($1, $1, $2)", o.args),

));
