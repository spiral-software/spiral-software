
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


saveComment := function(r, fld)
    local str, last;
    str := CommentBuffer();
    last := Position(List(str), "");
    ClearCommentBuffer();
    r.(fld) := str{[3..last-3]};
end;


Declare(AsmX86Unparser);

Unparse := function(code, unparser, i, is)
    local o, res;
    # D($(..)) magic allows to deal with functions that do not return a value. 
    # In this case res := D(_bag_0) without gap complaining. 
    # We need this so that cx_leave runs before the function returns
    if not IsBound(unparser.cx) then unparser := WithBases(unparser, rec(cx := empty_cx())); fi;
    if IsRec(code) then
    cx_enter(unparser.cx, code);
        o := ObjId(code);
    if IsBound(unparser.(o.name)) then
            res := D($(unparser.(o.name)(code, i, is)));

    elif IsBound(o.unparse) and IsBound(unparser.(o.unparse)) then
            res := D($(unparser.(o.unparse)(code, i, is)));

    else return Error("Cannot unparse <o>. Unparser ", unparser, " does not have field '",
                o.name, "'", When(IsBound(o.unparse), Concat(" or ",o.unparse), ""));
    fi;
    else
    res := D($(unparser.atomic(code, i, is)));
    fi;
    cx_leave(unparser.cx, code);
    return $res;
end;

Class(Unparser, rec(
   gen := meth(self, subname, o, opts)
        local oo;
        # using WithBases prevents mem leaks, by avoiding of keeping extra state around
    self := WithBases(self, rec(opts := opts, cx := empty_cx())); 
        oo := self.preprocess(o);
        Print(
            self.header(subname, oo),
            Unparse(oo, self, 4, 2),
            self.footer(subname, oo)
        );
    end,

   __call__ := (self, o, i, is) >> Unparse(o,self,i,is),
   fileinfo := (self,opts) >> Print(""),

    # str = format string with $n for arguments ($1 = first argument)
    # list of arguments , arguments are printed using this unparser
    printf := (self, str, args) >> ApplyFunc(PrintEvalF, Concatenation([str],
        List(args, a -> 
        Cond(IsFunc(a), a, 
         () -> Cond(IsType(a), self.declare(a, [], 0, 0), self(a, 0, 0)))))),

    # printf with initial indentation
    indprintf := (self, i, is, str, args) >> Print(Blanks(i),
        ApplyFunc(PrintEvalF, Concatenation([str],
                List(args, a-> Cond(IsFunc(a), a, 
         () -> Cond(IsType(a), self.declare(a, [], i+is, is), self(a, i+is, is))))))),

    prefix := meth(self, f, lst)
        local first, c;
        Print(f, "(");
        first := true;
        for c in lst do
            if first then first := false; else Print(", "); fi;
            self(c,0,4);
        od;
        Print(")");
    end,


    infixbreak := 8,

    #F infix(<lst>, <sep>, <i>)
    #F infix takes 2 - 3 arguments. the 3rd argument specifies the number of spaces to indent, otherwise 0
    #F
    infix := meth(arg)
        local self, lst, sep, i, count, c, first;
        if not Length(arg) in [3, 4] then
            Error("Usage: infix(<lst>, <sep>, [<i>])\n");
        fi;

        if Length(arg)=3 then
            [self, lst, sep] := arg;
            i := 0;
        else
            [self, lst, sep, i] := arg; 
        fi;

        count := 0;
        first := true;
        for c in lst do
            if (count > 0) then
                if ((count mod self.infixbreak) = 0) then
                    Print(sep, "\n",Blanks(i+4));
                    first := true;
                fi;
				if not first then
                    Print(sep);
                fi;
            fi;
            count := count + 1;
            self(c,0,4);
            first := false;
        od;
    end,

    # finfix( <list>, <func>, <separator> ) - applying <func> to each element of <list>
    # and printing <separator> in between.

    finfix := meth(self, lst, func, sep)
        local first, c;
        first := true;
        for c in lst do
            if first then first := false; else Print(sep); fi;
            func(c);
        od;
    end,

    # pinfix(<lst>, <sep>), similar to infix, but parenthesizes the expression
    pinfix := meth(self, lst, sep)
        Print("(");
        self.finfix(lst, c -> self(c, 0, 4), sep);
        Print(")");
    end,

    # ppinfix(<lst>, <sep>), similar to infix, but parenthesizes inside the expression
    ppinfix := meth(self, lst, sep)
        Print("(");
        self.finfix(lst, c -> Print("(", self(c, 0, 4), ")"), sep);
        Print(")");
    end,

    # condinfix(<lst>, <sep>), for conditions like leq.
    # condinfix([a,b,c], " sep1 ", " sep2 ") unparses to
    # (a sep1 b) sep2 (b sep1 c)
    #
    condinfix := meth(self, lst, sep1, sep2)
        local first, c, i;
        Print("(");
        for i in [1..Length(lst)-1] do
            if i<>1 then Print(sep2); fi;
            self.pinfix([lst[i], lst[i+1]], sep1);
        od;
        Print(")");
    end,
));

#F LayeredUnparser(<unparser-class1>, <unparser-class2>, ...)
#F
#F This creates an unnamed class with superclasses listed in arguments.
#F unparser-class1 takes priority over 2, etc.
#F
Class(LayeredUnparser, rec(
    __call__ := arg >> Checked(Length(arg)>1, ForAll(Drop(arg, 1), IsClass),
        WithBases(arg, rec(operations := PrintOps, __call__ := Unparser.__call__))),

    print := self >> Print("LayeredUnparser(", PrintCS(Drop(self.__bases__,1)), ")")
));


Loc.unparse     := "Loc";
Exp.unparse     := "Exp";
Command.unparse := "Command";

Class(CUnparserBase, Unparser, rec(
    preprocess := (self, o) >> o,
    preprocess_init := (self, o) >> o,

    # example: includes := [ "<math.h>" ]
    includes := [],

    generated_by := Concat("\
/*\
 * This code was generated by Spiral ", SpiralVersion, ", www.spiral.net\
 */\
\n"),

    extraHeader := "",

    header_top := meth(self, subname, o)
        local precomputed_data;
        Print(self.generated_by);

        self.fileinfo(self.opts);

        DoForAll(Concatenation(self.includes, self.opts.includes),
             inc -> Print("#include ", inc, "\n"));

        if IsBound(o.dimensions) then Print("/* ", o.dimensions, " */\n"); fi;

        if IsBound(o.runtime_data) then
            DoForAll(o.runtime_data, x->Print(self.opts.arrayDataModifier, " ",
                                              self.declare(x.t, x, 0, 4), ";\n"));
            Print("\n");
        fi;

        precomputed_data := List(Collect(o, data), x->[x.var, x.value]);
    DoForAll(precomputed_data, d -> When(IsArrayT(d[1].t), self.genData(d[1], d[2])));
    end,

    header_func := meth(self, subname, o)
        local loopvars;
        Print("void ", self.opts.funcModifier, " ", subname, "(",
          self.declare(Y.t, Y, 0, 0), ", ", self.declare(X.t, X, 0, 0));

        if IsBound(self.opts.sig) then
            DoForAll(self.opts.sig, p -> Print(", ", self.declare(p.t, p, 0, 0)));
        fi;
        Print(") {\n");

        loopvars := Set(List(Collect(o, @(1).cond(IsLoop)), x->x.var));
        if loopvars <> [] then Print(Blanks(4), "int ", PrintCS(loopvars), ";\n"); fi;
    end,

    header := meth(self, subname, o)
        self.header_top(subname, o);
        self.header_func(subname, o);
    end,

    footer := meth(self, subname, o)
        local init, loopvars;
        Print("}\n");
        Print("void init_", subname, "() {\n");
        if IsBound(o.runtime_init) then # unparse initialization code

            loopvars := Union(List(o.runtime_init, cc -> List(Collect(cc, @(1).cond(IsLoop)), x->x.var)));
            if loopvars <> [  ]  then
                Print(Blanks(4), "int ", PrintCS(loopvars), ";\n"); fi;
            for init in o.runtime_init do
                init := self.preprocess_init(init);
                self(SReduce(init, o), 4, 4);
            od;
        fi;
        Print(" }\n");
    end,

    genData := (self, v, val) >> Print(
        When(IsArrayT(val.t), self.opts.arrayDataModifier, self.opts.scalarDataModifier), " ",
        self.declare(val.t, v, 0, 4), " = ", self(val,2,2), ";\n",
        When(IsArrayT(val.t), "\n", "")),

    ####################
    ## General
    ####################
    atomic  := (self,o,i,is) >> Print(o),
    param   := (self,o,i,is) >> Print(o.id),
    var     := (self,o,i,is) >> Print(o.id),
    Loc     := (self,o,i,is) >> o.cprint(),

    ####################
    ## Commands
    ####################

    asmvolatile := (self,o,i,is) >> AsmX86Unparser.asmvolatile(o.asm),

    skip := (self,o,i,is) >> Print(Blanks(i), "/* skip */\n"),

    noUnparse :=(self,o,i,is) >> Print(o.str),

    assign := (self,o,i,is) >> Print(Blanks(i), self(o.loc,i,is), " = ", self(o.exp,i,is), ";\n"),

    assign_acc := (self,o,i,is) >> Print(Blanks(i), self(o.loc,i,is), " += ", self(o.exp,i,is), ";\n"),

    chain    := (self,o,i,is) >> DoForAll(o.cmds, c -> self(c, i, is)),
    brackets := (self,o,i,is) >> self.printf("($1)", [o.args[1]]),

    unparseChain := (self,o,i,is) >> DoForAll(o.cmds, c -> self(c, i, is)),

    kern := (self, o, i, is) >> When( IsBound(self.opts.SimFlexKernelFlag) and IsBound(o.bbnum),
        Print(
            "#if !defined(KERN) || defined(KERN", String(o.bbnum), ")\n",
            self(o.cmd, i, is),
            "#endif\n"
        ),
        self(o.cmd, i, is)
    ),
    unroll_cmd := ~.chain,

    # all datas are handled in the header, just proceed to children
    data := (self,o,i,is) >> Print(
        When(not IsArrayT(o.var.t), Print(Blanks(i), self.genData(o.var, o.value))),
        self(o.cmd, i, is)
    ),

    _lt := " <= ",
    loop := (self,o,i,is) >> Checked(IsRange(o.range),
        let(v := o.var, lo := o.range[1], hi := Last(o.range),
            Print(Blanks(i), "for(", v, " = ", lo, "; ", v, self._lt, hi, "; ", v, "++) {\n",
                self(o.cmd,i+is,is),
                Blanks(i), "}\n"))),

    loopn := (self,o,i,is) >>
        let(v := o.var, n := o.range,
            Print(Blanks(i), "for(", v, " = ", 0, "; ", v, self._lt, self(n-1,i,is), "; ", v, "++) {\n",
                self(o.cmd,i+is,is),
                Blanks(i), "}\n")),

    doloop := (self,o,i,is) >>
        let(v := o.var, n := o.range,
            Print(Blanks(i), "do {\n",
                self(o.cmd,i+is,is),
                Blanks(i), "} while( ", self(v,i,is), "<", self(n,i,is), " );\n")),

    loopn := (self, o, i, is) >> self.loop(o, i, is),

    IF := (self,o,i,is) >> Print(Blanks(i),
        "if (", self(o.cond,i,is), ") {\n", self(o.then_cmd,i+is,is), Blanks(i), "}",
        When(o.else_cmd = skip(),
             "\n",
             Print(" else {\n", self(o.else_cmd,i+is,is), Blanks(i), "}\n"))),

    DOWHILE := (self,o,i,is) >> Print(Blanks(i),
    "do \n",  Blanks(i), "{\n", self(o.then_cmd,i+is,is), Blanks(i), "}",
        "while (", self(o.cond,i,is), ");\n" ),

    WHILE := (self,o,i,is) >> Print(Blanks(i),
    "while (", self(o.cond,i,is), ")\n" ,
    Blanks(i), "{\n", self(o.then_cmd,i+is,is), Blanks(i), "}\n"),

    PRINT := (self,o, i, is) >> Print(Blanks(i),
        "printf(\"", o.fmt, "\"",
            When(o.vars <> [],
                Print(", ",
                    DoForAll( DropLast(o.vars,1),
                        e -> Print(self(e,i,is), ", ")
                    ),
                    self(Last(o.vars),i,is)
                ),
                Print("")
            ),
            ");\n"
    ),

    multi_if := meth(self,o,i,is)
        local j, conds;
    conds := o.args { [1..Int(Length(o.args)/2)]*2 - 1 };

        # degenerate case, no conditions, else branch only
        if Length(o.args)=1 then 
        self(o.args[1], i, is);

    # generate switch stmt
    elif ForAll(conds, c -> ObjId(c)=eq and ObjId(c.args[1]) in [var,param] and c.args[1]=conds[1].args[1] and IsValue(c.args[2])) then
        Print(Blanks(i), "switch(", self(conds[1].args[1], i, is), ") { \n");
            j := 1;
            while j < Length(o.args) do
                Print(Blanks(i+Int(is/2)), "case ", self(o.args[j].args[2], i, is), ": ");
        Cond(ObjId(o.args[j+1])=ret, 
             Print(self(o.args[j+1],0,is), Blanks(i+is), "break;\n "),
             Print("{\n", self(o.args[j+1],i+is,is), Blanks(i+is), "break; }\n"));
        j := j+2;
            od;
            # Print out the else branch if it exists (j=Length)
            if j = Length(o.args) then 
        Print(Blanks(i+Int(is/2)), "default: ");
        Cond(ObjId(o.args[j])=ret, 
             Print(self(o.args[j], 0, is)),
             Print("{\n", self(o.args[j], i+is, is), Blanks(i+Int(is/2)), "}\n"));
            fi;
        Print(Blanks(i), "}\n");

    # general IF cascade
    else

            j := 1;
            while j < Length(o.args) do
                Print(Blanks(i), When(j<>1, "else "), "if (",
                      self(o.args[j],  i+is,is), ") {\n",
                      self(o.args[j+1],i+is,is), Blanks(i), "}\n");
        j := j+2;
            od;
            # Print out the else branch if it exists (j=Length)
            if j = Length(o.args) then 
        Print(Blanks(i), "else {\n",
                      self(o.args[j],  i+is, is), Blanks(i), "}\n");
            fi;
    fi;
    end,

    zallocate := (self, o, i, is) >> Print(
        self(allocate(o.loc,o.exp),i,is),
        Blanks(i),"memset(",self(o.loc, i, is)," ,'\\0', sizeof(",
        self.declare(o.exp.t,[],0,0), ") * ", self(o.exp.size,i,is), ");\n"),
#        Blanks(i),"for(int iinit = 0; iinit <  ",self(o.exp.size,i,is),"; iinit++)\n",
#        Blanks(i),"    ", self(o.loc, i, is),"[iinit]=0;\n"),

    ####################
    ## Expressions
    ####################

    RewritableObjectExp := (self,o,i,is) >> Print(o.name, self.pinfix(o.rChildren(), ", ")),
    Exp := (self,o,i,is) >> Print(o.name, self.pinfix(o.args, ", ")),
    ExpCommand := (self,o,i,is) >> Print(Blanks(i), o.name, self.pinfix(o.args, ", "), ";\n"),
    Command := (self,o,i,is) >> Print(Blanks(i), o.name, self.pinfix(o.rChildren(), ", "), ";\n"),
    call := (self,o,i,is) >> Print(Blanks(i), self(o.args[1],i,is), self.pinfix(Drop(o.args,1), ", "), ";\n"),
    errExp := (self, o, i, is) >> self(o.t.zero(), i, is),

    eq  := (self, o, i, is) >> self.condinfix(o.args, " == ", " && "),
    neq := (self, o, i, is) >> self.condinfix(o.args, " != ", " && "),
    geq := (self, o, i, is) >> self.condinfix(o.args, " >= ", " && "),
    leq := (self, o, i, is) >> self.condinfix(o.args, " <= ", " && "),
    gt  := (self, o, i, is) >> self.condinfix(o.args, " > ",  " && "),
    lt  := (self, o, i, is) >> self.condinfix(o.args, " < ",  " && "),

    nth := (self,o,i,is) >> Print(self(o.loc,i,is), "[", self(o.idx,i,is), "]"),
    deref := (self,o,i,is) >> Print("*(", self(o.loc,i,is), ")"),
    addrof := (self,o,i,is) >> Print("&(", self(o.loc,i,is), ")"),
    fdiv := (self,o,i,is) >> Print("(((", self.declare(TReal, [],i,is), ")",
    self(o.args[1],i,is), ") / ", self(o.args[2],i,is), ")"),
    add := (self,o,i,is) >> self.pinfix(o.args, " + "),
    logic_and := (self,o,i,is) >> self.ppinfix(o.args, " && "),
    logic_or := (self,o,i,is) >> self.ppinfix(o.args, " || "),
    logic_neg := (self,o,i,is) >> Print("( !(",self(o.args[1],i,is), ") )"),
    sub := (self,o,i,is) >> self.pinfix(o.args, " - "),
    neg := (self,o,i,is) >> Print("-(", self(o.args[1],i,is), ")"),

    mul := (self,o,i,is) >> self.pinfix(o.args, "*"),
    div := (self,o,i,is) >> self.pinfix(o.args, " / "),
    idiv := (self,o,i,is) >> self.pinfix(o.args, " / "),
    imod := (self,o,i,is) >> self.pinfix(List(o.args, x -> When(IsPtrT(x.t), tcast(TSym("size_t"),x) ,x)), " % "),
    no_mod := (self,o,i,is) >> self(o.args[1],i,is),
    re     := (self,o,i,is) >> self.printf("creal($1)", [o.args[1]]),
    im     := (self,o,i,is) >> self.printf("cimag($1)", [o.args[1]]),
    cxpack := (self,o,i,is) >> self.printf("($1) + _Complex_I*($2)", [o.args[1], o.args[2]]),

    bin_and := (self,o,i,is) >> Print("((", self.pinfix(List(o.args, x -> When(IsPtrT(x.t), tcast(TSym("size_t"),x) ,x)), ")&("), "))"),
    bin_or := (self,o,i,is) >> Print("((", self.pinfix(List(o.args, x -> When(IsPtrT(x.t), tcast(TSym("size_t"),x) ,x)), ")|("), "))"),
    bin_xor := (self,o,i,is) >> Print("((", self(o.args[1],i,is), ")^(", self(o.args[2],i,is),"))"),

    abs := (self,o,i,is) >> Print("abs(", self(o.args[1],i,is), ")"),

    floor := (self,o,i,is) >> Print("((int)(", self(o.args[1],i,is), "))"),

    lShift := (self,o,i,is) >> Cond(IsBound(o.args[3]),
            Error("non implemented"),
            Print("((",self(o.args[1],i,is),") << (",self(o.args[2],i,is), "))")),

    rShift := (self,o,i,is) >> Cond(IsBound(o.args[3]),
            Error("non implemented"),
            Print("((",self(o.args[1],i,is),") >> (",self(o.args[2],i,is), "))")),

    arith_shr := (self, o, i, is) >> Cond( o.t.isSigned(), self.printf("(($1) >> ($2))", [o.args[1], o.args[2]]),
                                           Error("implement arith_shr for unsigned data type")),
    arith_shl := (self, o, i, is) >> Cond( o.t.isSigned(), self.printf("(($1) \<\< ($2))", [o.args[1], o.args[2]]),
                                           Error("implement arith_shl for unsigned data type")),
    
    xor := meth(self,o,i,is)
        Print("((", self(o.args[1],i,is));
        DoForAll(o.args{[2..Length(o.args)]}, e -> Print(")^(", self(e,i,is)));
        Print("))");
    end,

    max := (self,o,i,is) >> self(cond(geq(o.args[1],o.args[2]),o.args[1],o.args[2]),i,is),

    min := (self,o,i,is) >> self(cond(leq(o.args[1],o.args[2]),o.args[1],o.args[2]),i,is),

    log := (self,o,i,is) >> Cond( Length(o.args)=1,
        Cond( o.t = T_Real(32) or (IsBound(self.opts.TRealCtype) and self.opts.TRealCtype = "float"),
                self.printf("logf($1)", [o.args[1]]),
              # else
              self.printf("log((double)($1))", [o.args[1]])),
        Cond( o.t = T_Real(32) or (IsBound(self.opts.TRealCtype) and self.opts.TRealCtype = "float"),
                self.printf("logf($1)/logf($2)", [o.args[1], o.args[2]]),
              # else
              self.printf("log((double)($1))/log((double)($2))", [o.args[1], o.args[2]]))),
    
    pow := (self,o,i,is) >> Cond( 
        o.args[2]=2, 
            self(mul(o.args[1], o.args[1]), i, is),
        # else
            Error("Implement pow()")),

    sqrt  := (self,o,i,is) >> self.printf("sqrt($1)",    [o.args[1]]),
    rsqrt := (self,o,i,is) >> self.printf("$1/sqrt($2)", [o.t.one(), o.args[1]]),
    
    cond := (self,o,i,is) >> Cond(
      Length(o.args)=3,
          Cond(ObjId(o.args[3]) = errExp, 
           self(o.args[2], i, is), 
               Print("((",self(o.args[1],i,is),") ? (",self(o.args[2],i,is),") : (",self(o.args[3],i,is),"))")),

      # NOTE: no else case here, maybe just leave it like this?
      Length(o.args)=2, 
          self(o.args[2],i,is),

      # more than 3 args: do a binsplit trick
      Print("((",self(o.args[1],i,is),") ? (",self(o.args[2],i,is),") : ",
      self(ApplyFunc(cond, Drop(o.args, 2)), i,is),")")),

    _decval :=(self, v) >> let(pf := When(IsBound(self.opts.valuePostfix), self.opts.valuePostfix, ""),
            Print(v, pf)),

    Value := (self,o,i,is) >>
        Cond(
        o.t = TComplex, let(c:=Complex(o.v), re:=ReComplex(c), im:=ImComplex(c),
            Cond(re=0 and im=1,  Print(self.opts.c99.I),
                 re=0 and im=-1, Print("(- ", self.opts.c99.I, ")"),
                 im=0,           Print(self._decval(re)),
                 re=0,           Print("(", self._decval(im), " * ", self.opts.c99.I, ")"),
                 im < 0,         Print("(", self._decval(re), " - ", self.opts.c99.I, " * ", self._decval(-im), ")"),
                 Print("(", self._decval(re), " + ", self.opts.c99.I, " * ", self._decval(im), ")"))),

        o.t = TReal, let(
        v := Cond(IsCyc(o.v), ReComplex(Complex(o.v)), o.v),
        Cond(v < 0, 
         Print("(", self._decval(v), ")"),
         Print(self._decval(v)))),

        IsArray(o.t),                      Print("{", WithBases(self, rec(infixbreak:=4)).infix(o.v, ", ", i), "}"),
        o.v < 0,                           Print("(", o.v, ")"),
        o.t = TBool,                       When(o.v in [true, 1], Print("1"), Print("0")),

        o.t = TUInt,                       Print(o.v, "u"), 
                                           Print(o.v)
    ),

    ##########################
    ## Types and declarations
    ##########################

    # This function unparses the standard decl(var, value, code) Command.
    # First, we unparse array variables then group variables by their type,
    # and then call CUnparserBase.declare on each group

    decl := meth(self,o,i,is)
        local arrays, other, l, arri, myMem;
        [arrays, other] := SplitBy(o.vars, x->IsArray(x.t));
        DoForAll(arrays, v -> Print(Blanks(i), 
                                    When(self.opts.arrayBufModifier <> "", self.opts.arrayBufModifier::" ", ""), 
                                    self.declare(v.t, v, i, is), ";\n"));

        if (Length(other)>0) then
            other:=SortRecordList(other,x->x.t);
            for l in other do
               Sort(l, (a,b)->a.id < b.id);
               Print(Blanks(i), self.declare(l[1].t, l, i, is), ";\n");
            od;
        fi;

        self(o.cmd, i, is);

        #Pop arena for this decl
        if IsBound(self.opts.useMemoryArena) and self.opts.useMemoryArena and Length(arrays) > 0 and arrays[1].id[1] <> 'D' then
          myMem := 0;
          for arri in arrays do 
             # Account for vector allocations in memory arena (which is scalar)
             myMem := myMem + (arri.t.size * When(IsBound(arri.t.t) and ObjId(arri.t.t)=TVect, arri.t.t.size, 1));
          od;
          if ObjId(myMem) = Value then myMem := myMem.v; fi;
          Print(Blanks(i));
          Print("arenalevel += ", myMem, ";\n" );
        fi;
    end,

    # defines a struct
    define := meth(self, o, i, is)
        local e, ee;
        for e in o.types do
            Print(Blanks(i), "typedef struct {\n");
            for ee in e.getVars() do
                Print(Blanks(i+is));
                Print(self.declare(ee.t, [ee], i, is), ";\n");
            od;

            Print(Blanks(i), "} ", e.getName(), ";\n");
        od;
    end,

    tcast := (self, o, i, is) >> Print("((", self.declare(o.args[1], [], i, is), ") ", self(o.args[2],i,is), ")"),

    sizeof := (self, o, i, is) >> Print("sizeof(", self.declare(o.args[1], [], i, is), ")"),

    declare := (self, t, vars, i, is) >> When(
        IsBound(self.(t.name)),
        self.(t.name)(t, When(IsList(vars), vars, [vars]), i, is),
        Error("Can't declare ", vars, " no method '", t.name, "' in ", self)),

    TVect := (self, t, vars, i, is) >> Print("__m128 ", self.infix(vars, ", ", i+is)),

    TComplex := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TComplexCtype), self.opts.TComplexCtype, "complex_t "), " ",
        self.infix(vars, ", ",i+is)),

    T_Complex := (self, t, vars, i, is) >> Print("_Complex ", self.declare(t.params[1], vars, i, is)), 

    T_Real  := (self, t, vars, i, is) >> Print(Cond(
            t.params[1] = 128, "long double",
            t.params[1] = 80, "long double",   #x86 specific
            t.params[1] = 64, "double",
            t.params[1] = 32, "float",
            Error("Type is not supported")
        )," ",self.infix(vars, ", ",i+is)),

    T_Int  := (self, t, vars, i, is) >> Print(Cond(
            t.params[1] = 64, "__int64",
            t.params[1] = 32, "__int32",
            t.params[1] = 16, "__int16",
            t.params[1] = 8, "__int8",
            Error("Type is not supported")
        ), " ", self.infix(vars, ", ",i+is)),

    T_UInt  := (self, t, vars, i, is) >> Print("unsigned ", Cond(
            t.params[1] = 64, "__int64",
            t.params[1] = 32, "__int32",
            t.params[1] = 16, "__int16",
            t.params[1] = 8, "__int8",
            t.params[1] = 1, "__bit",
            Error("Type is not supported")
        ), " ", self.infix(vars, ", ",i+is)),

    T_Struct := (self, t, vars, i, is) >> Print(
        t.getName(), " ", self.infix(vars, ", ", i+is)
    ),

    TReal  := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TRealCtype), self.opts.TRealCtype, TReal.ctype), " ",
        self.infix(vars, ", ",i+is)),

    TInt  := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TIntCtype), self.opts.TIntCtype, TInt.ctype), " ",
        self.infix(vars, ", ",i+is)),

    TDummy := ~.TInt,

    TBool  := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TIntCtype), self.opts.TIntCtype, TInt.ctype), " ",
        self.infix(vars, ", ",i+is)),

    TUInt  := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TUIntCtype), self.opts.TUIntCtype, TUInt.ctype), " ",
        self.infix(vars, ", ",i+is)),

    TULongLong  := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TULongLongCtype), self.opts.TULongLongCtype, TULongLong.ctype), " ",
        self.infix(vars, ", ",i+is)),

    TChar  := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TCharCtype), self.opts.TCharCtype, TChar.ctype), " ",
        self.infix(vars, ", ",i+is)),

    TUChar  := (self, t, vars, i, is) >> Print(
        When(IsBound(self.opts.TUCharCtype), self.opts.TUCharCtype, TUChar.ctype), " ",
        self.infix(vars, ", ",i+is)),

    TVoid  := (self, t, vars, i, is) >> Print("void ", self.infix(vars, ", ",i+is)),

    _restrict := (self, t) >> let(opts := self.opts, rst := Concat(When(IsBound(opts.restrict),
                    opts.restrict(), "restrict"), " "), When(t._restrict, rst, "")),

    TPtr  := (self, t, vars, i, is) >>
        Print(Cond(not IsBound(t.qualifiers) or t.qualifiers=[], "", Print(self.infix(t.qualifiers, " "), " ")),
          Cond(vars=[],
          Print(self.declare(t.t, [], i, is), " *", self._restrict(t)),
          Print(self.declare(t.t, [], i, is),
              Print(" *", self._restrict(t) ),
              self.infix(vars, Concatenation(", *", self._restrict(t)), i+is)))),

    TSym := (self, t, vars, i, is) >> Print(t.id, " ", self.infix(vars, ", ")),

    _TFunc_args := (self, t, i, is) >> let(params := DropLast(t.params, 1),
        Print(
            DoForAllButLast(params, p -> Print(self.declare(p, [], i, is), ", ")),
            self.declare(Last(params), [], i, is))),

    TFunc  := (self, t, vars, i, is) >>
        Cond(Length(vars) in [0,1],
               PrintEvalF("$1 (*$2)($3)",
                 () -> self.declare(Last(t.params), [], i, is),
                 When(vars=[], "", vars[1].id),
                 () -> self._TFunc_args(t, i, is)),
               Print(
                   DoForAllButLast(vars, v -> Print(self.declare(t, v, i, is), ", ")),
                   self.declare(t, [Last(vars)], i, is))),

    TArray := meth(self,t,vars,i,is)
        local dims, elt, v, ptype, vsize;
        if Length(vars) > 1 then DoForAll(vars, v->Print(self.TArray(t, [v], i, is), "; "));
        elif Length(vars) = 0 then
            Print(self.declare(t.t, [], i, is), " *");
        else
            # at this point Length(vars)=1
            v := When(IsList(vars), vars[1], vars);
            dims := []; elt := t;
            while IsArray(elt) do Add(dims, elt.size); elt := elt.t; od;

          #NOTE: Ignoring twiddles by looking for "D" in .id
          # Better way: look for func.name="init" in parent/context
          if IsBound(self.opts.useMemoryArena) and self.opts.useMemoryArena and v.id[1] <> 'D' then
            #NOTE: Arena currently doesn't handle multiple dims.
            #NOTE: Slightly ugly hack to get this to be a pointer

            # To handle vectors. Arena is declared only for scalars. For
            # vectors, we must manually scale the allocation by vector length.
            vsize := 1; if ObjId(elt)=TVect then vsize := elt.size; fi;
#            ptype := Concatenation(elt.name, "Pointer");
#            self.(ptype)(elt, [v], i, is);
            self.TPtr(TPtr(elt), [v], i, is);
            Print(" =  &(ARENA[ (arenalevel-=",self(dims[1]*vsize,i,is),") ])");
          else
            self.(elt.name)(elt, [v], i, is);
            DoForAll(dims, d->Print("[",self(d,i,is),"]"));
          fi;

        fi;
    end,

    # comments embedded in code

    comment := (self, o, i, is) >> When(Length(o.exp) = 0,
        PrintLine(),
        PrintLine(Blanks(i), "/* ", o.exp, " */")
    ),

    quote := (self, o, i, is) >> Print("\"", self(o.cmd,i,is), "\"")
));

# Locate instances of assign(loc, Value), where types of Value and loc
# do not match. Fix the value to be of correct type.
# Note: For linear transforms the only possible Value is 0 (otherwise
# transform is not linear)
FixAssign0 := c -> SubstTopDownNR(c, [assign, @(1), @(2,Value,e->e.t <> @(1).val.t)],
    e -> assign(@(1).val,
            When(@(2).val.v = 0, @(1).val.t.zero(),
                                 @(1).val.t.value(@(2).val.v))));

Class(CMacroUnparser, CUnparserBase, rec(
    preprocess := (self, c) >> FixAssign0(PropagateTypes(c)),
    preprocess_init := (self, c) >> FixAssign0(PropagateTypes(c)),

    # Split non-binary "+" into nested binary ops
    add := (self,o,i,is) >> Cond(
        o.t=TInt or IsPtrT(o.t), Inherited(o, i, is), 
        Length(o.args)=2, self.prefixTT("ADD", o.args[1].t, o.args[2].t, o.args),
        let(rem := ApplyFunc(add, Drop(o.args, 1)),
            self.prefixTT("ADD", o.args[1].t, rem.t, [o.args[1], rem]))),

    sub := (self,o,i,is) >> Cond(o.t=TInt or IsPtrT(o.t), Inherited(o, i, is), 
    self.prefixTT("SUB", o.args[1].t, o.args[2].t, o.args)),

    nth := (self,o,i,is) >> Cond(o.t=TInt or IsPtrT(o.t), Inherited(o, i, is), self.prefixT("NTH", o.t, [o.loc, o.idx])),
    neg := (self,o,i,is) >> Cond(o.t=TInt or IsPtrT(o.t), Inherited(o, i, is), self.prefixT("NEG", o.t, o.args)),

    bin_and := (self, o, i, is) >> Cond(o.t=TInt or ObjId(o.t) in [T_Int, T_UInt], Inherited(o, i, is), self.prefixT("AND", o.t, o.args)),
    bin_xor := (self, o, i, is) >> Cond(o.t=TInt or ObjId(o.t) in [T_Int, T_UInt], Inherited(o, i, is), self.prefixT("XOR", o.t, o.args)),
    div  := (self,o,i,is) >> Cond(o.t=TInt, Inherited(o, i, is), self.prefixTT("DIV", o.args[1].t, o.args[2].t, o.args)),
    imod := (self,o,i,is) >> self.prefixT("IMOD", o.t, o.args),
    max  := (self,o,i,is) >> self.prefixTT("MAX", o.args[1].t, o.args[2].t, o.args),
    min  := (self,o,i,is) >> self.prefixTT("MIN", o.args[1].t, o.args[2].t, o.args),
    
    idiv := (self,o,i,is) >> Cond(o.t=TInt and ForAll(o.args,x->x.t=TInt), Inherited(o, i, is), 
        self.prefixT("IDIV", o.t, o.args)),

    fdiv := (self,o,i,is) >> self.prefixTT("FDIV", o.args[1].t, o.args[2].t, o.args),

    re     := (self,o,i,is) >> self.prefixT("RE", o.args[1].t, o.args),
    im     := (self,o,i,is) >> self.prefixT("IM", o.args[1].t, o.args),
    cxpack := (self,o,i,is) >> self.prefixT("C", o.t, o.args),

    no_mod := (self,o,i,is) >> self(o.args[1],i,is),

    Value := (self, o, i, is) >> Cond(
        IsArray(o.t), Print("{", self.infix(o.v, ", "), "}"),
        let(fmt := self._const(o),
        pfx := Cond(self.cx.isInside(data), "CD_", "C_"),
            Cond(fmt[2]=[], Print(pfx, fmt[1]),
                 fmt[1]="INT", Print(fmt[2][1]), 
                 self.prefix(Concat(pfx, fmt[1]), fmt[2])))),

    # mults by different constants are unparsed differently
    # Split non-binary "*" into nested binary ops
    mul := (self,o,i,is) >> Cond(
        o.t=TInt, Inherited(o, i, is), 
        Length(o.args)<>2, self(mul(o.args[1], ApplyFunc(mul, Drop(o.args, 1))), i, is),
        let(
         # check if constant ended up in slot #2
         a := When(IsValue(o.args[2]), o.args[2], o.args[1]),
         b := When(IsValue(o.args[2]), o.args[1], o.args[2]),
         When(not (IsValue(a) or (IsVar(a) and IsBound(a.value))),
            # <a> is not a constant
            self.prefix(Concat("MUL_", self._pfx(a.t), "_", self._pfx(b.t)), [a,b]),
            # <a> is a constant
            let(fmt := self._const( When(IsValue(a), a, a.value) ),
                self.prefix(Concat("MUL_", fmt[1], "_", self._pfx(b.t)),
                     # check if constant is special, and does not go into MUL args
                     # for example fmt[1]="I" denotes sqrt(-1), one such constant
                     When(fmt[2]=[], [b], [a, b])))))),

    prefixT := (self, funcname, t, args) >>
        self.prefix(Concat(funcname, "_", self._pfx(t)), args),

    prefixTT := (self, funcname, t1, t2, args) >>
        self.prefix(Concat(funcname, "_", self._pfx(t1), "_", self._pfx(t2)), args),

    prefixTTT := (self, funcname, t1, t2, t3, args) >> 
        self.prefix(Concat(funcname, "_", self._pfx(t1), "_", self._pfx(t2), "_", self._pfx(t3)), args),
    # this is getting really ugly
    prefixTTTT := (self, funcname, t1, t2, t3, t4, args) >> 
        self.prefix(Concat(funcname, "_", self._pfx(t1), "_", self._pfx(t2), "_", self._pfx(t3), "_", self._pfx(t4)), args),

    prefix_T := (self, funcname, args) >>
        self.prefix(funcname :: ConcatList(args, a -> "_" :: self._pfx(a.t)), args),

    _pfx := (self, t) >> Cond(
        IsComplexT(t),   "CPX",
        IsRealT(t),      "FLT",
        IsOrdT(t),       "INT",
        t = TUnknown,    "UNK",
        ObjId(t) = TSym, "SYM",
        IsPtrT(t),       "P" :: self._pfx(t.t),
        IsArrayT(t),     "A" :: self._pfx(t.t),
        IsVecT(t), Cond(
            IsComplexT(t.t), "FC" :: StringInt(t.size),
            IsRealT(t.t),    "FV" :: StringInt(t.size),
            IsOrdT(t.t),     "IV" :: StringInt(t.size),
            Error("Can't handle type ", t)
        ),
        Error("Can't handle type ", t)),

    # returns a tuple [suffix, args], where suffix is used for MUL_XXX or C_XXX,
    # and args are additional parameters into C_XXX
    _const := (self,o) >> Cond(
        # we assume here that complex constants with 0 imaginary parts have
        # been converted to TReal already
        o.t = TComplex, let(c:=Complex(o.v), re:=ReComplex(c), im:=ImComplex(c),
            Cond(re=0 and im=1,  ["I", []],
                re=0 and im=-1, ["NI", []],
                re=0,           ["IM", [im]],
                im < 0,         ["CPXN", [re, -im]],
                               ["CPX", [re, im]])),
        ObjId(o.t) = T_Complex, let(c:=Complex(o.v), re:=ReComplex(c), im:=ImComplex(c),
            Cond(re=0 and im=1,  ["I", []],
                re=0 and im=-1, ["NI", []],
                re=0,           ["IM", [im]],
                im < 0,         ["CPXN", [re, -im]],
                               ["CPX", [re, im]])),
        o.t = TReal and IsCyc(o.v),        ["FLT", [ReComplex(Complex(o.v))]],
        o.t = TReal,                       ["FLT", [o.v]],
        o.t = TInt,                        ["INT", [o.v]],
        o.t = TUnknown,                    ["INT", [o.v]], # NOTE: there is a bug that creates V(0) with TUnknown
        o.t = TString,                     ["STR", [o.v]],
        o.t = TBool,                       ["INT", [When(o.v in [true, 1], 1, 0)]],
        IsVecT(o.t), Cond(
            o.t.t = TReal, [Concat("FV",StringInt(o.t.size)), List(o.v, x->x.v)],
            o.t.t = TComplex, [Concat("FC",StringInt(o.t.size)), List(o.v, x->x.v)],
            o.t.t = TInt, [Concat("IV",StringInt(o.t.size)), List(o.v, x->x.v)],
            Error("Don't know how to handle constant of type ", o.t)
        ),
        Error("Don't know how to handle constant of type ", o.t)
    ),

    tcast := (self, o, i, is) >> Print("((", self.declare(o.args[1], [], i, is), ") ", self(o.args[2],i,is), ")"),

    TVect    := (self, t, vars, i, is) >> Print(self._pfx(t), " ", self.infix(vars, ", ")),
    TComplex := (self, t, vars, i, is) >> Print(self._pfx(t), " ", self.infix(vars, ", ")),
    TReal    := (self, t, vars, i, is) >> Print(self._pfx(t), " ", self.infix(vars, ", ")),
    TInt     := (self, t, vars, i, is) >> Print("int ", self.infix(vars, ", ")),

    gen := meth(self, subname, o, opts)
        local oo;
        self.opts := CopyFields(opts, rec(subName := subname));
        oo := self.preprocess(o);
        Print(self.header(subname, oo), Unparse(oo, self, 0, 4), self.footer(subname, oo));
    end,

    fld := (self, o, i, is) >> Print(self(o.loc, i, is), When(IsPtrT(o.loc.t), "->", "."), o.id),
    ufld := ~.fld,

    header := (self, subname, o) >> Print(
        self.generated_by,
        self.extraHeader,
        self.fileinfo(self.opts),
        DoForAll(self.includes, inc -> Print("#include ", inc, "\n")),
        DoForAll(self.opts.includes, inc -> Print("#include ", inc, "\n"))
    ),

    footer := Ignore,

    data := (self,o,i,is) >> Print(Blanks(i), self.genData(o.var, o.value), self(o.cmd, i, is)),

    func := (self, o, i, is) >> let(
        parameters:=Flat(o.params),
        id := Cond(o.id="transform" and IsBound(self.opts.subName),
                     self.opts.subName,
                   o.id="init"      and IsBound(self.opts.subName),
                     Concat("init_",self.opts.subName),
                   o.id="destroy"   and IsBound(self.opts.subName),
                     Concat("destroy_",self.opts.subName),
                   o.id),
        Print("\n", Blanks(i),
            When(IsBound(o.inline) and o.inline,"inline ",""),
            self.opts.funcModifier, self.declare(o.ret, var(id, o.ret), i, is), "(",
            DoForAllButLast(parameters, p->Print(self.declare(p.t, p,i,is), ", ")),
            When(Length(parameters)>0, self.declare(Last(parameters).t, Last(parameters),i,is), ""), ") ",
            "{\n",
            self(o.cmd, i+is, is),
            Blanks(i),
            "}\n")),

    # C99 style, loop var declared inside
    # - needed for correct operation of OpenMP
    # - simplifies function declarations (no need to worry about declaring loop vars)
    loop := (self, o, i, is) >> let(v := o.var, lo := o.range[1], hi := Last(o.range),
        Print(Blanks(i), "for(int ", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) {\n",
          self(o.cmd,i+is,is),
          Blanks(i), "}\n")),

    loopn := (self, o, i, is) >> let(v := o.var, lo := 0, hi := o.range, #NOTE: YSV what is the right thing here?
        Print(Blanks(i), "for(int ", self(v, i, is), " = ", self(lo, i, is), "; ", self(v, i, is), " < ", self(hi, i, is), "; ", self(v, i, is), "++) {\n",
          self(o.cmd,i+is,is),
          Blanks(i), "}\n")),

    program := (self,o,i,is) >> DoForAll(o.cmds, c -> self(c,i,is)),

    allocate := (self, o, i, is) >> Print(Blanks(i),
        self(o.loc, i, is), " = (", self.declare(o.exp.t, [],0,0), "*) MALLOC(sizeof(",
        self.declare(o.exp.t,[],0,0), ") * ", self(o.exp.size,i,is), ");\n"),

#        Blanks(i),"for(int iinit = 0; iinit <  ",self(o.exp.size,i,is),"; iinit++)\n",
#        Blanks(i),"    ", self(o.loc, i, is),"[iinit]=0;\n"),

    deallocate := (self, o, i, is) >> Print(
        Blanks(i),
        "FREE(",
        self(o.loc, i, is),
        ");\n"
    ),

    ret := (self, o, i, is) >> Print(Blanks(i), "return ", self(o.args[1], i+is, is), ";\n"),

    tcvt := (self, o, i, is) >> self.printf("(($1)($2))", [ o.args[1], o.args[2] ])
));

# can handle program/func, etc
#
Class(CUnparser, CUnparserBase, rec(
    gen := meth(self, subname, o, opts)
        local oo;
        self.opts := CopyFields(opts, rec(subName := subname));
        oo := self.preprocess(o);
		self.checkPrintRuleTree(o, opts);
        Print(self.header(subname, oo), Unparse(oo, self, 0, 4), self.footer(subname, oo));
    end,

    fld := (self, o, i, is) >> Print(self(o.loc, i, is), When(IsPtrT(o.loc.t), "->", "."), o.id),
    ufld := ~.fld,

    header := (self, subname, o) >> Print(
        self.generated_by,
        self.extraHeader,
        self.fileinfo(self.opts),
        DoForAll(self.includes, inc -> Print("#include ", inc, "\n")),
        DoForAll(self.opts.includes, inc -> Print("#include ", inc, "\n"))
    ),
	
	checkPrintRuleTree := meth(self, o, opts)
		if IsBound(opts.printRuleTree) and opts.printRuleTree and IsBound(o.ruletree) then
			Print("/* RuleTree:\nrt :=\n");
			Print(o.ruletree);
			Print("\n;\n*/\n\n");
		fi;
	end,

    footer := Ignore,

    data := (self,o,i,is) >> Print(Blanks(i), self.genData(o.var, o.value), self(o.cmd, i, is)),

    func := (self, o, i, is) >> let(
        parameters:=Flat(o.params),
        id := Cond(o.id="transform" and IsBound(self.opts.subName),
                     self.opts.subName,
                   o.id="init"      and IsBound(self.opts.subName),
                     Concat("init_",self.opts.subName),
                   o.id="destroy"   and IsBound(self.opts.subName),
                     Concat("destroy_",self.opts.subName),
                   o.id),
        Print("\n", Blanks(i),
            self.opts.funcModifier, self.declare(o.ret, var(id, o.ret), i, is), "(",
            DoForAllButLast(parameters, p->Print(self.declare(p.t, p,i,is), ", ")),
            When(Length(parameters)>0, self.declare(Last(parameters).t, Last(parameters),i,is), ""), ") ",
            "{\n",
            When(IsBound(self.opts.postalign), DoForAll(parameters, p->self.opts.postalign(p,i+is,is))),
            self(o.cmd, i+is, is),
            Blanks(i),
            "}\n")),

    # C99 style, loop var declared inside
    loop := (self, o, i, is) >> let(v := o.var, lo := o.range[1], hi := Last(o.range),
        Print(When(IsBound(self.opts.looppragma), self.opts.looppragma(o,i,is)),
          Blanks(i), "for(int ", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) {\n",
          self(o.cmd,i+is,is),
          Blanks(i), "}\n")),

    loopn := (self, o, i, is) >> let(v := o.var.id, lo := 0, hi := o.range,
        Print(Blanks(i), "for(int ", v, " = ", lo, "; ", v, " < ", self(hi,i,is), "; ", v, "++) {\n",
          self(o.cmd,i+is,is),
          Blanks(i), "}\n")),

    program := (self,o,i,is) >> DoForAll(o.cmds, c -> self(c,i,is)),

    allocate := (self, o, i, is) >> Print(Blanks(i),
        self(o.loc, i, is), " = (", self.declare(o.exp.t, [],0,0),
        "*) calloc(", self(o.exp.size,i,is), ", sizeof(", self.declare(o.exp.t,[],0,0), "));\n"),

    deallocate := (self, o, i, is) >> Print(
        Blanks(i),
        "free(",
        self(o.loc, i, is),
        ");\n"
    ),

    ret := (self, o, i, is) >> Print(Blanks(i), "return ", self(o.args[1], i+is, is), ";\n"),

    call := (self, o, i, is) >> Print(Blanks(i), o.args[1].id, self.pinfix(Drop(o.args, 1), ", "), ";\n"),

    fcall := (self, o, i, is) >> Print(self(o.args[1],0,0), "(", self.infix(Drop(o.args, 1), ", "), ")"),

    # structure definition
    struct := (self, o, i, is) >> Print(
        Blanks(i), "typedef struct {\n",
        DoForAll(o.fields, f ->
            Print(Blanks(i+is), self.declare(f.t, f, i+is, is), ";\n")
        ),
        Blanks(i), "} ", o.id, ";\n\n"
    )
));

# old style variable declarations. this is for older/stricter compilers (like gcc-2.5.2 used by simplescalar)
# {int v; for(v=0; ....  rather than for(int v=0; ...
# and
# { double x; .... rather than double x
Class(C89Unparser, CUnparser, rec(

    loop := (self, o, i, is) >> let(v := o.var, lo := o.range[1], hi := Last(o.range),
        Print(Blanks(i), "{int ", v, "; for(", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) {\n",
          self(o.cmd,i+is,is),
          Blanks(i), "}}\n")),

    loopn := (self, o, i, is) >> let(v := o.var.id, lo := 0, hi := o.range,
        Print(Blanks(i), "{int ", v, "; for(", v, " = ", lo, "; ", v, " < ", self(hi,i,is), "; ", v, "++) {\n",
          self(o.cmd,i+is,is),
          Blanks(i), "}}\n")),

    # exactly like the normal decl except wrapped in {}
#   decl := meth(self,o,i,is)
#       local arrays, other, l;
#       Print("{");
#       [arrays, other] := SplitBy(o.vars, x->IsArray(x.t));
#       DoForAll(arrays, v -> Print(Blanks(i), self.opts.arrayBufModifier, " ", self.declare(v.t, v, i, is), ";\n"));

#       if (Length(other)>0) then
#           other:=SortRecordList(other,x->x.t);
#           for l in other do
#              Sort(l, (a,b)->a.id < b.id);
#              Print(Blanks(i), self.declare(l[1].t, l, i, is), ";\n");
#           od;
#       fi;

#       self(o.cmd, i, is);
#       Print("}");
#   end,
));

# FFTX: Temporary to handle idiv(imod()).
CUnparser.("idivmod") := (self,o,i,is) >> self(imod(idiv(o.args[1], o.args[3]), o.args[2]), i, is);

#
# NOTE: These are obsolete names
#
CUnparserProg := CUnparser;
CMacroUnparserProg := CMacroUnparser;
C89UnparserProg := C89Unparser;
