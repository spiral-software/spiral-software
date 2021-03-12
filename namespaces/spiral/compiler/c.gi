
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


EPS:=1E-14;
RoundEPS:=(v)->Cond((v>-EPS) and (v<EPS),0,v);
IsVal:=(x,v)->((x>(v-EPS)) and (x<(v+EPS)));
IsZero:=(x)->IsVal(x,0);
IsOne:=(x)->IsVal(x,1);
IsMinusOne:=(x)->IsVal(x,-1);

# Printing to C

Declare(CGen);

ISum.cprint := meth(self,i,is)
    Print("for( ", self.var.cprint(), " = ", self.domain, ") { \n");
    self.expr.cprint(i+is,i);
    Print(Blanks(i), "}\n");
end;

Compose.cprint := meth(self,i,is)
    local c;
    for c in self.children() do
	    Print(Blanks(i+is));
        c.cprint(i+is, is);
        Print(";\n");
    od;
    Print("\n");
end;

# -------------------
# TYPES
# -------------------

re.cprint := self >> Print(self.args[1].cprint(), ".r");
im.cprint := self >> Print(self.args[1].cprint(), ".i");

TComplex.ctype := "struct{double r,i;}";
TDouble.ctype  := "double";
TInt.ctype     := "int";

Typ.cdecl := (self, var) >> When(IsList(var),
    Print(self.ctype, " ", PrintCS(List(var, v->v.id))),
    Print(self.ctype, " ", var.id));

TArray.cdecl  := meth(self,var)
    local dims, elemtype;
    if IsList(var) then
	    DoForAll(var, v->Print(v.t.cdecl(v), "; "));
    elif not IsArray(self.t) then
        Print(self.t.ctype, Print(" ", var.id, "[", self.size, "]"));
    else
        dims := [];
        elemtype := self;
        while IsArray(elemtype) do
            Add(dims, elemtype.size);
            elemtype := elemtype.t;
        od;
        Print(elemtype.ctype, " ", var.id);
        DoForAll(dims, d->Print("[",d,"]"));
    fi;
end;


# -------------------
# EXPRESSIONS
# -------------------

_cprintcs := function ( lst )
    local  i;
    if Length(lst) = 0 then
	    return; 
	fi;
    lst[1].cprint();
    for i  in [ 2 .. Length(lst) ]  do
        Print(", ", lst[i].cprint());
    od;
end;

ccprint := (v,t) -> v;

Command.cprint := (self,i,is) >> self.print(i,is);
Exp.cprint := self >> self.print();
Loc.cprint := self >> self.print();

Value.idx_cprint := self >> Print(self.v);

Value.cprint := self >> Cond(
    self.t = TComplex, let(c:=Complex(self.v),
        Print("{", ReComplex(c), ",", ImComplex(c), "}")),
    self.t = TDouble and IsCyc(self.v),
        Print("(", ccprint(ReComplex(Complex(self.v)), self.t), ")"),
    IsArray(self.t),
        Print("{", 
		(self.v), "}"),
    self.v < 0,
        Print("(", ccprint(self.v, self.t), ")"),
    Print(ccprint(self.v, self.t)));

infix_print := function(lst, sep)
    local first, c;
    first := true;
    for c in lst do
        if first then
		    first := false;
        else
		    Print(sep);
        fi;
        c.cprint();
    od;
    return "";
end;

pinfix_print := function(lst, sep)
    local first, c;
    first := true;
    for c in lst do
        if first then
		    first := false;
        else
		    Print(sep);
        fi;
        if not IsBound(c.rChildren) or c.rChildren()=[] then
		    c.cprint();
        else
		    Print("(",c.cprint(), ")");
		fi;
    od;
    return "";
end;


add.idx_cprint := self >> Print( "(", infix_print(self.args, " + "), ")");
sub.idx_cprint := self >> Print( "(", infix_print(self.args, " - "), ")");
mul.idx_cprint := self >> Print( "", infix_print(self.args, "*"), "");
neg.idx_cprint := self >> Print("-(", self.args[1].cprint(), ")");

ExpCommand.cprint := (self,i,is) >>
    Print(Blanks(i), self.name, "(", infix_print(self.args, ", "), ");\n");

nth.cprint := self >> Print(self.loc.cprint(), "[", self.idx.cprint(), "]");
Exp.cprint := self >> Print(self.name, "(", infix_print(self.args, ", "), ")");
add.cprint := self >> Print( "(", infix_print(self.args, " + "), ")");
sub.cprint := self >> Print( "(", infix_print(self.args, " - "), ")");
mul.cprint := self >> Print( "", infix_print(self.args, "*"), "");
neg.cprint := self >> Print("-(", self.args[1].cprint(), ")");

div.cprint := self >> Print( "", pinfix_print(self.args, "/"), "");
idiv.cprint := self >> Print( "(", pinfix_print(self.args, "/"), ")");
imod.cprint := self >> Print( "(", pinfix_print(self.args, "%"), ")");
no_mod.cprint := self >> self.args[1].cprint();

cond.cprint := self >> Checked(Length(self.args)=3,
    Print("((", self.args[1].cprint(), ") ? (",
            self.args[2].cprint(), ") : (",
        self.args[3].cprint(), "))"));

leq.cprint := self >> Chain(
        Print("(1"),
        FoldL1(self.args, (a,b) ->
        Chain(Print(" && ", "(", a.cprint(), " <= ", b.cprint(), ")"), b)),
        Print(")"));

# -------------------
# COMMANDS
# -------------------

assign.cprint := meth(self, i, is)
    Print(Blanks(i), self.loc.cprint(), " = ", self.exp.cprint(), ";\n");
end;

chain.cprint := meth(self,i,is)
    local c, vars;
    Print(Blanks(i), "{", Blanks(is-1));
    vars := Set(
    List(Collect(self, [assign, @.target(var), @(1)]),
		e->e.loc));
    if false and Length(vars) > 0 then
	    Print("double ");
	    PrintCS(vars);
	    Print(";\n");
    else
	    Print("\n");
    fi;
    for c in self.cmds do
	    c.cprint(i+is, is);
    od;
    Print(Blanks(i), "}\n");
end;

data.cprint := meth(self,i,is)
    Print(Blanks(i), "{", Blanks(is-1), When(IsArray(self.value.t), "static ", ""),
        self.value.t.cdecl(self.var), " = ", self.value.cprint(), ";");
    Print("\n");
    self.cmd.cprint(i+is, is);
    Print(Blanks(i), "}\n");
end;

decl.cprint := meth(self,i,is)
    local v, first, types;
    Print(Blanks(i), "{", Blanks(is-1));
    if Length(self.vars) > 1 and Length(Set(List(self.vars, v->v.t)))=1 then
        # all types are the same
        self.vars[1].t.cdecl(self.vars);
        Print(";\n");
    else
        first := self.vars[1];
        if IsArray(first.t) then
		    Print("static ");
		fi;
        first.t.cdecl(first);
		Print(";\n");
        for v in Drop(self.vars, 1) do
            Print(Blanks(i+is));
            if IsArray(v.t) then
			    Print("static ");
			fi;
            v.t.cdecl(v);
            Print(";\n");
        od;
    fi;
    self.cmd.cprint(i+is, is);
    Print(Blanks(i), "}\n");
end;

loop.cprint_c99 := meth(self,i,is)
       local v, lo, hi;
       Constraint(IsRange(self.range));
       v := self.var; 
	   lo := self.range[1]; 
	   hi := Last(self.range);
       Print(Blanks(i));
       Print("for(int ", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) \n");
       self.cmd.cprint(i+is, is);
end;

loop.cprint := meth(self,i,is)
       local v, lo, hi;
       Constraint(IsRange(self.range));
       v := self.var; 
	   lo := self.range[1]; 
	   hi := Last(self.range);
       Print(Blanks(i));
       Print("{int ",v,"; for(", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) \n");
       self.cmd.cprint(i+is, is);
       Print("}");
end;

IF.cprint := meth(self,i,is)
       local v, lo, hi;
       Print(Blanks(i));
       Print("if(", self.cond.cprint(), ") {\n");
       self.then_cmd.cprint(i+is, is);
       Print(Blanks(i), "} else {\n");
       self.else_cmd.cprint(i+is, is);
       Print(Blanks(i), "}\n");
end;


CGenScalarANSI_C := function(subname, code)
   Constraint(IsCommand(code));
   Print(
    "extern void init_",subname," ();\n",
    "extern void ",subname," (",TDouble.ctype , " *, ",TDouble.ctype , " *);\n"
   );
   Print("void ", subname, "(", TDouble.ctype, " *Y, ", TDouble.ctype, " *X) {\n");
   code.cprint(4,2);
   Print("}\n");

   Print("void init_", subname, "() { }\n");
end;

CGenScalarIntel_C := function(subname, code)
   Constraint(IsCommand(code));
   Print(
    "extern void init_",subname," ();\n",
    "extern void ",subname," (",TDouble.ctype , " *, ",TDouble.ctype , " *);\n"
   );
   Print("void ", subname, "(", TDouble.ctype, " *restrict Y, ", TDouble.ctype, " *restrict X) {\n",
   "    __assume_aligned(X, 16)\n",
   "    __assume_aligned(Y, 16)\n");
   code.cprint(4,4);
   Print("}\n");

   Print("void init_", subname, "() { }\n");
end;


CGenScalarVectorC := function(subname, code)
   Constraint(IsCommand(code));
   Print(
    "extern void init_",subname," ();\n",
    "extern void ",subname," (",TDouble.ctype , " *, ",TDouble.ctype , " *);\n"
   );
   Print("void ", subname, "(__declspec(alignedvalue(16)) ", TDouble.ctype, " *restrict Y, __declspec(alignedvalue(16)) ", TDouble.ctype, " *restrict X) {\n");
   code.cprint(4,4);
   Print("}\n");

   Print("void init_", subname, "() { }\n");
end;



CGenScalarCPP := function(subname, code)
   Constraint(IsCommand(code));
   Print(
    "#ifdef __cplusplus\n",
    "extern \"C\" {\n",
    "#endif\n",
    "extern void init_",subname," ();\n",
    "extern void ",subname," (",TDouble.ctype , " *, ",TDouble.ctype , " *);\n",
    "#ifdef __cplusplus\n",
    "}\n",
    "#endif\n"
   );
   Print("void ", subname, "(", TDouble.ctype, " *Y, ", TDouble.ctype, " *X) {\n");
   code.cprint(4,4);
   Print("}\n");

   Print("void init_", subname, "() { }\n");
end;

CGen:=CGenScalarANSI_C;

# CGenDMP is set to a dummy-value here.
# Will be set to DMPUnparser.gen as soon as this is defined
# CGenDMP:=CGenScalarANSI_C;

CRuntimeTest := function ( testFunc, cmd, opts )
    local prog, result, cgenFunc;
    opts := Copy(opts);
    Constraint(IsCommand(cmd));
    Constraint(IsRec(opts));
    if not IsBound(opts.dataType) then 
	    opts.dataType := "real"; 
	fi;
    opts := DeriveSPLOptions(cmd, opts);
    prog := ProgSPL(cmd, opts);
    prog.type := ProgInputType.TargetSource;

    cgenFunc := When(IsBound(opts.unparser), (f, cmd) -> opts.unparser.gen(f, cmd, opts), CGen);
    PrintTo(prog.file, cgenFunc(prog.sub_name, cmd));
    result := testFunc(cmd, opts, prog);

    if IsBool(result) and result=false then
        Error("Compiled code execution failed, offending program kept in '", prog.file, "'");
    else
        return result;
    fi;
end;


Declare(CodeRuleTreeOpts);
CVerifyRT := function(rt, opts)
    local c, m1,m2,nt;
    nt:=rt.node;
    c := CodeRuleTreeOpts(rt, opts);
    m1 := MatSPL(nt);
    m2 := CMatrix(c, opts);
    return Maximum(Flat(MapMat(m1-m2, i->AbsComplex(ComplexAny(i)))));
end;

CVerifyMMM:=(c,opts,t)->opts.profile.verify(c,opts,t);
