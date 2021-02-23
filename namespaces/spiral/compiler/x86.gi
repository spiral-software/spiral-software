
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(AsmX86Unparser, Unparser, rec(


    asmvolatile := meth(self, o)
        local oo,a,x;

	a := [];
        oo := self.preprocess(o);
	while(ObjId(oo)=data) do
	  oo.var.asmpseudoregister:=Length(a);
	  Append(a,[[oo.var,oo.value]]);
	  Print("static const double ",oo.var.id,"=",oo.value.v,";\n");
	  oo:=oo.cmd;
	 od;
        Print(
	    "asm volatile (\n",
            Unparse(oo, self, 4, 2),
	    ":: ");
	for x in a do
	   Print("\"m\"(",x[1].id,"), ");
	od;
	Print("\"eax\"(X), \"ecx\"(Y));");
    end,

    preprocess := (self, o) >> o,

    # NOTE: dynamically set mul and ofs based on X.t, Y.t
    #        note -- these are in bytes (8 bytes = 64 bits = double fp)
    #switched to double %% for asm volatile
    regmap := rec(
	S := rec(reg:="%%esp",ofs:=-8, mul:=-8),
	X := rec(reg:="%%eax",ofs:=0, mul:=8),
	Y := rec(reg:="%%ecx",ofs:=0, mul:=8),

	A := rec(reg:="%%eax",ofs:=0, mul:=16),
	B := rec(reg:="%%ebx",ofs:=0, mul:=8),
	C := rec(reg:="%%ecx",ofs:=0, mul:=16),

	r0 := rec(reg:="%%xmm0"),
	r1 := rec(reg:="%%xmm1"),
	r2 := rec(reg:="%%xmm2"),
	r3 := rec(reg:="%%xmm3"),
	r4 := rec(reg:="%%xmm4"),
	r5 := rec(reg:="%%xmm5"),
	r6 := rec(reg:="%%xmm6"),
	r7 := rec(reg:="%%xmm7"),
	r8 := rec(reg:="%%xmm8"),
	r9 := rec(reg:="%%xmm9"),
	r10 := rec(reg:="%%xmm10"),
	r11 := rec(reg:="%%xmm11"),
	r12 := rec(reg:="%%xmm12"),
	r13 := rec(reg:="%%xmm13"),
	r14 := rec(reg:="%%xmm14"),
	r15 := rec(reg:="%%xmm15"),

	A0 := rec(reg:="%%xmm8"),
	A1 := rec(reg:="%%xmm9"),
	A2 := rec(reg:="%%xmm10"),
	A3 := rec(reg:="%%xmm11"),
	A4 := rec(reg:="%%xmm12"),
	A5 := rec(reg:="%%xmm13"),
	A6 := rec(reg:="%%xmm14"),
	A7 := rec(reg:="%%xmm15"),
    ),

    startFunc := meth(self, fname, stackofs, loadptrs)
        local v, i;
        PrintLine(".globl ", fname);
#	PrintLine("    .def  _", fname, "; .scl 2; .type 32; .endef");
        PrintLine("    .type  ", fname, ", @function");
	PrintLine("", fname, ":");

	# load arguments into registers
	for i in [1..Length(loadptrs)] do
	    v := loadptrs[i];
	    PrintLine("    movl  ", (stackofs+i-1)*4, "(%esp), ", self.regmap.(v.id).reg);
	od;
    end,
	
    header_top := meth(self, subname, o) 
        local precomputed_data;
	PrintLine("    .data");
	PrintLine("    .align 8");
        precomputed_data := List(Collect(o, data), x->[x.var, x.value]);
	DoForAll(precomputed_data, d -> self.genData(d[1], d[2]));
	PrintLine("");
	PrintLine("    .text");
	PrintLine("    .align 8");
    end,

    header_func := (self, subname, o) >> self.startFunc(subname, 1, [Y, X]),

    header := meth(self, subname, o)
        self.header_top(subname, o);
        self.header_func(subname, o);
    end,

    footer := meth(self, subname, o)
        local init, loopvars;
	PrintLine("    ret");
	PrintLine("");
	self.startFunc(Concat("init_", subname), 0, []);
	PrintLine("    ret");
    end,
    
    genData := (self, v, val) >> Cond(
	val.t = TDouble, PrintLine(v, ": .double ", val.v),
	val.t = TInt,    PrintLine(v, ": .long ", val.v),
	IsArray(val.t) and val.t.t = TDouble,
	    val.t.t=TDouble, PrintLine(v, ": .double ", PrintCS(val.v)),
	IsArray(val.t) and val.t.t = TInt,
	    val.t.t=TDouble, PrintLine(v, ": .long ", PrintCS(val.v)),
        Error("Can't handle type ", val.t)),

    ###################
    # Helpers
    ###################


#HERE is a HACK to print correct GCC intrincsics
#old code is commented out 
    instr2 := (self, mnem, src, dest) >>
#        Print("    ", mnem, "  ", self(src,0,0), ", ", self(dest,0,0), "\n"),
        Print("\"    ", mnem, "  ", self(src,0,0), ", ", self(dest,0,0), "\\n\\t\"\n"),

    isReg := (self, loc) >> IsVar(loc) and IsBound(self.regmap.(loc.id)),

    ####################
    ## General
    ####################
    atomic  := (self,o,i,is) >> Print(o),
    Loc     := (self,o,i,is) >> o.cprint(),

    ####################
    ## Commands
    ####################

    # just sequence them
    #removed the chain label
#    chain := (self,o,i,is) >> Print("chain_", BagAddr(o), ":\n", 
#	DoForAll(o.cmds, c -> self(c, i, is))), 
    chain := (self,o,i,is) >> Print( DoForAll(o.cmds, c -> self(c, i, is))), 


    # all datas are handled in the header, just proceed to children
    data := (self,o,i,is) >> self(o.cmd, i, is),

    # nothing need to be declared in assembly, we just have registers and stack space
    decl := (self,o,i,is) >> self(o.cmd, i, is),

    # on Pentium M using sd seems to be more efficient
    assign     := (self,o,i,is) >> let(
	sfx := When(self.isReg(o.exp) and self.isReg(o.loc), "sd", "sd"), # apd
	self.instr2(Concat("mov", sfx), o.exp, o.loc)),

    assign_add := (self,o,i,is) >> let(	
	sfx := When(self.isReg(o.exp) and self.isReg(o.loc), "sd", "sd"),
	self.instr2(Concat("add",sfx), o.exp, o.loc)),

    assign_sub := (self,o,i,is) >> let(
	sfx := When(self.isReg(o.exp) and self.isReg(o.loc), "sd", "sd"),
	self.instr2(Concat("sub", sfx), o.exp, o.loc)),

    assign_mul := (self,o,i,is) >> let(
	sfx := When(self.isReg(o.exp) and self.isReg(o.loc), "sd", "sd"),
	self.instr2(Concat("mul", sfx), o.exp, o.loc)),

    ####################
    ## Expressions
    ####################

    Exp := (self,o,i,is) >> Error("X86 Assembly backend expects 2-operand low-level representation, it must not have expressions, so I can't handle <o>"),

    dup := (self,o,i,is) >> 
        Print( "shufpd(", self.regmap.(o.args[1].id).reg, ")"),

    nth := (self,o,i,is) >> Checked(IsValue(o.idx), 
	let(d := self.regmap.(o.loc.id),
	    Print( (o.idx.v * d.mul + d.ofs), "(", d.reg, ")"))),

    var := (self,o,i,is) >> When(IsBound(self.regmap.(o.id)),
	Print(self.regmap.(o.id).reg),
	When(IsBound(o.asmpseudoregister),Print("%",o.asmpseudoregister),
	Print(o.id))),

    Value := CUnparser.Value
));
