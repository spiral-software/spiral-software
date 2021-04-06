
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Cell Unparser
# Author: schellap

@TInt := @.cond(x->x.t=TInt);
@TRealOrInt := @.cond(x->x.t=TInt or x.t=TReal);
@TReal := @.cond(x->x.t=TReal);
@TVect := @.cond(x->ObjId(x.t)=TVect);

Class(CellUnparser, CMacroUnparserProg, rec(

    func_ppe := (self,o,i,is) >> "",

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
           Print("{", self.infix(o.v, ", "), "}"),
       o.t = TString,
           #Print("\"", o.v, "\"")
           Print(o.v),
       o.t = TBool, Print(When(o.v, "1", "0"))

    ),

    vdup         := (self, o, i, is) >> Print("spu_splats((", self.opts.vector.isa.ctype, ")", self(o.args[1], i, is), ")"),
    vsplat_8x16i := (self, o, i, is) >> Print("spu_splats((", self.opts.vector.isa.ctype, ")", self(o.args[1], i, is), ")"),
    vsplat_4x32f := (self, o, i, is) >> Print("spu_splats((", self.opts.vector.isa.ctype, ")", self(o.args[1], i, is), ")"),
    vsplat_2x64f := (self, o, i, is) >> Print("spu_splats((", self.opts.vector.isa.ctype, ")", self(o.args[1], i, is), ")"),

    # Declarations
    TVect := (self, t, vars, i, is) >> let(ctype := self.ctype(t, self.opts.vector.isa), 
              Print(ctype, " ",
              self.infix(vars, ", "))),

    TVectPointer := (self, t, vars, i, is) >> let(ctype := self.ctype(t, self.opts.vector.isa), 
              Print(ctype,
              When(IsBound(self.opts.useMemoryArena) and self.opts.useMemoryArena, "* ", " "),
              self.infix(vars, ", "))),

    TReal := ~.TVect, 
    TRealPointer := ~.TVectPointer,

    TInt := (self, t, vars, i, is) >> Print("int ", self.infix(vars, ", ")),

    # Arithmetic
    # --------------------------------
    # -- mul -- 
    mul := (self, o, i, is) >> CondPat(o, 
    [mul, @TReal, @TVect], 
        Cond(self.opts.vector.isa.v = 2,
            self.printf("spu_mul(((vector $1){$2,$2}), $3)",       [self.ctype(o.args[1].t, self.opts.vector.isa), o.args[1], o.args[2]]),
        self.opts.vector.isa.v = 4,
            self.printf("spu_mul(((vector $1){$2,$2,$2,$2}), $3)", [self.ctype(o.args[1].t, self.opts.vector.isa), o.args[1], o.args[2]]),
        Error("Don't know how to unparse vector arch of length: ", self.opts.vector.isa.v)
        ),
    [mul, @TVect,   @TVect], 
        self.printf("spu_mul($1, $2)",              [o.args[1], o.args[2]]),
    [mul, @TRealOrInt, @TRealOrInt],  
        self.printf("($1 * $2)", o.args),
    [mul, @TRealOrInt, @TRealOrInt, @TRealOrInt], 
        self.printf("($1 * $2 * $3)", o.args),
    Error("Don't know how to unparse <o>. Unrecognized type combination: ", o)),

    # -- add -- 
    add := (self, o, i, is) >> When(Length(o.args) > 2, 
       self(_computeExpType(add(o.args[1], _computeExpType(ApplyFunc(add, Drop(o.args, 1))))), i, is), 
       CondPat(o, 
           [add, @TVect,   @TVect], 
              self.printf("spu_add($1, $2)", [o.args[1], o.args[2]]),
           #[add, @TRealOrInt, @TRealOrInt],  
           [add, @TVect,   @], 
              self.printf("spu_add($1, $2)", [o.args[1], o.args[2]]),
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
              self.printf("spu_sub($1, $2)", [o.args[1], o.args[2]]),
           [sub, @TVect,   @], 
              self.printf("spu_sub($1, $2)", [o.args[1], o.args[2]]),
           [sub, @TRealOrInt, @TRealOrInt],  
              self.printf("($1 - $2)", o.args),
           [sub, @, @],  
              self.printf("($1 - $2)", o.args),
           Error("Don't know how to unparse <o>. Unrecognized type combination")
       )
    ),

    # -- neg -- 
    neg := (self, o, i, is) >> CondPat(o,
       #[neg, @TVect],
       #    self(o.t.value(Replicate(o.t.size,0)) - o.args[1], i, is),
       #self.printf("(-$1)", o.args)
       [neg, @TVect], Cond( self.opts.vector.isa.v = 2, self.printf("_negated2($1)", o.args),
                            self.opts.vector.isa.v = 4, self.printf("_negatef4($1)", o.args),
                            Error("What kind of vector length are you trying to pull on me?")
                ),
        self.printf("(-$1)", o.args)
    ),

    v_neg01 := (self, o, i, is) >> self.printf("_negated2($1)", o.args),


    vpack := (self, o, i, is) >> Print(self.opts.vector.isa.vconstv, "{", self.infix(o.args, ", "), "}"),

    # Spiral: fma(rt, x, y, z)    -> rt = x + (y*z)
    # SPU:    FMA(rt, ra, rb, rc) -> rt = (ra*rb) + rc
    # -- fma --
    fma  := (self, o, i, is) >> CondPat(o,
       [fma, @TVect, @TVect, @TVect],
          self.printf("spu_madd($2, $3, $1)",  [o.args[1], o.args[2], o.args[3]]),
       [fma, @TVect, @TReal, @TVect],
          self.printf("spu_madd(((vector $4){$2,$2,$2,$2}), $3, $1)",  [o.args[1], o.args[2], o.args[3], self.ctype(o.args[2].t, self.opts.vector.isa)]),
       [fma, @TReal, @TReal, @TReal],
          self.printf("$1 + ($2 * $3)", [o.args[1], o.args[2], o.args[3]]),
        Error("Don't know how to unparse fma instruction <o>. Unrecognized type combination:", o.args[1].t, o.args[2].t, o.args[3].t)
    ),

    # Spiral: fms(rt, x, y, z)    -> rt = x - (y*z)
    # SPU:    FNMS(rt, ra, rb, rc) -> rt = rc - (ra*rb)
    fms  := (self, o, i, is) >> CondPat(o,
       [fms, @TVect, @TVect, @TVect],
          self.printf("spu_nmsub($2, $3, $1)",  [o.args[1], o.args[2], o.args[3]]),
       [fms, @TVect, @TReal, @TVect],
          self.printf("spu_nmsub(((vector $4){$2,$2,$2,$2}), $3, $1)",  [o.args[1], o.args[2], o.args[3], self.ctype(o.args[2].t, self.opts.vector.isa)]),
       [fms, @TInt, @TVect, @TVect], #HACK
          self.printf("spu_nmsub($2, $3, (($4){$1,$1,$1,$1}))",  [o.args[1].v, o.args[2], o.args[3], self.ctype(o.args[2].t, self.opts.vector.isa)]),
       [fms, @TInt, @TReal, @TVect], #HACK
          self.printf("spu_nmsub(((vector $4){$2,$2,$2,$2}), $3, ((vector $4){$1,$1,$1,$1}))",  [o.args[1].v, o.args[2], o.args[3], self.ctype(o.args[2].t, self.opts.vector.isa)]),
       [fms, @TReal, @TReal, @TReal],
          self.printf("$1 + ($2 - $3)", [o.args[1], o.args[2], o.args[3]]),
        Error("Don't know how to unparse fms instruction <o>. Unrecognized type combination")
    ),

    # Spiral: nfma(rt, x, y, z)   -> rt = (y*z) - x
    # SPU:    FMS(rt, ra, rb, rc) -> rt = (ra*rb) - rc
    #NOTE: Change above and below vector constants to spu_splats so the v=2 or v=4 cases are both implicitly taken care of
    nfma := (self, o, i, is) >> CondPat(o,
       [nfma, @TVect, @TVect, @TVect],
          self.printf("spu_msub($2, $3, $1)",  [o.args[1], o.args[2], o.args[3]]),
       [nfma, @TVect, @TReal, @TVect],
          self.printf("spu_msub(((vector $4){$2,$2,$2,$2}), $3, $1)",  [o.args[1], o.args[2], o.args[3], self.ctype(o.args[2].t, self.opts.vector.isa)]),
        Error("Don't know how to unparse nfma instruction <o>. Unrecognized type combination")
    ),

    # logic
    # --------------------------------
    bin_and := (self, o, i, is) >> self.prefix("spu_and", o.args),

    bin_or := (self, o, i, is) >> self.prefix("spu_or", o.args),

    # comparison
    # --------------------------------
    eq := (self, o, i, is) >> CondPat(o,
        [eq, @TVect, @], self.printf("spu_cmpeq($1, spu_splat($2))", o.args),
        [eq, @, @TVect], self.printf("spu_cmpeq(spu_splat($1), $2)", o.args),
        [eq, @, @], self.printf("(($1) == ($2))", o.args)
        ),
    cmpg := (self, o, i, is) >> CondPat(o,
        [eq, @TVect, @], self.printf("spu_cmpgt($1, spu_splat($2))", o.args),
        [eq, @, @TVect], self.printf("spu_cmpgt(spu_splat($1), $2)", o.args),
        [eq, @, @], self.printf("(($1) > ($2))", o.args)
        ),
    cmpl := (self, o, i, is) >> CondPat(o,
        [eq, @TVect, @], self.printf("spu_LT_IS_EXPENSIVE($1, spu_splat($2))", o.args),
        [eq, @, @TVect], self.printf("spu_LT_IS_EXPENSIVE(spu_splat($1), $2)", o.args),
        [eq, @, @], self.printf("(($1) < ($2))", o.args)
        ),

    promote_spu8x16i := (self, o, i, is) >> self.prefix("spu_promote", o.args),
    promote_spu4x32f := (self, o, i, is) >> self.prefix("spu_promote", o.args),
    promote_spu2x64f := (self, o, i, is) >> self.prefix("spu_promote", o.args),

    extract_spu8x16i := (self, o, i, is) >> self.prefix("spu_extract", o.args),
    extract_spu4x32f := (self, o, i, is) >> self.prefix("spu_extract", o.args),
    extract_spu2x64f := (self, o, i, is) >> self.prefix("spu_extract", o.args),

    insert_spu8x16i  := (self, o, i, is) >> self.prefix("spu_insert",  o.args),
    insert_spu4x32f  := (self, o, i, is) >> self.prefix("spu_insert",  o.args),
    insert_spu2x64f  := (self, o, i, is) >> self.prefix("spu_insert",  o.args),

    vparam_spu := (self, o, i, is) >>
      Print("((vector unsigned char){", PrintCS(prep_perm_string_spu(o.p)), "})"),

   vzero_8x16i := (self, o, i, is) >> Print("((", self.opts.vector.isa.vtype ,"){", PrintCS(List([1..self.opts.vector.isa.v], i->0)), "})"),
   vzero_4x32f := (self, o, i, is) >> Print("((", self.opts.vector.isa.vtype ,"){", PrintCS(List([1..self.opts.vector.isa.v], i->0)), "})"),
   vzero_2x64f := (self, o, i, is) >> Print("((", self.opts.vector.isa.vtype ,"){", PrintCS(List([1..self.opts.vector.isa.v], i->0)), "})"),

   # ----------------------------------------------------------------------------------
   # ISA specific : spu_8x16i
   # ----------------------------------------------------------------------------------
   vperm_8x16i_spu := (self, o, i, is) >>
     self.printf("spu_shuffle($1, $2, $3)", o.args),

   vuperm_8x16i_spu := (self, o, i, is) >>
     self.printf("spu_shuffle($1, $1, $2)", o.args),

   vloadu8_spu8x16i := (self, o, i, is) >> 
   Print("spu_or(spu_slqwbyte(*((vector signed short*) (&(",self(o.args[1],i,is),"))), (unsigned) ((vector signed short*) (&(",self(o.args[1],i,is),"))) & 15), spu_rlmaskqwbyte(*(((vector signed short*) (&(",self(o.args[1],i,is),")))+1), ((unsigned) ((vector signed short*) (&(",self(o.args[1],i,is),"))) & 15)-16))"),

   # ----------------------------------------------------------------------------------
   # ISA specific : spu_4x32f
   # ----------------------------------------------------------------------------------
   vperm_4x32f_spu := (self, o, i, is) >>
     self.printf("spu_shuffle($1, $2, $3)", o.args),

   vuperm_4x32f_spu := (self, o, i, is) >>
     self.printf("spu_shuffle($1, $1, $2)", o.args),

   #vloadu1,2,4 should produce:
   #spu_shuffle(spu_slqwbyte((*((vector float *)(&X[element]))), (unsigned int)(&X[element]) & 15), ((vector float){0,0,0,0}), ((vector unsigned char){0, 1, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}));
   #spu_shuffle(spu_promote(X[element], 0), spu_promote(X[element+1], 0), ((vector unsigned char){0, 1, 2, 3, 16, 17, 18, 19, 128, 128, 128, 128, 128, 128, 128, 128}));
   #spu_or(spu_slqwbyte(*ptr,            (unsigned) ptr & 15), spu_rlmaskqwbyte(*(ptr+1), ((unsigned) ptr & 15)-16));

   #NOTE: Both these are HACKs!
   slqwbyte_spu4x32f := (self, o, i, is) >>
       self.printf("spu_slqwbyte(*$1, (unsigned) ($1) & 15)", o.args),

   rlmaskqwbyte_spu4x32f := (self, o, i, is) >>
       self.printf("spu_rlmaskqwbyte(*$1, ((unsigned) $1 & 15)-16)", o.args),

   vloadu_spu4x32f := (self, o, i, is) >> 
       Print("spu_or(spu_slqwbyte(*",self(o.args[1],i,is),", (unsigned) ",self(o.args[1],i,is)," & 15), spu_rlmaskqwbyte(*(",self(o.args[1],i,is),"+1), ((unsigned) ",self(o.args[1],i,is)," & 15)-16))"),

   vloadu4_spu4x32f := (self, o, i, is) >> 
   Print("spu_or(spu_slqwbyte(*((vector float*) (&(",self(o.args[1],i,is),"))), (unsigned) ((vector float*) (&(",self(o.args[1],i,is),"))) & 15), spu_rlmaskqwbyte(*(((vector float*) (&(",self(o.args[1],i,is),")))+1), ((unsigned) ((vector float*) (&(",self(o.args[1],i,is),"))) & 15)-16))"),

# spu_or(
#       spu_slqwbyte(*((vector float*) (&(",ptr,"))), (unsigned) ((vector float*) (&(",ptr,"))) & 15),
#       spu_rlmaskqwbyte(*(((vector float*) (&(",ptr,")))+1), ((unsigned) ((vector float*) (&(",ptr,"))) & 15)-16))

   #spu_or(spu_slqwbyte(*ptr,            (unsigned) ptr & 15), spu_rlmaskqwbyte(*(ptr+1), ((unsigned) ptr & 15)-16));

   # ----------------------------------------------------------------------------------
   # ISA specific : spu_2x64f
   # ----------------------------------------------------------------------------------

   vperm_2x64f_spu := (self, o, i, is) >>
     self.printf("spu_shuffle($1, $2, $3)", o.args),

   vuperm_2x64f_spu := (self, o, i, is) >>
     self.printf("spu_shuffle($1, $1, $2)", o.args),

   # ----------------------------------------------------------------------------------
   # Cell Parallel unparser
   # ----------------------------------------------------------------------------------
   extraHeader := "TODO",
#Concat("#include <spu_mfcio.h>\n\
#extern spe_infostruct spe_info;\n\
#//extern DATATYPE_NO_QUOTES* Xalt;\
#//extern DATATYPE_NO_QUOTES* Yalt;\
#extern DATATYPE_NO_QUOTES* XYalt;\
#extern mfc_list_element_t gathlist[2048], scatlist[2048];\
##define Xalt XYalt\
##define Yalt XYalt\
#/* extern volatile int writeSignal[4];\n\
#extern int check;\n\
#extern void sig_barrier();\n\
#n*/\n\
##include \"spumacros.h\"\n\
#// Declare memory arena\
##ifdef ARENA_SIZE\
#  __attribute__((aligned(16))) DATATYPE_NO_QUOTES ARENA[ARENA_SIZE];\
#  int arenalevel = ARENA_SIZE;\
##endif\
#\n\
#\n\n\n"),



   dist_loop := meth(self, o, i, is)
       Print(Blanks(i),    "{\n");
       #Print(Blanks(i+is), "unsigned int ", o.var, " = spe_info.spuid;\n"); # Not needed since spuid is now a param
       self(o.cmd,i+is,is);
       Print(Blanks(i),    "}\n");
       #Print(Blanks(i),    "//ALL_TO_ALL_BARRIER;\n");
       #Print(Blanks(i),    "BLOCK_ON_READ();\n");
    end,

    #NOTE: this should really be just a berrier (shouldn't have
    #BLOCK_ON_CPUDMA). But leaving this in there for legacy compatibility.
    #Should'nt affect performance significantly.

    dist_barrier := (self,o,i,is) >> Print(Blanks(i), "BLOCK_ON_CPUDMA; ALL_TO_ALL_BARRIER;\n"),

    dma_barrier := (self,o,i,is) >> Print(Blanks(i), "BLOCK_ON_CPUDMA;\n"),

    #NOTE: might be hacks. Need to possibly fix inside parent unparser's methods.
    call := (self, o, i, is) >> Print(Blanks(i), o.args[1].id, self.pinfix(Drop(o.args, 1), ", "), ";\n"),

    fcall := (self, o, i, is) >> Print(self(o.args[1],0,0), "(", self.infix(Drop(o.args, 1), ", "), ")"),

    #fcall_addr := (self, o, i, is) >> Print(self(o.args[1],0,0), "(", self.infix(Drop(o.args, 1), ", "), ")"),

   # ----------------------------------------------------------------------------------
   # Cell Multibuffering unparser
   # ----------------------------------------------------------------------------------
    multibuffer_loop := meth(self, o, i, is) 
       local v, lo, hi, measSteadyState, swapx, swapy, swapxy, block_on_diag, swap_twiddles, n;

       n  := Length(o.range);
       v  := o.var;
       lo := o.range[1] + 1;
       hi := Last(o.range) - 1;
       measSteadyState := When(
            IsBound(self.opts.measSteadyState) and self.opts.measSteadyState = true,
            true,
            false);

       #swapx := Concat("SWAP(", o.x.id, ", ", o.x.id, "alt);\n");
       #swapy := Concat("SWAP(", o.y.id, ", ", o.y.id, "alt);\n");

       swapx  := Concat("SWAP(", o.x.id, ", XYalt);\n");
       swapy  := Concat("SWAP(", o.y.id, ", XYalt);\n");
       swapxy := Concat("SWAP(", o.y.id, ", ", o.x.id, ");\n");


       swap_twiddles := When(o.twiddles=[], "\n",
            Concat("SWAP(", o.bufs[1].id, ", ", o.bufs[2].id, ");\n")
       );

       block_on_diag := When(o.twiddles=[], "\n", "BLOCK_ON_DIAG;\n");

       Print(

       Blanks(i), When(measSteadyState, "/*", ""),
       Blanks(i), "\n{// ------------- Multibuffer header begin -----------\n",
       Blanks(i+is), "int ", v, " = ", lo-1, ";\n",
       # Begin skewiter:
       Blanks(i+is), v, "= (", v, "+(spuid/  ( SPUS>",n," ? (SPUS/",n,") : 1  )  ))%", n, ";\n",


       Blanks(i+is), "{\n",
       self(o.gathmem, i+is, is),
       When(o.twiddles=[], "", self(o.twiddles, i+is, is)),
       Blanks(i+is), "}\n",


       Blanks(i+is), "BLOCK_ON_MEMDMA;\n",
       Blanks(i+is), block_on_diag,
       Blanks(i+is), swapx,
       Blanks(i+is), swap_twiddles,


       Blanks(i+is), "{\n",
       # skewiter increment
       Blanks(i+is), v, "= (", v, "+1) % ", n, ";\n",
       #Blanks(i+is), v, "++;\n",
       self(o.gathmem, i+is, is),
       When(o.twiddles=[], "", self(o.twiddles, i+is, is)),
       # skewiter decrement
       Blanks(i+is), v, "= (", v, "+", n, "-1) % ", n, ";\n",
       #Blanks(i+is), v, "--;\n",
       Blanks(i+is), "}\n",


       Blanks(i+is), "{ // Loopbody begin\n",
       When(IsBound(self.opts.mbuf_nobody) and self.opts.mbuf_nobody = true, "", self(o.cmd,i+is,is)),
       Blanks(i+is), "} // Loopbody end\n",
       Blanks(i+is), "BLOCK_ON_MEMDMA;\n",
       Blanks(i+is), block_on_diag,
       Blanks(i), "}// ------------- Multibuffer header end -----------\n\n",
       Blanks(i), When(measSteadyState, "*/", ""),

       Blanks(i), "for(int ", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) {\n",
       # Begin skewiter:
       Blanks(i+is), v, "= (", v, "+(spuid/  ( SPUS>",n," ? (SPUS/",n,") : 1  )  ))%", n, ";\n",
       Blanks(i+is), swapy,
       Blanks(i+is), swapxy,
       # skewiter decrement
       Blanks(i+is), v, "= (", v, "+", n, "-1) % ", n, ";\n",
       #Blanks(i+is), v, "--;\n",
       self(o.scatmem, i+is, is),
       # skewiter increment
       Blanks(i+is), v, "= (", v, "+1) % ", n, ";\n",
       #Blanks(i+is), v, "++;\n",
       #Blanks(i+is), swapx,
       Blanks(i+is), swap_twiddles,
       # skewiter increment
       Blanks(i+is), v, "= (", v, "+1) % ", n, ";\n",
       #Blanks(i+is), v, "++;\n",
       self(o.gathmem, i+is, is),
       When(o.twiddles=[], "", self(o.twiddles, i+is, is)),
       # skewiter decrement
       Blanks(i+is), v, "= (", v, "+", n, "-1) % ", n, ";\n",
       #Blanks(i+is), v, "--;\n",
       Blanks(i+is), "{ // Loopbody begin\n",
       When(IsBound(self.opts.mbuf_nobody) and self.opts.mbuf_nobody = true, "", self(o.cmd,i+is,is)),
       Blanks(i+is), "} // Loopbody end\n",
       Blanks(i+is), "BLOCK_ON_MEMDMA;\n",
       Blanks(i+is), block_on_diag,
       # End skewiter:
       Blanks(i+is), v, "= (", v, "+", n, "-(spuid/  ( SPUS>",n," ? (SPUS/",n,") : 1  )  ))%", n, ";\n",
       Blanks(i), "}\n",

       Blanks(i), When(measSteadyState, "/*", ""),
       Blanks(i), "\n{// ------------- Multibuffer footer begin -----------\n",
       Blanks(i+is), "int ", v, "=", hi, "+1;\n",
       # Begin skewiter:
       Blanks(i+is), v, "= (", v, "+(spuid/  ( SPUS>",n," ? (SPUS/",n,") : 1  )  ))%", n, ";\n",
       Blanks(i+is), swapy,
       Blanks(i+is), swapxy,
       # skewiter decrement
       Blanks(i+is), v, "= (", v, "+", n, "-1) % ", n, ";\n",
       #Blanks(i+is), v, "--;\n",


       Blanks(i+is), "{\n",
       self(o.scatmem, i+is, is),
       Blanks(i+is), "}\n",


       # skewiter increment
       Blanks(i+is), v, "= (", v, "+1) % ", n, ";\n",
       #Blanks(i+is), v, "++;\n",
       #Blanks(i+is), swapx,
       Blanks(i+is), swap_twiddles,
       Blanks(i+is), "{ // Loopbody begin\n",
       When(IsBound(self.opts.mbuf_nobody) and self.opts.mbuf_nobody = true, "", self(o.cmd,i+is,is)),
       Blanks(i+is), "} // Loopbody end\n",
       Blanks(i+is), "BLOCK_ON_MEMDMA;\n",
       Blanks(i+is), swapy,
       #Blanks(i+is), v, "++;\n",

       Blanks(i+is), "{\n",
       self(o.scatmem, i+is, is),
       Blanks(i+is), "}\n",

       Blanks(i+is), "BLOCK_ON_MEMDMA;\n",
       Blanks(i), "}// ------------- Multibuffer footer end -----------\n\n",
       Blanks(i), When(measSteadyState, "*/", "")
       );

       #NOTE: Last wait can be avoided by DMA_putting Y directly (same for X?)

    end,


#F For debugging:
#F Multibuffer_loop that doesn't do multibuffering. Easier for humans to parse when looking at C code
    mem_loop := meth(self, o, i, is) 
       local v, lo, hi, swapx, swapy, block_on_diag, swap_twiddles;

       swapx := Concat("SWAP(", o.x.id, ", ", o.x.id, "alt);\n");
       swapy := Concat("SWAP(", o.y.id, ", ", o.y.id, "alt);\n");
       swap_twiddles := When(o.twiddles=[], "\n",
            Concat("SWAP(", o.bufs[1].id, ", ", o.bufs[2].id, ");\n")
       );

       v  := o.var;
       lo := o.range[1];
       hi := Last(o.range);


       block_on_diag := When(o.twiddles=[], "\n", "BLOCK_ON_DIAG;\n");

       Print(


       Blanks(i), "for(int ", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) {\n",

       self(o.gathmem, i+is, is),
       When(o.twiddles=[], "", self(o.twiddles, i+is, is)),
       Blanks(i+is), "BLOCK_ON_MEMDMA;\n",
       Blanks(i+is), block_on_diag,
       Blanks(i+is), swapx,
       Blanks(i+is), swap_twiddles,

       Blanks(i+is), "{ // Loopbody begin\n",
       When(IsBound(self.opts.mbuf_nobody) and self.opts.mbuf_nobody = true, "", self(o.cmd,i+is,is)),
       Blanks(i+is), "} // Loopbody end\n",

       Blanks(i+is), swapy,
       self(o.scatmem, i+is, is),
       Blanks(i+is), "BLOCK_ON_MEMDMA;\n",

       Blanks(i), "}\n"


       );

    end,



    kern := (self, o, i, is) >> self(o.cmd, i, is)

));

Class(CellUnparser_parallel, CellUnparser);

