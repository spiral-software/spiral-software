
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


IsSIMD_ISA := x -> IsRec(x) and IsBound(x.isSIMD_ISA) and x.isSIMD_ISA=true;
IsISA      := x -> IsRec(x) and IsBound(x.isISA) and x.isISA=true;

Declare(ISAOps);
ISAOps := rec(
    operations := OpsOps,
    name := "ISAOps",
    Print := s->s.print(),
    # backward compatibility hack - descendants equal to parent classes
    \= := (c1, c2) -> Same(c1,c2) or (IsISA(c1) and IsISA(c2) and (Same(ObjId(c1),c2) or Same(c1,ObjId(c2)) or c1.id()=c2.id())),
    \< := (c1, c2) -> not ISAOps.\=(c1, c2) and Cond(not IsISA(c1) or not IsISA(c2), BagAddr(c1) < BagAddr(c2),  c1.id()<c2.id()), 
);

Class(ISA, rec(
    verbose := false,
    isISA   := true,
    id      := self >> self.__name__,

    fixProblems := (c,opts) -> c,

    rChildren := self >> [],
    from_rChildren := (self, rch) >> self,

    gt := self >> self.t,

    operations := ISAOps,
    print := self >> Print(self.__name__, When(self._cplx, ".cplx()", "")),

    autolib := rec(
        includes      := () -> [], # list of includes, ex: [ "<pmmintrin.h>" ]
        timerIncludes := () -> [], # list of timer includes
    ),
    # wrap(<spl>) wraps spl to ISA boundaries container (at this moment VContainer)
    wrap := (self, spl) >> spl,
));

Class(SIMD_ISA, ISA, rec(
    isSIMD_ISA := true,

    _cplx := false,

    # isCplx()
    isCplx := self >> self._cplx,

    # cplx()  -- set the complex vectorization flag, which makes .getV() return v/2
    #            which is needed so that VContainer inside complex vectorization 
    #            region correctly resolves its vector length
    cplx := self >> CopyFields(self, rec(_cplx := true)),

    # uncplx()  -- unset the complex vectorization flag, which makes .getV() return v
    #              which is needed so that VContainer inside complex vectorization 
    #              region correctly resolves its vector length
    uncplx := self >> CopyFields(self, rec(_cplx := false)),

    # getV() -- returns the effective vector length that should be used, it is
    #           equal to ISA's vector length normally, but inside complex vector-
    #           ization regions it is vlen/2.
    getV := self >> Cond(self._cplx, self.v/2, self.v),


    getTags := self >> [AVecReg(self)],
    getTagsCx := self >> [AVecRegCx(self)],
    getOpts := self >> self.splopts,
    rules_loaded := false,
    rules_built := false,
    setRules := meth(self, rules)
                    self.rules := rules;
                    self.rules_loaded := true;
                end,
    flushRules := meth(self)
                      self.rules_loaded := false;
                      self.rules_built := false;
                      if IsBound(self.rules) then Unbind(self.rules); fi;
                  end,
    
    # realVect()  -- returns true if real vectorization is allowed
    realVect := True,
    # cplxVect()  -- returns true if complex vectorization is allowed
    cplxVect := True,

    simpIndicesInside := [], # to which instructions we should apply expensive index simplification rules,
                             # these should include gather/scatter instructions
    # ISA atomic data type
    gt := self >> self.t.t,

    wrap := (self, spl) >> spiral.paradigms.vector.sigmaspl.VContainer(spl, self),

    #F .loadCont(<n>, <y>, <yofs>, <x>, <xofs>, <xofs_align>)
    #F
    #F Read <n> values from address <x> + <xofs> into (lower) slots in vector pointer at <y> + <yofs>
    #F
    #F <n> - integer, how many points to load, must be <= self.v
    #F <y> - destination pointer
    #F <yofs> - destination offset in vectors
    #F <x> - source pointer
    #F <xofs> - source offset
    #F <xofs_align> - alignment, must be (xofs mod self.v), this parameter allows us
    #F                to pass in a simplified expression which may be constant
    #F
    #F Assumptions: 1 <= n <= self.v
    #F              xofs_align = xofs mod self.v
    #F 
    #F When n = self.v, this operation becomes an unaligned load (unless xofs_align=0)
    #F
    loadCont := (self, n, y, yofs, x, xofs, xofs_align, opts) >> let(
	a  := _unwrap(xofs_align),
	nn := _unwrap(n), 
	yy := vtref(self.t, y, yofs),
        When(IsBound(self.loadc_align) and IsInt(a) and not IsUnalignedPtrT(x.t),
	     self.loadc_align(nn, a, opts)(yy, x, xofs),
	     self.loadc(nn, opts)(yy, nth(x, xofs)))),

    #F .storeCont(<n>, <y>, <yofs>, <yofs_align>, <x>, <xofs>)
    #F
    #F Store (lower) <n> values from address <x> + <xofs> into address <y> + <yofs>
    #F
    #F <n> - integer, how many points to store, must be <= self.v
    #F <y> - destination pointer
    #F <yofs> - destination offset
    #F <yofs_align> - alignment, must be (yofs mod self.v), this parameter allows us
    #F                to pass in a simplified expression which may be constant
    #F <x> - source pointer
    #F <xofs> - source offset in vectors
    #F 
    #F Assumptions: 1 <= n <= self.v
    #F              yofs_align = yofs mod self.v
    #F 
    #F When n = self.v, this operation becomes an unaligned load (unless yofs_align=0)
    #F
    storeCont := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
	a  := _unwrap(yofs_align),
	nn := _unwrap(n),
	xx := vtref(self.t, x, xofs),
        When(IsBound(self.storec_align) and IsInt(a) and not IsUnalignedPtrT(y.t), 
	     self.storec_align(nn, a, opts)(y, yofs, xx), # storec_align is not implemented anywhere
	     self.storec[nn](nth(y, yofs), xx))),

    storeContAcc := (self, n, y, yofs, yofs_align, x, xofs, opts) >> let(
	a  := _unwrap(yofs_align),
	nn := _unwrap(n),
	xx := vtref(self.t, x, xofs),
	t  := TempVec(TArray(xx.t.t, xx.t.size)),
	decl([t], chain(
	    self.loadCont(nn, t, 0, y, yofs, yofs_align, opts), 
	    assign(vtref(self.t, t, 0), vtref(self.t, t, 0) + xx), 
	    self.storeCont(nn, y, yofs, yofs_align, t, 0, opts)))),

    # ===============================================================================
    # Other fields that must be defined in subclasses

    # Required:
    #  active
    #  bin_shl1  bin_shl2  bin_shr1  bin_shr2  bin_shrev
    #  bits, ctype, t, v
    #  countrec
    #  includes
    #  info
    #  instr
    #  isFixedPoint, isFloat
    #  loadc + loadc_align (optional)  OR  loadCont
    #  mul_cx, mul_cx_conj
    #  reverse
    #  splopts
    #  storec  OR  storeCont
    #  svload
    #  svstore
    #  RCVIxJ2
    #  dupload, duploadn
    #  hadd
    #  swap_cx
    #  vzero

    # Fixed point:
    #  fracbits
    #  saturatedArithmetic

    # Viterbi:
    #  interleavedmask, hmin, average (?), isSigned

    # Hacks (used in 2x32f), supported but not required:
    #  loadop,      requireLoad
    #  storeop,     requireStore
    #  scalarVar,   needScalarVarFix 

));

IsSIMD_ISA := x -> IsRec(x) and IsBound(x.isSIMD_ISA) and x.isSIMD_ISA=true;
IsISA      := x -> IsRec(x) and IsBound(x.isISA) and x.isISA=true;

Class(SIMD_ISA_DB, rec(
    verbose := false,
    isa_db := rec(),
    addISA := meth(self, isa) self.isa_db.(isa.name) := isa; end,
    installed := self >> Filtered(RecFields(self.isa_db), x -> not IsSystemRecField(x)),
    active := self >> List(Filtered(Filtered(RecFields(self.isa_db), x -> not IsSystemRecField(x)), e->self.isa_db.(e).active), k->self.isa_db.(k)),
    info := self >> Print("\nSpiral SIMD ISA database\n",
                          "installed ISAs: ", PrintCS(self.installed()), "\n",
                          "active ISAs: ", PrintCS(self.active()), "\n"),
    getISA := (self, isa) >> self.isa_db.(isa),
#------------------------------------------
    HASH_FILE := file -> let(p := Conf("path_sep"), base := Conf("spiral_dir"), Concat(base, p, "namespaces", p, "spiral", p, "platforms", p, "_", file, "_generated1.gi")),
    RULES_FILE := file -> let(p := Conf("path_sep"), base := Conf("spiral_dir"), Concat(base, p, "namespaces", p, "spiral", p, "platforms", p, "_", file, "_generated0.gi")),
    hash := HashTableDP(),
    hashFlush := meth(self) self.hash := HashTableDP(); end,
    hashSave := meth(self)
                    local _entry, entry, item, isa;
                    if self.verbose then Print("saving hash\n"); fi;
                    _entry := Flat(Filtered(self.hash.entries, True));

                    for isa in self.active() do
                        PrintTo(self.HASH_FILE(isa.file), "");
                    od;

                    for isa in self.active() do
#                        Error();
                        for entry in Filtered(_entry, (a) -> a.key.getTags()[1].isa = isa)  do
#                            for item in entry  do
                            item := entry; #only save first hash entry
                            AppendTo(self.HASH_FILE(isa.file), self.name, ".hashAdd(", item.key, ", ", item.data, ");\n");
#                            od;
                        od;
                    od;
                end,
    hashAdd := meth(self, a, b) HashAdd(self.hash, a,b); end,
    getHash:= self >> Copy(self.hash),
    hash_rebuilt := false,
    rules_rebuilt := false,
    init0 := meth(self)
        local isa;
        self.rules_rebuilt := false;
        if self.verbose then Print("\n"); fi;
        for isa in self.active() do
            if self.verbose then Print(isa, ": base cases...\n"); fi;
            if not IsBound(isa.rules) then
                Print(isa, " rules are being rebuilt and saved...\n");
                isa.buildRules();
                self.rules_rebuilt := true;
            fi;
        od;
        if self.rules_rebuilt then self.saveRules(); fi;
    end,
    init1 := meth(self)
        local isa;
        self.hash_rebuilt := false;
        if self.verbose then Print("\n"); fi;
        for isa in self.active() do
            if self.verbose then Print(isa, ": TL hash...\n"); fi;
            if not self.checkBases(isa) then
                Print(isa, " hash is being rebuilt and saved...\n");
                self.buildBases(isa);
                self.hash_rebuilt := true;
            fi;
        od;
        if self.hash_rebuilt then self.hashSave(); fi;
    end,
    saveRules := meth(self)
                    local isa;
                    if self.verbose then Print("saving rules\n"); fi;
                    paradigms.vector.sigmaspl.VPerm.plong();

                    for isa in self.active() do
                        PrintTo(self.RULES_FILE(isa.file), "");
                    od;

                    for isa in self.active() do
                        if IsBound(isa.rules) then
                            AppendTo(self.RULES_FILE(isa.file), isa.name, ".setRules(", isa.rules, ");\n");
                        fi;
                    od;
                    paradigms.vector.sigmaspl.VPerm.pshort();
                 end,
    reset := meth(self)
                local isa, verb;
                verb := self.verbose;
                for isa in self.active() do
                    PrintTo(self.RULES_FILE(isa.file), "");
                    PrintTo(self.HASH_FILE(isa.file), "");
                od;
                self.verbose := true;
                self.hashFlush();
                for isa in self.active() do
                    isa.flushRules();
                od;
                self.init0();
                self.init1();
                self.verbose := verb;
             end,
    required_bases := isa -> Concat(
            When(isa.realVect(), [
                arch -> TL(2*arch.v,2,1,1).withTags(arch.getTags()),
                arch -> TL(2*arch.v,arch.v,1,1).withTags(arch.getTags()),
                arch -> TL(arch.v^2,arch.v,1,1).withTags(arch.getTags()) ], []),
            When(isa.cplxVect(), [
                arch -> TL(arch.v^2/4,arch.v/2,1,2).withTags(arch.getTags()) ], []),
            When(isa.cplxVect() and isa.v=8, [
                arch -> TL(4,2,1,2).withTags(arch.getTags()) ], [])),
    getBases := (self, isa) >> List(self.required_bases(isa), i-> i(isa)),
    lookupBases := (self, isa) >> List(self.getBases(isa), i-> HashLookup(self.hash, i)),
    checkBases := (self, isa) >> not ForAny(self.lookupBases(isa), i -> i=false or i=[])
));

