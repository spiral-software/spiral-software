
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


####################################################
##
## NOTE: Chained unparser!
##

_bitsInt := (bits) -> Sum(List([1..Length(bits)], i -> bits[i]*2^(i-1)));


_sklr_unaligned_store := (bits) -> meth(self, o, i, is )
    local loc0, loc1, offs, p, loffs, mask, l;

    if  IsUnalignedPtrT(o.loc.t) then
        Error("cannot do unaligned store when base pointer is not aligned");
    fi;

    loc0 := vtref(o.exp.t, o.loc, idiv(o.offs, bits));
    loc1 := vtref(o.exp.t, o.loc, idiv(o.offs, bits)+1);
    offs := _unwrap(o.offs) mod bits;
    p    := _unwrap(o.p);
    loffs:= p+offs-bits;
    mask := (n) -> o.exp.t.value(Replicate(n, 1) :: Replicate(bits-n, 0));
    l    := Cond(
                (IsSymbolic(offs) and (p>1)) or (not IsSymbolic(offs) and (offs + p > bits)),
                          # a0 = a0 xor (( a0 xor (val << offs)) & (mask << offs));
                          # a1 = (offs + p <= bits) ? a1 :
                          #          ((a1 >> loffs) << loffs) | ((val & mask) >> bits-offs)
                        [ assign(loc0, bin_xor(loc0, bin_and(bin_shl(mask(p), offs), bin_xor(loc0, bin_shl(o.exp, offs))))),
                          assign(loc1, cond(leq(offs + p, bits), loc1, bin_or(bin_shl(bin_shr(loc1, loffs), loffs), bin_shr(bin_and(o.exp, mask(p)), bits-offs))))],
                p = bits and offs = 0,
                        [ assign(loc0, o.exp) ],
                #else
                        [ assign(loc0, bin_xor(loc0, bin_and(bin_shl(mask(p), offs), bin_xor(loc0, bin_shl(o.exp, offs))))) ]);
    return Print(Blanks(i), self.infix(ESReduce(l, self.opts), Blanks(i)));
end;

Class(SKLRBUnparser, SSEUnparser, rec(
   # none exp must be gone before it reaches unparser
   noneExp := (self, o, i, is) >> self(o.args[1].zero(), i, is),

   utype := (self, t) >> (ObjId(t) = BitVector or (ObjId(t) = TVect and t.t in  [T_Int(1), T_UInt(1), TInt, TUInt])) 
                         and t.size in [16, 32, 64],

   BitVector := (self, t, vars, i, is) >> self.T_UInt(T_UInt(t.size), vars, i, is),

   Value := (self, o, i, is) >> When( self.utype(o.t), 
        #self(T_UInt(o.t.size).value(_bitsInt(o.v)), i, is), 
        Print("0x", HexStringInt(_bitsInt(o.v)), When(o.t.size=64, "LL", "")),
        Inherited(o, i, is)),
   
   mul := (self, o, i, is) >> Cond( self.utype(o.t), Print("(", self.infix(o.args, ")&("), ")"), 
                                    # NOTE: HACK HACK 16x8i mul -> binary and 
                                    o.t = TVect(T_Int(8), 16), self( ApplyFunc(bin_and, o.args), i, is ),
                                    Inherited(o, i, is)),
   add := (self, o, i, is) >> When( self.utype(o.t), Print("(", self.infix(o.args, ")^("), ")"), Inherited(o, i, is)),
   sub := ~.add,

   rCyclicShift := (self, o, i, is) >> When( not self.utype(o.t), Inherited(o, i, is),
                      Cond( o.t.size = 16, self.printf("__rotl16($1, $2)", [o.args[1], o.args[2]]),
                            o.t.size = 32, self.printf("__rotl32($1, $2)", [o.args[1], o.args[2]]),
                            o.t.size = 64, self.printf("__rotl64($1, $2)", [o.args[1], o.args[2]]),
                            Error("unsupported size")
                      )),

   sklr_bcast_32x1i := (self, o, i, is) >> self.printf("$1", [tcast(o.t, arith_shr(tcast(T_Int(32), bin_shl(o.args[1], 31-o.args[2])), 31))]),
   sklr_bcast_64x1i := (self, o, i, is) >> self.printf("$1", [tcast(o.t, arith_shr(tcast(T_Int(64), bin_shl(o.args[1], 63-o.args[2])), 63))]),

   sklr_storeu_32x1i := _sklr_unaligned_store(32),
   sklr_storeu_64x1i := _sklr_unaligned_store(64),
));

SKLRBUnparser.bin_shl := (self, o, i, is) >> When( self.utype(o.t) or IsOrdT(o.t), 
         self.printf("(($1)<<($2))", [o.args[1], o.args[2]]), 
     # else
         Inherited(o, i, is));
SKLRBUnparser.bin_shr := (self, o, i, is) >> Cond( 
       self.utype(o.t) or IsUIntT(o.t), self.printf("(($1)>>($2))", [o.args[1], o.args[2]]), 
       self.utype(o.t) or IsIntT(o.t), 
           Print("((", self.declare(o.t, [], 0, 0), ")(((", self.declare(When(ObjId(o.t)=T_Int, T_UInt(o.t.params[1]), TUInt), [], 0, 0), ")(",
               self(o.args[1], i, is), "))>>(", self(o.args[2], i, is), ")))"), 
       Inherited(o, i, is));
SKLRBUnparser.arith_shr := (self, o, i, is) >> Cond( 
       IsIntT(o.t), self.printf("(($1)>>($2))", [o.args[1], o.args[2]]), 
       self.utype(o.t) or ObjId(o.t)=T_UInt, 
           Print("((", self.declare(o.t, [], 0, 0), ")(((", self.declare(T_Int(When(ObjId(o.t)=T_Int,o.t.params[1], o.t.size)), [], 0, 0), ")(",
               self(o.args[1], i, is), "))>>(", self(o.args[2], i, is), ")))"), 
       Inherited(o, i, is));
SKLRBUnparser.bin_and := (self, o, i, is) >> When( self.utype(o.t) or o.t in [T_Int(32), T_UInt(32), T_Int(64), T_UInt(64)], 
       Print("((", self.infix(o.args, ")&("), "))"), 
       Inherited(o, i, is));
SKLRBUnparser.bin_xor := (self, o, i, is) >> When( self.utype(o.t) or o.t in [T_Int(32), T_UInt(32), T_Int(64), T_UInt(64)], 
       Print("(", self.infix(o.args, ")^("), ")"), 
       Inherited(o, i, is));
SKLRBUnparser.bin_or := (self, o, i, is) >> When( self.utype(o.t) or o.t in [T_Int(32), T_UInt(32), T_Int(64), T_UInt(64)], 
       Print("(", self.infix(o.args, ")|("), ")"), 
       Inherited(o, i, is));
SKLRBUnparser.tcast := (self, o, i, is) >> When( not self.utype(o.t), Inherited(o, i, is), 
       Print("((", self.declare(o.args[1], [], i, is), ")(", self(o.args[2], i, is), "))"));

