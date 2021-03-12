
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(rDFTSkew, realDFTPerm, fRealAccPease, _rDFT_u_func);

# A function to calculate "next" values of u for iteration j (used in rDFTSkew C-T rule)
_rDFT_u_func := function(u, j, k)
    if (u=0) then
        return V(fdiv(j, (2*k)));
    else 
        if (IsEvenInt(j)) then
            return fdiv((u + idiv(j,2)), k);
        else
            return fdiv((1 - u + idiv(j,2)), k);
        fi;
    fi;        
end;


# Recusively determine the value for u given k, q, and l, where
#    k = radix/2, 
#    q = log_k(m),
#    l = access function (or iterator).
_rDFT_u_rec := function(depth, u, k, q, l)   
   if (depth = q) then
       return V(u);
   else
       return _rDFT_u_func(_rDFT_u_rec(depth+1, u, k, q, l), idiv(imod(l, k^(depth+1)), V(k^(depth))), k);
   fi;
end;

# An Exp wrapper for _rDFT_u_rec.  This is used so Spiral does not try to evaluate the expression
# until Process_fPrecompute is called.
Class(_rDFT_u_rec_exp, Exp, rec(
    ev := self >> let(
        depth := self.args[1].ev(),
        u := self.args[2].ev(),
        k := self.args[3].ev(),
        q := self.args[4].ev(),
        l := self.args[5].ev(),
        _rDFT_u_rec(depth, u, k, q, l).ev()
    ),
    
    computeType := self >> TReal
));

# A function wrapper for _rDFT_u_rec_exp.  The rewriting rule uses this function.  Then, it is evaluated to the
# Exp above.  Then, that will evetually be evaluated into the final value.
Class(_rDFT_u_rec_wrapper, FuncClass, rec(
    def := (d, u, k, q, l) -> rec(N := 1, n := 1),
    lambda := self >> let(i := Ind(1), Lambda(i, _rDFT_u_rec_exp(self.params[1], self.params[2], self.params[3], self.params[4], self.params[5]))),
    t := TReal,
    eval := self >> When(self.domain() = 1 and self.rank() = 0, self.at(0).eval(), self),
    ev := self >> Checked(self.domain() = 1, self.at(0).ev()),
));


# An Exp wrapper for _rDFT_u_func.  This is used in the C-T rule, so the "next" values of u can be
# calcualted (but not evaluated until they can be.)
Class(_rDFT_u_r_exp, Exp, rec(
   ev := self >> let(
       u := self.args[1].ev(),
       j := self.args[2].ev(),
       k := self.args[3].ev(),
       _rDFT_u_func(u, j, k).ev()
   ),
   
   computeType := self >> TReal
));



# rDFTSkew(n,u).  This is the transform our streaming and Pease rules apply to.
Class(rDFTSkew, TaggedNonTerminal, rec(
    abbrevs := [(n,u) -> Checked(IsPosInt(n), [n, u])],
    dims := self >> let(n := self.params[1], [n,n]),
    isReal := self >> true,

    terminate := self >> let(
        n := self.params[1],
        k := 2,
        m := n/(2*k),
        a := Cond(IsValue(self.params[2]), self.params[2], V(self.params[2])),
        Cond(n=4,
            
            Cond(a=0,
                Tensor(F(2), I(2)),
                Diag(1,1,1,-1) * Tensor(F(2), I(2)) * 
                   DirectSum(I(2), Mat([[cospi(a.ev()), -1*sinpi(a.ev())], [sinpi(a.ev()), cospi(a.ev())]])) *
                   L(4,2)
            ),
            
            RC(Cond(a=0, Kp(k*m, m), K(k*m, m))) *
            DirectSum(
                rDFTSkew(2*m, _rDFT_u_func(a, 0, k)).terminate(), 
                rDFTSkew(2*m, _rDFT_u_func(a, 1, k)).terminate()
            ) *
            Tensor(rDFTSkew(2*k, a), I(m))
        )
    ),
)); 



NewRulesFor(PkRDFT1, rec(

    # PkRDFT1(n,0) --> DirectSum(F(2), I(n-2)) * rDFTSkew(n,0)
    # We simplify by writing the direct sum already in TTensorInd form.

    PkRDFT1_rDFTSkew := rec(
        forTransposition := false,
        applicable := (self, nt) >> nt.params[2] = 1,  #! temporary hack
        children := (self, t) >> let(
            n := t.params[1],
            i := Ind(n/2),
            tags := t.getTags(),
            
            [[ TCompose([TTensorInd(COND(eq(i,0), F(2), I(2)), i, APar, APar), rDFTSkew(n,V(0))]).withTags(tags) ]]
        ),

        apply := (nt, c, cnt) -> c[1]        
    ),

    PkRDFT1_DFT := rec(
        forTransposition := false,
        applicable := (self, nt) >> nt.params[2] = 1, #! temporary
        children := (self, t) >> 
            [[TCompose([TConjEven(t.params[1]), TRC(DFT(t.params[1]/2))]).withTags(t.getTags())]],

        apply := (nt, c, cnt) -> c[1]
    )
));


NewRulesFor(rDFTSkew, rec(
    # opts.breakdownRules := rec(rDFTSkew := [CopyFields(rDFT_Skew_CT, rec(radix:=4))])
    #    


    # rDFTSkew(2*k*m,0) --> Perm * (I_k x rDFTSkew(2*m, f(u,l))) * (rDFTSkew(2*k, u) x I_m)
    rDFT_Skew_CT := rec(
        forTransposition := false,
        applicable := (self, nt) >> nt.params[1] > 4 and not nt.isTag(1, AStream),
        children := (self, t) >> let(
            n := t.params[1],
            m := 2,
            k := n/(2*m),

            u := t.params[2],
            j := Ind(k),

            [[ rDFTSkew(2*m, _rDFT_u_r_exp(u, j, k)), rDFTSkew(2*k, u), InfoNt(j) ]]
         ),

        apply := (t,c,nt) -> let(
            n := t.params[1],
            m := 2,
            k := n/(2*m),
            u := t.params[2],
            j := nt[3].params[1],
            D := Dat1d(TReal, 1),

            Data(D, fPrecompute(FList(TReal, [cospi(u)])),
                COND(eq(nth(D,0),1), Tensor(Kp(k*m,m), I(2)), Tensor(K(k*m,m), I(2))) *
                IDirSum(j, k, c[1]) * 
                Tensor(c[2], I(m))
            )
            
        )
        
    ),


    # rDFTSkew(4,u) base rule.
    rDFT_Skew_Base4 := rec(
        forTransposition := false,
        applicable := (self, nt) >> nt.params[1] = 4,
        apply := (t, c, nt) -> let(
            u := t.params[2],
            DC := Dat1d(TReal, 1),
            DS := Dat1d(TReal, 1),

            Data(DC, fPrecompute(FList(TReal, [cospi(u)])),
                Data(DS, fPrecompute(FList(TReal, [sinpi(u)])),
                    COND(eq(nth(DC,0),1), I(4), Diag(1,1,1,-1)) *
                    Tensor(F(2), I(2)) *
                    DirectSum(I(2), Mat([[nth(DC,0), -1*nth(DS,0)], [nth(DS,0), nth(DC,0)]])) *
                    COND(eq(nth(DC,0),1), I(4), L(4,2))
                )
                
            )
        )
    ),


    # Streaming rDFTSkew(n,u) rule.
    rDFT_Skew_Stream := rec(
        forTransposition := false,
        
        radix := 4,

        applicable := (self, nt) >> let(
            k := self.radix/2,
            m := nt.params[1]/(2*k),
            logM := LogInt(m,k),
            IsInt(nt.params[1]/(2*k)) and nt.isTag(1, AStream) and nt.getTags()[1].bs >= self.radix and
            (k ^ logM = m)
        ),

        freedoms := t -> [[ ]],

        children := (self, nt) >> let(
            P    := nt.params,
            k    := self.radix/2,
            m    := P[1]/(k*2),
            tags := nt.getTags(),
            q    := LogInt(m,k),
            u    := nt.params[2], 
            
            itvars := List([0..q], i->Ind(m)),
            
            [[ TCompose(Concatenation(
                 [ TPrm(realDFTPerm(2*k*m, k, u)) ],
                 [ TTensorInd(rDFTSkew(2*k, _rDFT_u_rec_wrapper(0, u, k, q, itvars[1])), itvars[1], APar, APar) ],
                 List([1..q], i-> TCompose([
                     Cond(i=1, 
                         TTensorI(TPrm(L(2*k^(i+1), 2*k)), m/(k^i), APar, APar),
                         TTensorI(TPrm(Compose(Tensor(I(k), L(2*(k^i), k^(i-1))), L(2*k^(i+1), 2*k))), m/(k^i), APar, APar)
                     ),
                     TTensorInd(rDFTSkew(2*k, _rDFT_u_rec_wrapper(0, u, k, q-i, bin_shr(itvars[i+1], Log2Int(k^i)))),
                         itvars[i+1], APar, APar)

                 ])),
                 [ TPrm(L(2*k^(q+1), m)) ]
             )).withTags(tags)]]
        ),

        apply := (nt, c, cnt) -> c[1]
                      
    ),

    # Non-uniform radix streaming breakdown rule for rDFTSkew(n,u)
    # This does not work correctly when u<>0 and k > 2.   I still need to debug this.
    # NOTE
    rDFT_Skew_Stream_Mult_Radices := rec(
        forTransposition := false,
        
        applicable := (self, nt) >> let(
            n  := nt.params[1],
            rp := paradigms.stream.rDFT_Skew_Stream.radix,
            kp := rp/2,
            mp := kp ^ LogInt(n/(2*kp), kp),
            m  := kp * mp,
            k  := n/(2*m),
            u := nt.params[2],   

            IsInt(nt.params[1]/(2*k)) and nt.isTag(1, AStream) and
                not paradigms.stream.rDFT_Skew_Stream.applicable(nt) and (u=0 or k<=2)
        ),
    
        freedoms := t -> [[ ]],
        
        children := (self, nt) >> let(
            n  := nt.params[1],
            rp := paradigms.stream.rDFT_Skew_Stream.radix,
            kp := rp/2,
            mp := kp ^ LogInt(n/(2*kp), kp),
            m  := kp * mp,
            k  := n/(2*m),
            logM := LogInt(m,k),

            tags := nt.getTags(),
            u := nt.params[2],
            i := Ind(k), 
            
            [[ TCompose([ TPrm(Tensor(COND(eq(u,0), Kp(k*m, m), K(k*m,m)), I(2))),
                          TTensorInd(rDFTSkew(2*m, _rDFT_u_r_exp(u, i, k)), i, APar, APar),
                          TPrm(L(2*k*m, 2*k)),
                          TTensorI(rDFTSkew(2*k, u), m, APar, APar),
                          TPrm(L(2*k*m, m)) ]).withTags(tags) ]]

        ),

        apply := (nt, c, cnt) -> c[1]

    ),

    # Pease rDFTSkew(n,0) rule
    rDFT_Skew_Pease := rec(
        forTransposition := false,

        unroll_its := 1,
        radix      := 4,
        
        # For some k between minRadix/2 and maxRadix/2, P[1]=2*k*m, where m=k^p where p is an integer
        applicable := (self, nt) >> let(
            k := self.radix/2,
            m := nt.params[1]/(2*k),
            logM := LogInt(m, k),

            nt.params[2] = 0 and   # If we were to support u <> 0, we would need to change the
                                   # the way we store constants.  The trick we currently use
                                   # to go from O(n log n) to O(n) would not work without modification
                                   # (it may in fact not work at all).
            IsInt(nt.params[1]/(2*k)) and
            (k ^ logM = m) and
            IsInt((logM+1)/self.unroll_its) and
            (logM+1)/self.unroll_its > 1 and
            nt.isTag(1, AStream)
        ),
                  
        freedoms := t -> [[ ]],

	    children := (self, nt) >> let(
            P     := nt.params,
            k     := self.radix/2,
            m     := P[1]/(k*2),
            tags  := nt.getTags(),
            q     := LogInt(m,k),
            it_i  := Ind((q+1) / self.unroll_its),
            it_l  := Ind(m),
            u     := 0,

            it_ur := i >> self.unroll_its * it_i + i,
            f_ur  := i >> fComputeOnline(fRealAccPease(k, m, it_ur(i), it_l)),
            u_ur  := i >> _rDFT_u_rec_wrapper(0, u, k, q, f_ur(i)),

            # This provides stage i of the datapath.
            stage      := i >> TCompose([TTensorInd(rDFTSkew(2*k, u_ur(i)), it_l, APar, APar), TL(2*k*m, 2*m, 1, 1)]),

            # This gives a full stage (multiple stage(i) when we unroll).
            full_stage := List([1..self.unroll_its], t -> stage(t-1)),

            [[ TCompose([
                  realDFTPerm(2*k*m, k, u),
                  TICompose(it_i, it_i.range, TCompose(full_stage)),
                  TL(2*k*m, k*m, 1, 1)
               ]).withTags(tags)]]
        ),

        apply := (nt, c, cnt) -> c[1]
    ),
));


# Pease rDFTSkew access function for rDFTSkew(2*k*m, 0) where i is the stage and l is the position in stage.
#     fRealAccPease(k, m, i, l)
Class(fRealAccPease, FuncClass, rec(
   def := (k, m, i, l) -> rec(N:=m, n:=1),

   domain := self >> 1,
   range  := self >> self.params[2],
   
   lambda := self >> let(k := self.params[1],
                         m := self.params[2],
                         i := self.params[3],
			             l := self.params[4],
			             j := Ind(1),
                         logk := LogInt(k,2),

       Lambda(j, bin_and(l, bin_shr(m-1, logk*i)))),

   computeType := self >> TInt,
   t := TInt,
   eval := self >> When(self.domain() = 1 and self.rank() = 0, self.at(0).eval(), self),
   ev := self >> Checked(self.domain() = 1, self.at(0).ev())
    
));




# The output permutation in Pease or streaming rDFTSkew algorithms
# realDFTPerm(n, k), where n = 2*k*m
Class(realDFTPerm, TaggedNonTerminal, rec(

    abbrevs := [(n, k, u)      -> [n, k, u]],

    dims := self >> [ self.params[1], self.params[1] ],
    domain := self >> self.params[1],
    range := self >> self.params[1],

    terminate := self >> let(
	    n := self.params[1],
        k := self.params[2],
        u := Cond(IsExp(self.params[3]), self.params[3].ev(), self.params[3]),
        m := n/(2*k),
        r := LogInt(m,k),

        res := Cond(n < 8, I(n),
            Cond(u=0,
	            Tensor(
                    Compose(List([0..r-1], l->
                        DirectSum(Kp(k*m/(k^l), k*m/(k^(l+1))),
                            Tensor(I(k^l-1), K(k*m/(k^l), k*m/(k^(l+1))))
                        )
                    )),
                    I(2)
                ),
	            Tensor(
                    Compose(List([0..r-1], l->
                        Tensor(I(k^l), K(k*m/(k^l), k*m/(k^(l+1))))
                    )),
                    I(2)
                )
            )
        ),

        Cond(self.transposed, res.transpose(), res)
    ),

    isReal := self >> true,

    print := meth(self, indent, indentStep)
        local lparams, mparams;
        if not IsBound(self.params) then Print(self.name); return; fi;
        Print(self.name, "(");
        if IsList(self.params) then
            lparams := Filtered(self.params, i->not (IsList(i) and i=[]));
            mparams := Filtered(lparams, i->not (IsBool(i) and not i));
            DoForAllButLast(mparams, x -> Print(x, ", "));
            Print(Last(mparams));
        else
            Print(self.params);
        fi;
    Print(")", When(self.transposed, ".transpose()", ""));
    end,

));


NewRulesFor(realDFTPerm, rec(

    # Rule to place realDFTPerm into a StreamPermGen wrapper.
    # realDFTPerm --> StreamPermGen(realDFTPerm)

   
    realDFTPerm_Stream := rec(
        isApplicable := (self, P) >> Checked(IsList(P[2]) and IsBound(P[2][1].bs),
	                                      IsInt((2^P[1])/P[2][1].bs)),

      children := (self, nt) >> [[]],
      apply := (nt, c, cnt) -> StreamPerm([realDFTPerm(nt.params[1], nt.params[2], nt.params[3])], 1, nt.getTags()[1].bs, 0),
   )
));

Class(TConjEvenStreamBB, NonTerminal, rec(
    abbrevs := [(n,j) -> Checked(IsPosInt(n), [n,j])],
    dims    := self >> [4,4],
    isReal  := self >> true,

    terminate := self >> let(
        j  := self.params[2].ev(),
        n  := self.params[1],

        c  := cospi(2*j/n)/2,
        s  := sinpi(2*j/n)/2,
        
        Cond(j=0,
            Mat([[1,1,0,0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            
            Diag(1,1,1,-1) *
            Tensor(F(2), I(2)) *
            DirectSum(Diag(1/2, 1/2), Mat([[c, -1*s], [s, c]])) *
            L(4,2) * 
            Mat([[1,0,1,0],[0,1,0,1],[0,1,0,-1],[-1,0,1,0]])
            

        )
    )
));


NewRulesFor(TConjEven, rec(
    TConjEven_stream := rec(
        switch := false,
        applicable := (self, t) >> 2^Log2Int(t.params[1]) = t.params[1] and
                                       t.isTag(1, AStream) and t.getTags()[1].bs >= 4,
               
        children := (self, nt) >> let(
            n    := nt.params[1],
            tags := nt.getTags(),
            i    := Ind(n/4),
	    rot  := nt.params[2],

            [[ TCompose([
#                    TPrm(Tensor(DirectSum(I(n/4+1), J(n/4-1)) * L(n/2,2), I(2))),
#                    TPrm(Tensor(Kp(n/2, 2), I(2))),
                    TRC(TPrm(Kp(n/2, 2))),
		    # YSV: NOTE -- handle <rot> below
                    TTensorInd(TConjEvenStreamBB(n, i), i, APar, APar),
#                    TPrm(Tensor(L(n/2, n/4) * DirectSum(I(n/4+1), J(n/4-1)), I(2)))
#                    TPrm(Tensor(Kp(n/2, 2).transpose(), I(2)))
                    TRC(TPrm(Kp(n/2, 2).transpose()))
               ]).withTags(tags)]]
        ),

        apply := (nt, c, cnt) -> c[1]
                          
    )
));

NewRulesFor(TConjEvenStreamBB, rec(
    TConjEvenStreamBB_base := rec(
        switch := false,
        forTransposition := false,
        applicable := (self, nt) >> IsPosInt(nt.params[1]),

        # new: better alg.  store n/2 words, perform 4 multiplications
        apply := (t, c, nt) -> let(
            j := t.params[2],
            n := t.params[1],
            cs := Dat1d(TReal, 1),
            sn := Dat1d(TReal, 1),
            
            Data(cs, fPrecompute(FList(TReal, [fdiv(cospi(fdiv(2*j,n)),2)])),
                Data(sn, fPrecompute(FList(TReal, [fdiv(sinpi(fdiv(2*j,n)),2)])),
                    COND(eq(j,0),
                        Mat([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                        Diag(1,1,1,-1) *
                        Tensor(F(2), I(2)) *
                        Mat([[1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, nth(cs,0), -1*nth(sn,0)], [0, 0, nth(sn,0), nth(cs,0)]]) *
                        L(4,2) *
                        Mat([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [-1, 0, 1, 0]])
                    )                               
                )
            )
        )

        

        # Store n/2 words
#         apply := (t, c, nt) -> let(
#             j := t.params[2],
#             n := t.params[1],
#             d0 := Dat1d(TReal, 1),
#             d1 := Dat1d(TReal, 1),
            
#             Data(d0, fPrecompute(FList(TReal, [fdiv(sinpi(fdiv(2*j,n)),2)])),
#                 Data(d1, fPrecompute(FList(TReal, [fdiv((cospi(fdiv(2*j,n))),2)])),
#                     COND(eq(j,0), 
#                         Mat([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
#                         let(
#                             b0 := 0.5+nth(d0,0),
#                             b2 := 0.5-nth(d0,0),
#                             Mat([[b0, nth(d1,0), b2, nth(d1,0)],
#                                  [-1*nth(d1, 0), b0, nth(d1,0), -1*b2],
#                                  [b2, -1*nth(d1,0), b0, -1*nth(d1,0)],
#                                  [-1*nth(d1, 0), -1*b2, nth(d1,0), b0]])
#                         )
#                     )
#             ))
#         )

        # Store 3n/4 words
#         apply := (t, c, nt) -> let(
#             j := t.params[2],
#             n := t.params[1],
#             d0 := Dat1d(TReal, 1),
#             d1 := Dat1d(TReal, 1),
#             d2 := Dat1d(TReal, 1),

#             Data(d0, fPrecompute(FList(TReal, [fdiv((1+sinpi(fdiv(2*j,n))),2)])),
#                 Data(d1, fPrecompute(FList(TReal, [fdiv((cospi(fdiv(2*j,n))),2)])),
#                     Data(d2, fPrecompute(FList(TReal, [fdiv((1-sinpi(fdiv(2*j,n))),2)])),
#                         COND(eq(j,0), 
#                             Mat([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
#                             Mat([[nth(d0, 0), nth(d1,0), nth(d2,0), nth(d1,0)],
#                                  [-1*nth(d1, 0), nth(d0,0), nth(d1,0), -1*nth(d2,0)],
#                                  [nth(d2, 0), -1*nth(d1,0), nth(d0,0), -1*nth(d1,0)],
#                                  [-1*nth(d1, 0), -1*nth(d2,0), nth(d1,0), nth(d0,0)]])
#                         )
#             )))
#         )


     )
));
