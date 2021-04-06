
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(GT, rec(

# Why must we do parallelization and streaming in one big rule? Because they
# must both be done together. If done separately, we'll get two sets of outer
# perms (LxI)s, and we don't want that. In addition, the parallel breakdown
# rule has to tell a special mbuf rule how many iterations to multibuffer for,
# and a) this is messy, and b) this is trivial to handle in one single rule,
# and therefore best done within a single rule.


#F GT_ParStreamCore
#F Params: (p, u). p=# of procs, u=minimum packet size
    GT_ParStreamCore := rec(
        kmin          := 16, #128  # Minimum kernel size. DMA packet size = 4*k bytes for single precision real
        kmax          := 2048,  # Based on Max packet size for single precision (2048*4*2 = 16 Kbytes) NOTE: HARDCODED!

        requiredFirstTag := ParStream,

        # Variable naming scheme: either (In x Am) or (Am x In)

        applicable := (self, t) >> let(
            rank := Length(t.params[4]),     # Eg.: (Am x In)
            g    := t.params[2],
            s    := t.params[3],

            m    := Rows(t.params[1]),       # (splsize)
            n    := t.params[4][1],          # Available iterations (n)
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size

            b    := Maximum((m*n)/(self.kmax*p), 4), # Number of multibuffered iterations
            r    := n/(b*p),                 # Remaining iterations.

            # Conditions:
            rank = 1
            and PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
            and r >= 1       # If not, we have too few iterations for mbuf+par
            and When(g=GTPar and s=GTPar, r*m >= u,  r >= u)  # Respect specified packet size
        ),

        children := (self, t) >> let(
            g    := t.params[2],
            s    := t.params[3],

            m    := Rows(t.params[1]),
            n    := t.params[4][1],
            p    := t.firstTag().params[1],  # procs

            b    := Maximum((m*n)/(self.kmax*p), 4), # Number of multibuffered iterations
            r    := n/(b*p),                         # Remaining iterations.

            pkSizeG := When(g=GTPar, r*m, r),
            pkSizeS := When(s=GTPar, r*m, r),

            tags   := Drop(t.getTags(), 1), # Eat up tag

            When(r > 1,
              [ [ GT(t.params[1], g, s, [r]).withTags(tags),
                InfoNt(b, pkSizeG, pkSizeS) ] ],
              [ [ t.params[1].withTags(tags), InfoNt(b, pkSizeG, pkSizeS) ] ]
            )
        ),

        apply := (self, t, C, Nonterms) >> let(
            dft := Nonterms[1],
            g   := t.params[2],
            s   := t.params[3],
            n   := t.params[4][1],
            m   := Rows(t.params[1]),
            mc  := Cols(t.params[1]),
            p   := t.firstTag().params[1],
            b   := Nonterms[2].params[1],
            pkSizeG := Nonterms[2].params[2],
            pkSizeS := Nonterms[2].params[3],
            gg      := When(g.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),
            ss      := When(s.params[1]=[0,1], XChain([0,1,2]), XChain([1,2,0])),


            i      := var("spuid", TInt, p),
            j      := Ind(b),

            sp     := When(s=GTPar,
                        ScatDist(p, n*m/p, p, i),
                        ScatSend(ss.part(1, i, m, [p, 1]), n/p, p, i)),

            gp     := When(g=GTPar,
                        GathDist(p, n*m/p, p, i),
                        GathRecv(gg.part(1, i, mc, [p, 1]), n/p, p, i)),

            sm     := When(s=GTPar,
                        ScatMem(s.part(1, j, 1, [b]), pkSizeS),
                        ScatMem(s.part(1, j, m,  [b]), pkSizeS)),

            gm     := When(g=GTPar,
                        GathMem(g.part(1, j, 1, [b]), pkSizeG),
                        GathMem(g.part(1, j, mc, [b]), pkSizeG)),

            DistSum(p, i, p, 
                sp * MultiBufISum(j, b, sm, C[1], gm) * gp
            )

        ) #apply (let)
    ) #GT_ParStreamCore
));

NewRulesFor(GT, rec(
#F GT_StreamParChip
#F Params: (p, um, uc=u/p). p=# of procs, um=min pksize (mem), uc=min pksize (on-chip)
    GT_StreamParChip := rec(
        kmin          := 16, #128  # Minimum kernel size. DMA packet size = 4*k bytes for single precision real
        kmax          := 2048,  # Based on Max packet size for single precision (2048*4*2 = 16 Kbytes) NOTE: HARDCODED!


        requiredFirstTag := ParStream,

        # Variable naming scheme: either (In x Am) or (Am x In)
        applicable := (self, t) >> let(
            rank := Length(t.params[4]),     # Eg.: (Am x In)
            g    := t.params[2],
            s    := t.params[3],

            m    := Rows(t.params[1]),       # (splsize)
            n    := t.params[4][1],          # Available iterations (n)
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size (mem)

            b    := Maximum((m*n)/(self.kmax*p), 4), # Number of multibuffered iterations
            r    := n/b,                     # Remaining iterations.

            # Conditions:
            rank = 1
            and PatternMatch(t, [GT, @(1), @(2,XChain), @(3,XChain), ...], empty_cx())
            and r >= 1       # If not, we have too few iterations for mbuf+par
            and When(g=GTPar and s=GTPar, r*m >= u,  r >= u)  # Respect specified packet size
            #and not GT_ParStreamCore.applicable(t) # For now, prioritize the GT_ParStreamCore rule
        ),



        children := (self, t) >> let(

            g    := t.params[2],
            s    := t.params[3],

            m    := Rows(t.params[1]),
            n    := t.params[4][1],
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size (mem)
            uc   := u/p,                     # Minimum packet size (on-chip)

            b    := Maximum((m*n)/(self.kmax*p), 4), # Number of multibuffered iterations
            r    := n/b,                     # Remaining iterations.

            tags   := Drop(t.getTags(), 1), # Eat up tag

            # Add a parallel tag to child
            When(r > 1,
               [[ GT(t.params[1], g, s, [r]).setTags(Concatenation([ParCell(p, uc)], tags)) ]],
               [[ t.params[1].setTags(Concatenation([ParCell(p, uc)], tags)) ]]
            )
        ),

        apply := (self, t, C, Nonterms) >> let(
            dft     := Nonterms[1],
            g       := t.params[2],
            s       := t.params[3],

            m    := Rows(t.params[1]),
            n    := t.params[4][1],
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size (mem)

            b    := Maximum((m*n)/(self.kmax*p), 4), # Number of multibuffered iterations
            r    := n/b,                             # Remaining iterations.

            pkSizeS := When(s=GTPar, r*m, r),
            pkSizeG := When(g=GTPar, r*m, r),

            i      := Ind(b),
            sm     := When(s=GTPar,
                        ScatMem(s.part(1, i, 1, [b]), pkSizeS),
                        ScatMem(s.part(1, i, Rows(dft)/pkSizeS, [b/pkSizeS]), pkSizeS)),
            gm     := When(g=GTPar,
                        GathMem(g.part(1, i, 1, [b]), pkSizeG),
                        GathMem(g.part(1, i, Cols(dft)/pkSizeG, [b/pkSizeG]), pkSizeG)),

            MultiBufISum(i, b,
                sm,
                C[1],
                gm
            )


        ) #let/apply
    ), #GT rule

));


# NOTE: Works if the specified u is the max u. Else, fails.
# NOTE: Should convert this rule so that u is automatically picked.
NewRulesFor(GT, rec(
#F This is the ParStream version of the MBufCell_spec vector recursion rule
#F Automatically choses k,n, subject to a specified minimum
    GT_DFT_ParStreamCore_VecRecur := rec(
        kmin          := 16, #128  # Minimum kernel size. DMA packet size = 4*k bytes for single precision real
        kmax          := 2048,  # Maximum kernel size on a single SPE

        requiredFirstTag := ParStream,

        # Variable naming scheme: either (In x Am) or (Am x In)

        #applicable := false, # THIS RULE IS INCORRECT!

        applicable := (self, t) >> let(
            rank := Length(t.params[4]),     # Eg.: (Am x In)
            g    := t.params[2],
            s    := t.params[3],

            k    := t.params[4][1],          # Available iterations
            mn   := Rows(t.params[1]),       # (splsize)
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size

            mns  := Filtered(DivisorPairs(mn), 
                       i->let(m:=i[1], n:=i[2],
                          n >= u                # n is a packet size in second factor
                          and m >= 4*p          # multibuffer and parallelize around m iterations in second factor
                          and m*n >= self.kmin  # First factor kernel size is big enough
                          and m*n <= self.kmax  # First  factor fits on SPEs
                          and k*n <= self.kmax) # Second factor fits on SPEs
                    ),
#           Error("BP--applicable"),
            rank = 1
#           and not GT_ParStreamCore.applicable(t)  # Regular rule is not applicable
            and g=GTVec and s=GTPar             # Only consider (IxD)L
            and PatternMatch(t, [GT, DFT, XChain, XChain, @, @, @], empty_cx()) # Only consider GTs with DFT for a kernel
            and k >= 4*p                  # First loop is multibuffered and parallelized around k
            and k >= u                    # k is also the one of the packet sizes (2nd factor)
            and Length(mns) >= 1          # We can break the DFT based on all our constraints
        ),

        children := (self, t) >> let(
            g    := t.params[2],
            s    := t.params[3],
            k    := t.params[4][1],          # Available iterations (n)
            mn   := Rows(t.params[1]),       # (splsize)
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size

            tags := Drop(t.getTags(), 1),
            #oldtags := Drop(t.getTags(), 1),
            #tags := Concatenation([ParStream(p, u)], oldtags),

            mns  := Filtered(DivisorPairs(mn), 
                       i->let(m:=i[1], n:=i[2],
                          n >= u                # n is a packet size in second factor
                          and m >= 4*p          # multibuffer around m iterations in second factor
                          and m*n >= self.kmin  # First factor kernel size is big enough
                          and m*n <= self.kmax  # First  factor fits on SPEs
                          and k*n <= self.kmax) # Second factor fits on SPEs
                    ),

            Map2(mns, (m,n)-> [
              GT(DFT(m,1), GTVec, GTVec, [n]).withTags(tags),
              GT(DFT(n,1), GTVec, GTPar, [k]).withTags(tags),
              InfoNt(m, n)
              ])
        ),

        apply := (self, t, C, Nonterms) >> let(
            k    := t.params[4][1],
            m    := Nonterms[3].params[1],
            n    := Nonterms[3].params[2],
            p    := t.firstTag().params[1],  # procs

            j1    := Ind(k/p),
            j2    := Ind(m/p),
            pvar := var("spuid", TInt, p),
            pkSize1 := Rows(C[1]),

            #Error("BP----"),

            # WORKS:
            #MultiBufISum(j1, k,
            #      ScatMem(GTPar.part(1, j1, 1, [k]), pkSize1),
            #      C[1] * Diag(fPrecompute(Tw1(m*n, n, 1))),
            #      GathMem(GTPar.part(1, j1, 1, [k]), pkSize1)
            #)
            #*
            #MultiBufISum(j2, m,
            #     ScatMem(fTensor(fId(k), fBase(j2)), n),
            #     C[2],
            #     GathMem(fTensor(fId(n), fBase(j2)), k)
            #)

            perms := L(k*m, m),
            ixfs  := fTensor(fBase(pvar), fId(m*k/p)),

            permg := L(m*n, m),
            ixfg  := fTensor(fBase(pvar), fId(m*n/p)),

            DistSum(p, pvar, p,
                ScatDist(p, k*n*m/p, p, pvar) *
                MultiBufISum(j1, k/p,
                  ScatMem(GTPar.part(1, j1, 1, [k]), pkSize1),
                  C[1] * Diag(fPrecompute(Tw1(m*n, n, 1))),
                  GathMem(GTPar.part(1, j1, 1, [k]), pkSize1)
                ) * GathDist(p, k*n*m/p, p, pvar)
            ) *
            DistSum(p, pvar, p,
               ScatSend(fCompose(perms, ixfs), n, p, pvar) *

               MultiBufISum(j2, m/p, 
                 ScatMem(fTensor(fBase(j2), fId(k)), n),
                  C[2],
                 GathMem(fTensor(fBase(j2), fId(n)), k)
               ) *

               GathRecv(fCompose(permg, ixfg), k, p, pvar)
            )


        ) #let/apply
    )
));

NewRulesFor(GT, rec(
#F Vector recursion using streaming to entire chip
#F Automatically choses k,n, subject to a specified minimum pksize
    GT_DFT_StreamParChip_VecRecur := rec(
        kmin          := 128,  # Minimum kernel size. DMA packet size = 4*k bytes for single precision real
        kmax          := 2048,  # Maximum kernel size on a single SPE

        requiredFirstTag := ParStream,

        # Variable naming scheme: either (In x Am) or (Am x In)

        applicable := (self, t) >> let(
            rank := Length(t.params[4]),     # Eg.: (Am x In)
            g    := t.params[2],
            s    := t.params[3],

            k    := t.params[4][1],          # Available iterations
            mn   := Rows(t.params[1]),       # (splsize)
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size

            mns  := Filtered(DivisorPairs(mn), 
                       i->let(m:=i[1], n:=i[2],
                          n >= u                  # n is a packet size in second factor
                          and m >= 4              # multibuffer around m iterations in second factor
                          and m*n >= self.kmin    # First factor kernel size is big enough
                          and m*n <= self.kmax*p  # First  factor fits on SPEs
                          and k*n <= self.kmax*p) # Second factor fits on SPEs
                    ),
            rank = 1
#          and not GT_ParStreamCore.applicable(t)  # None of the 3 regular rules is applicable: GT_StreamParChip and GT_ParStreamCore should also not be applicable
#           and not GT_DFT_ParStreamCore_VecRecur.applicable(t)
           and not GT_StreamParChip.applicable(t)
            and g=GTVec and s=GTPar             # Only consider (IxD)L
            and PatternMatch(t, [GT, DFT, XChain, XChain, @, @, @], empty_cx()) # Only consider GTs with DFT for a kernel
            and k >= 4                    # First loop is multibuffered around k
            and k >= u                    # k is also the one of the packet sizes (2nd factor)
            and Length(mns) >= 1          # We can break the DFT based on all our constraints
        ),

        children := (self, t) >> let(
            g    := t.params[2],
            s    := t.params[3],
            k    := t.params[4][1],          # Available iterations (n)
            mn   := Rows(t.params[1]),       # (splsize)
            p    := t.firstTag().params[1],  # procs
            u    := t.firstTag().params[2],  # Specified minimum packet size (mem)
            uc   := u/p,                     # Minimum packet size (on-chip)

            oldtags := Drop(t.getTags(), 1),
            tags := Concatenation([ParCell(p, uc)], oldtags),

            mns  := Filtered(DivisorPairs(mn), 
                       i->let(m:=i[1], n:=i[2],
                          n >= u                  # n is a packet size in second factor
                          and m >= 4              # multibuffer around m iterations in second factor
                          and m*n >= self.kmin    # First factor kernel size is big enough
                          and m*n <= self.kmax*p  # First  factor fits on SPEs
                          and k*n <= self.kmax*p) # Second factor fits on SPEs
                    ),


            Map2(mns, (m,n)-> [
              GT(DFT(m,1), GTVec, GTVec, [n]).withTags(tags),
              GT(DFT(n,1), GTVec, GTPar, [k]).withTags(tags),
              InfoNt(m, n)
              ])
        ),

        apply := (self, t, C, Nonterms) >> let(
            k    := t.params[4][1],
            m    := Nonterms[3].params[1],
            n    := Nonterms[3].params[2],

            i    := Ind(k),
            j    := Ind(m),
            pkSize1 := Rows(C[1]),

            MultiBufISum(i, k,
                ScatMem(GTPar.part(1, i, 1, [k]), pkSize1),
                C[1] * Diag(fPrecompute(Tw1(m*n, n, 1))),
                GathMem(GTPar.part(1, i, 1, [k]), pkSize1)
            )
            * 
            MultiBufISum(j, m,
                ScatMem(fTensor(fId(k), fBase(j)), n),
                C[2],
                GathMem(fTensor(fId(n), fBase(j)), k)
            )

        ) #let/apply
    )
));




#NOTES
#-----------------------------------------------------------------------------
# Mon 11 Jan 2010 12:56:58 AM EST

# Outline for cleanup:

# We need to separate and disentangle all these rules: vecrecur should be
# independent of ParStreamCore or StreamParChip. ParStreamCore and
# StreamParChip should simply split the parallelism and let the lower level
# mbuf and par rules handle the rest. SigmaSPL rewrite rules should either
# /naturally/ be able to understand whether something is ParStreamCore or
# StreamParChip and act accordingly, or, if this is not possible naturally,
# this should be marked up somehow at the breakdown-rule level. Ugly ways for
# SigmaSPL rules to 'try' to figure out what's going on is as bad as Clippy.


