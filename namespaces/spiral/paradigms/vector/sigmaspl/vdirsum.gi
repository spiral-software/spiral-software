
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(VDirectSum, DirectSum, rec(
    abbrevs := [(L, v) -> [L, v]],
    new := meth(self, L, v)
        if ForAll(L, IsIdentitySPL)
            then return I(Sum(L, Rows));
        else
            return SPL(WithBases( self,
            rec( _children := L,
            v := v,
            dimensions := [ Sum(L, t -> t.dimensions[1]),
                            Sum(L, t -> t.dimensions[2]) ] )));
        fi;
    end,
#    sums := meth(self)
#        nblocks, c, cols, rows, bkcols, bkrows, gcd;
#        nblocks  :=  self.numChildren();
#        c := self.children();
#        cols  :=  EvalScalar(Cols(self));
#        rows := EvalScalar(Rows(self));
#        bkcols := List(c, i->EvalScalar(Cols(i)));
#        bkrows := List(c, i->EvalScalar(Rows(i)));
#        gcd := self.v;
#        bkrows_by_gcd := List(bkrows, i->i/gcd);
#        psr_by_gcd := List(DropLast(_partSum(bkrows), 1), i ->idiv(i,gcd));
#    fi;
#    psc := DropLast(Concat([0], List(Drop(_partSum(bkcols), 1), i->i-overlap)), 1);
#    return SUM( Map([1..nblocks],
#        i -> Compose(
#                    Scat(fTensor(fAdd(rows/gcd, bkrows_by_gcd[i], Cond(i=1, 0, i=2, bkrows_by_gcd[i-1], psr_by_gcd[i])), fId(gcd))),
#                    self.child(i).sums(),
#                    Gath(fTensor(fAdd(cols/gcd, bkcols[i]/gcd, psc[i]/gcd), fId(gcd))))
#            ));
));

