
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(Ext, Sym, rec(
    abbrevs := [ 
	(n, ext) -> [n, ext, ext, 1, 1],
	(n, l, r) -> [n, l, r, 1, 1] ],

    def := function(n, l, r, lscale, rscale)
         local stack;
         Checked(IsPosInt(n), true);
         stack := [];

         if (IsInt(l) and l>0) then 
           Add(stack,O(l,n));
         elif (IsRec(l) and not l.n=0 ) then 
           l := When(IsSPL(l), l, l.left(n));
           Add(stack, When(lscale<>1, lscale*Gath(l), Gath(l)));
         fi;

         Add(stack, I(n));

         if (IsInt(r) and r>0) then 
           Add(stack,O(r,n));
         elif (IsRec(r) and not r.n=0 ) then
           r := When(IsSPL(r), r, r.right(n));
           Add(stack, When(rscale<>1, rscale*Gath(r), Gath(r)));
         fi;

	    return VStack(stack);
    end
));


Class(DownSample, Sym, rec(
    abbrevs := [ (n, fact, offset) -> 
        Checked(IsPosInt(n), IsPosInt(fact), IsPosInt0(offset), [n, fact, offset]) ],

    def := (n, fact, offset) -> let(m := Int((n-offset-1) / fact + 1),
         Gath(H(n, m, offset, fact)))
));

Class(UpSample, Sym, rec(
    abbrevs := [ (n, fact, offset) -> 
        Checked(IsPosInt(n), IsPosInt(fact), IsPosInt0(offset), [n, fact, offset]) ],

    def := (n, fact, offset) -> let(m := Int((n-offset-1) / fact + 1),
         Scat(H(n, m, offset, fact)))
));

