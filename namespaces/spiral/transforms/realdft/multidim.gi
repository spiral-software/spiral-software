
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(MD_PRDFT, TaggedNonTerminal, rec(
    abbrevs := [ (dims)     -> Checked(IsList(dims), ForAll(dims, IsPosIntSym), [dims, 1]), 
                 (dims,rot) -> Checked(IsList(dims), ForAll(dims, IsPosIntSym), IsIntSym(rot), [dims, rot]) 
    ], 
    
    terminate := self >> let(
        n := self.params[1], rot := self.params[2], 
        n0 := DropLast(n, 1),
        lst := Last(n), 
        res := RC(Tensor(MDDFT(n0), I(Rows(PRDFT(lst))/2))) * Tensor(I(Product(n0)), PRDFT(lst, rot)),
        When(self.transposed, res.transpose(), res)),

    toAMat := self >> self.terminate().toAMat(), 

    isReal := True,

    normalizedArithCost := self >> let(n := Product(self.params[1]), 
       IntDouble(2.5 * n * d_log(n) / d_log(2))),

    hashAs := self >> let(t:=ObjId(self)(self.params[1], 1).withTags(self.getTags()),
        When(self.transposed, t.transpose(), t)),

    dims := self >> let(
        n := self.params[1], 
        n0 := DropLast(n, 1),
        lst := Last(n), 
        d := [ Product(n0) * 2*(idiv(lst,2)+1), Product(n)], 
        When(self.transposed, Reversed(d), d)),

    omega := (N,k,r,c) -> E(N)^(k*r*c),
));


NewRulesFor(MD_PRDFT, rec(
    MD_PRDFT_Base := rec(
        applicable := t -> true,
        freedoms := t -> [],
        child := (t, fr) -> let(
            n := t.params[1], rot := t.params[2], 
            n0 := DropLast(n, 1),
            lst := Last(n), 
            RC(
                GT(MDDFT(n0), GTVec, GTVec, [Rows(PRDFT(lst))/2]).withTags(t.getTags())
            ) *
            GT(PRDFT(lst, rot), GTPar, GTPar, [Product(n0)]).withTags(t.getTags())
        ),
        apply := (self, t, C, Nonterms) >> C[1]
    )
));

# tst := vec -> let(n := Length(vec), dim := Sqrt(n), cols := 2*(Int(dim/2)+1),
#     out := TransposedMat(MatSPL(MD_PRDFT([dim,dim], 1)) * TransposedMat([vec]))[1],
#     outmat := List([0..dim-1], x->out{[1 + x*cols .. cols + x * cols]}),
#     PrintMat(outmat),
#     outmat);

# ctst := vec -> let(n := Length(vec), dim := Sqrt(n), 
#     out := TransposedMat(MatSPL(MDDFT([dim,dim], 1)) * TransposedMat([vec]))[1],
#     outmat := List([0..dim-1], x->out{[1 + x*dim .. dim + x * dim]}),
#     PrintMat((outmat)),
#     outmat);

# CX = X * RE
# X^-1 = RE * CX^-1
