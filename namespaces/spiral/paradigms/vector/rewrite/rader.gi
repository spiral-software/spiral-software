
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(TerminateRaderMid, RuleSet);
RewriteRules(TerminateRaderMid, rec(
    terminateRaderMid := Rule(@(1, VecRaderMid), e-> e.term())
));

Class(StretchRaderMid, RuleSet);
RewriteRules(StretchRaderMid, rec(

    stretchCT_A := ARule(Compose,
        [ [ DelayedDirectSum, @(2, VScat_sv, e->Cols(e)=getV(e)), [ @(3, Compose), ..., @(4, IxVGath_pc) ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, @(7, VGath_sv, e->Rows(e)=getV(e)), [ @(8, Compose),
                @(9, IxVScat_pc, s-> let(g:=@(4).val, s.k=g.k and s.N=g.N and s.ofs=g.ofs and s.v=g.v )), ... ] ]
        ],
        e -> [ DelayedDirectSum(@(2).val, Compose(DropLast(@(3).val.children(), 1))),
               let(g:=@(4).val , @(5).val.stretchCT(g.N)),
               DelayedDirectSum(@(7).val, Compose(Drop(@(8).val.children(), 1))) ]),

    stretchCT := ARule(Compose,
        [ [ DelayedDirectSum, [ @(1, Compose), ..., @(2, VScat_sv, e->Cols(e)=getV(e)) ], [ @(3, Compose), ..., @(4, IxVGath_pc) ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, [ @(6, Compose), @(7, VGath_sv, e->Rows(e)=getV(e)), ... ], [ @(8, Compose),
                @(9, IxVScat_pc, s-> let(g:=@(4).val, s.k=g.k and s.N=g.N and s.ofs=g.ofs and s.v=g.v )), ... ] ]
        ],
        e -> [ DelayedDirectSum(@(1).val, Compose(DropLast(@(3).val.children(), 1))),
               let(g:=@(4).val , @(5).val.stretchCT(g.N)),
               DelayedDirectSum(@(6).val, Compose(Drop(@(8).val.children(), 1))) ]),

#--
    stretchCT2_A := ARule(Compose,
        [ [ DelayedDirectSum, @(2, VScat_sv, e->Cols(e)=getV(e)), [ @(3, Compose), ..., @(4, IxVGath_pc), [VScat_sv, fId] ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, @(7, VGath_sv, e->Rows(e)=getV(e)), [ @(8, Compose),
                [VGath_sv, fId], @(9, IxVScat_pc, s-> let(g:=@(4).val, s.k=g.k and s.N=g.N and s.ofs=g.ofs and s.v=g.v )), ... ] ]
        ],
        e -> [ DelayedDirectSum(@(2).val, Compose(DropLast(@(3).val.children(), 2))),
               let(g:=@(4).val , @(5).val.stretchCT(g.N)),
               DelayedDirectSum(@(7).val, Compose(Drop(@(8).val.children(), 2))) ]),

    stretchCT2 := ARule(Compose,
        [ [ DelayedDirectSum, [ @(1, Compose), ..., @(2, VScat_sv, e->Cols(e)=getV(e)) ], [ @(3, Compose), ..., @(4, IxVGath_pc), [VScat_sv, fId] ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, [ @(6, Compose), @(7, VGath_sv, e->Rows(e)=getV(e)), ... ], [ @(8, Compose),
                [VGath_sv, fId], @(9, IxVScat_pc, s-> let(g:=@(4).val, s.k=g.k and s.N=g.N and s.ofs=g.ofs and s.v=g.v )), ... ] ]
        ],
        e -> [ DelayedDirectSum(@(1).val, Compose(DropLast(@(3).val.children(), 2))),
               let(g:=@(4).val , @(5).val.stretchCT(g.N)),
               DelayedDirectSum(@(6).val, Compose(Drop(@(8).val.children(), 2))) ]),

#--
    stretchPFA_A := ARule(Compose,
        [ [ DelayedDirectSum, @(2, VScat_sv, e->Cols(e)=getV(e)), [ @(3, Compose), ..., @(4, VStretchGath), [VScat_sv, fId] ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, @(7, VGath_sv, e->Rows(e)=getV(e)), [ @(8, Compose),
                [VGath_sv, fId], @(9, VStretchScat, s-> let(g:=@(4).val, s.part=g.part and s.func.range()=g.func.range() and s.v=g.v )),  ... ] ]
        ],
        e -> [ DelayedDirectSum(@(2).val, Compose(DropLast(@(3).val.children(), 2))),
               let(g:=@(4).val , @(5).val.stretchPFA(g.part, g.func.range(), @(4).val.func)),
               DelayedDirectSum(@(7).val, Compose(Drop(@(8).val.children(), 2))) ]),

    stretchPFA := ARule(Compose,
        [ [ DelayedDirectSum, [ @(1, Compose), ..., @(2, VScat_sv, e->Cols(e)=getV(e)) ], [ @(3, Compose), ..., @(4, VStretchGath), [VScat_sv, fId] ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, [ @(6, Compose), @(7, VGath_sv, e->Rows(e)=getV(e)), ... ], [ @(8, Compose),
                [VGath_sv, fId], @(9, VStretchScat, s-> let(g:=@(4).val, s.part=g.part and s.func.range()=g.func.range() and s.v=g.v )),  ... ] ]
        ],
        e -> [ DelayedDirectSum(@(1).val, Compose(DropLast(@(3).val.children(), 2))),
               let(g:=@(4).val , @(5).val.stretchPFA(g.part, g.func.range(), @(4).val.func)),
               DelayedDirectSum(@(6).val, Compose(Drop(@(8).val.children(), 2))) ]),
#--
    stretchPFA2_A := ARule(Compose,
        [ [ DelayedDirectSum, [ @(1, Compose), ..., @(2, VScat_sv, e->Cols(e)=getV(e)) ], [ @(3, Compose), ..., @(4, VStretchGath), ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, [ @(6, Compose), @(7, VGath_sv, e->Rows(e)=getV(e)), ... ], [ @(8, Compose),
                @(9, VStretchScat, s-> let(g:=@(4).val, s.part=g.part and s.func.range()=g.func.range() and s.v=g.v )),  ... ] ]
        ],
        e -> [ DelayedDirectSum(@(1).val, Compose(DropLast(@(3).val.children(), 1))),
               let(g:=@(4).val , @(5).val.stretchPFA(g.part, g.func.range(), @(4).val.func)),
               DelayedDirectSum(@(6).val, Compose(Drop(@(8).val.children(), 1))) ]),

    stretchPFA2 := ARule(Compose,
        [ [ DelayedDirectSum, @(2, VScat_sv, e->Cols(e)=getV(e)), [ @(3, Compose), ..., @(4, VStretchGath), ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, @(7, VGath_sv, e->Rows(e)=getV(e)), [ @(8, Compose),
                @(9, VStretchScat, s-> let(g:=@(4).val, s.part=g.part and s.func.range()=g.func.range() and s.v=g.v )),  ... ] ]
        ],
        e -> [ DelayedDirectSum(@(2).val, Compose(DropLast(@(3).val.children(), 1))),
               let(g:=@(4).val , @(5).val.stretchPFA(g.part, g.func.range(), @(4).val.func)),
               DelayedDirectSum(@(7).val, Compose(Drop(@(8).val.children(), 1))) ]),
#--
    stretchPFA3_A := ARule(Compose,
        [ [ DelayedDirectSum, [ @(1, Compose), ..., @(2, VScat_sv, e->Cols(e)=getV(e)) ], [ @(3, Compose), ..., @(4, DelayedPrm) ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, [ @(6, Compose), @(7, VGath_sv, e->Rows(e)=getV(e)), ... ], [ @(8, Compose),
                @(9, DelayedPrm, s-> let(g:=@(4).val, 1=1)),  ... ] ]
        ],
        e -> [ DelayedDirectSum(@(1).val, Compose(DropLast(@(3).val.children(), 1))),
               let(g:=@(4).val , @(5).val.stretchPFA(1, Cols(g), @(4).val.func)),
               DelayedDirectSum(@(6).val, Compose(Drop(@(8).val.children(), 1))) ]),

    stretchPFA3 := ARule(Compose,
        [ [ DelayedDirectSum, @(2, VScat_sv, e->Cols(e)=getV(e)), [ @(3, Compose), ..., @(4, DelayedPrm) ] ],
          @(5, VecRaderMid),
          [ DelayedDirectSum, @(7, VGath_sv, e->Rows(e)=getV(e)), [ @(8, Compose),
                @(9, DelayedPrm, s-> let(g:=@(4).val, 1=1)),  ... ] ]
        ],
        e -> [ DelayedDirectSum(@(2).val, Compose(DropLast(@(3).val.children(), 1))),
               let(g:=@(4).val , @(5).val.stretchPFA(1, Cols(g), @(4).val.func)),
               DelayedDirectSum(@(7).val, Compose(Drop(@(8).val.children(), 1))) ])
));
