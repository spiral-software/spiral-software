
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


NewRulesFor(DFT, rec(
    DFT_PRDFT := rec(
    switch := false,
    applicable := t -> Rows(t) >= 3 and t.getTags()=[],
    children := t -> [[ PRDFT1(Rows(t),t.params[2]) ]],
    apply := (t, ch, nonterms) -> let(n := Rows(t),
        bc := Mat([[1, E(4)], [1, -E(4)]]),
        DirectSum(
        Mat([[1,0]]),
        LIJ(n-1).transpose() * Cond(IsEvenInt(n),
            DirectSum(Tensor(I((n-2)/2), bc), Mat([[1,0]])),
            Tensor(I((n-1)/2), bc))) *
        ch[1])

    )
));

NewRulesFor(DFT2, rec(
    DFT2_PRDFT2 := rec(
    switch := false,
    applicable := t -> Rows(t) >= 3,
    children := t -> [[ PRDFT2(Rows(t), t.params[2]) ]],
    apply := (t, ch, nonterms) -> let(n := Rows(t), sgn := (-1)^t.params[2],
        bc := Mat([[1, E(4)], sgn*[1, -E(4)]]),
        DirectSum(
        Mat([[1,0]]),
        LIJ(n-1).transpose() * Cond(IsEvenInt(n),
            DirectSum(Tensor(I((n-2)/2), bc), Mat([[0,E(4)]])),
            Tensor(I((n-1)/2), bc))) *
        ch[1])

    )
));

NewRulesFor(DFT3, rec(
    DFT3_PRDFT3 := rec(
    switch := false,
    applicable := t -> Rows(t) >= 3,
    children := t -> [[ PRDFT3(Rows(t), t.params[2]) ]],
    apply := (t, ch, nonterms) -> let(n := Rows(t),
        bc := Mat([[1, E(4)], [1, -E(4)]]),
        LIJ(n).transpose() *
        Cond(IsEvenInt(n),
         Tensor(I(n/2), bc),
         DirectSum(Tensor(I((n-1)/2), bc), Mat([[1,0]]))) *
        ch[1])

    )
));


NewRulesFor(DFT4, rec(
    DFT4_PRDFT4 := rec(
    switch := false,
    applicable := t -> Rows(t) >= 3,
    children := t -> [[ PRDFT4(Rows(t),t.params[2]) ]],
    apply := (t, ch, nonterms) -> let(n := Rows(t), sgn := (-1)^t.params[2], jj := E(4)^t.params[2],
        bc := Mat([[1, E(4)], sgn*[1, -E(4)]]),
        LIJ(n).transpose() *
        Cond(IsEvenInt(n),
         Tensor(I(n/2), bc),
                 # last element could be real or imaginary based on
                 # rotation (t.params[2]) (but not both), so we scale both slots
         DirectSum(Tensor(I((n-1)/2), bc), Mat([[jj,jj]]))) *
        ch[1])

    )
));
