
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(ImageData, ImageVar, ImageUnk);

Class(ImageUnk, SumsBase, BaseContainer, rec(
    new := (self, dims) >> Checked(IsList(dims) and Length(dims) = 2 and ForAll(dims, IsPosInt),
        SPL(WithBases(self, rec(dimensions := dims, _children := [], dims := self >> self.dimensions)))),
    print := (self, i, is) >> Print("ImageUnk(", self.dims(), ")"),
    codeletShape := self >> let(d := self.dims(), Concat("ImageUnk_", StringInt(d[1]),"x",StringInt(d[2]))),
    fftImage := self >> ApplyFunc(O, TRDFT2D(self.dims()).dims()/self.dims()[2]),
    freqImage := self >> let(lindims := Product(TRDFT2D(self.dims()).dims())/(2*Product(self.dims())),
        fPrecompute(fConst(TComplex, lindims, 1+E(4)))),
    timeImage := self >> O(self.dims()[1], self.dims()[2]),
    randomImage := (self, m, n) >> ImageUnk(m, n),
    toAMat := self >> NullAMat(self.dimensions),
    unknownImage := self >> self
));

Class(ImageData, Mat, rec(
    print := (self, i, is) >> Print("ImageData(", Mat(self.element).dims(), ")"),
    codeletShape := self >> let(d := self.dims(), Concat("ImageData_", StringInt(d[1]),"x",StringInt(d[2]))),
    fftImage := self >> TRDFT2D(self.dims()).compute(self.element),
    freqImage := self >> let(fftRData := 1/Product(self.dims()) * Flat(self.fftImage()),
        fftCxData := List([1..Length(fftRData)/2], i->Complex(fftRData[2*i-1], fftRData[2*i])),
        FData(List(fftCxData, i->TComplex.value(i)))),
    timeImage := self >> self.element,
    randomImage := (self, m, n) >> let(data := List([1..m], i->List([1..n], j->FloatRat(Random([1..1000]/1000)))),
        ImageData(data)),
    unknownImage := self >> ImageUnk(self.dims())
));

Class(ImageVar, SumsBase, BaseContainer, rec(
    new := (self, dims) >> Checked(IsList(dims) and Length(dims) = 2 and ForAll(dims, IsPosInt),
        SPL(WithBases(self, rec(dimensions := dims, _children := [], dims := self >> self.dimensions)))),
    print := (self, i, is) >> Print("ImageVar(", self.dims(), ")"),
    codeletShape := self >> let(d := self.dims(), Concat("ImageVar_", StringInt(d[1]),"x",StringInt(d[2]))),
    fftImage := self >> ApplyFunc(O, TRDFT2D(self.dims()).dims()/self.dims()[2]),
    freqImage := self >> let(lindims := Product(TRDFT2D(self.dims()).dims())/(2*Product(self.dims())),
        type := TArray(TComplex, lindims), d := param(type, "freqImage"), FDataOfs(d, lindims, 0)),
    timeImage := self >> O(self.dims()[1], self.dims()[2]),
    randomImage := (self, m, n) >> ImageVar(m, n),
    toAMat := self >> NullAMat(self.dimensions),
    unknownImage := self >> ImageUnk(self.dims())
));


#F TRConv2D(img)
# For an image
# img := [ [ 1, 2, 3, 4 ],
#          [ 5, 6, 7, 8 ],
#          [ 9, 10, 11, 12 ],
#          [ 13, 14, 15, 16 ] ];
#
# MatSPL(TRConv2D(RealImage(img))) =
#
#[ [ 1, 4, 3, 2, 13, 16, 15, 14, 9, 12, 11, 10, 5, 8, 7, 6 ],
#  [ 2, 1, 4, 3, 14, 13, 16, 15, 10, 9, 12, 11, 6, 5, 8, 7 ],
#  [ 3, 2, 1, 4, 15, 14, 13, 16, 11, 10, 9, 12, 7, 6, 5, 8 ],
#  [ 4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5 ],
#  [ 5, 8, 7, 6, 1, 4, 3, 2, 13, 16, 15, 14, 9, 12, 11, 10 ],
#  [ 6, 5, 8, 7, 2, 1, 4, 3, 14, 13, 16, 15, 10, 9, 12, 11 ],
#  [ 7, 6, 5, 8, 3, 2, 1, 4, 15, 14, 13, 16, 11, 10, 9, 12 ],
#  [ 8, 7, 6, 5, 4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10, 9 ],
#  [ 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2, 13, 16, 15, 14 ],
#  [ 10, 9, 12, 11, 6, 5, 8, 7, 2, 1, 4, 3, 14, 13, 16, 15 ],
#  [ 11, 10, 9, 12, 7, 6, 5, 8, 3, 2, 1, 4, 15, 14, 13, 16 ],
#  [ 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 16, 15, 14, 13 ],
#  [ 13, 16, 15, 14, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2 ],
#  [ 14, 13, 16, 15, 10, 9, 12, 11, 6, 5, 8, 7, 2, 1, 4, 3 ],
#  [ 15, 14, 13, 16, 11, 10, 9, 12, 7, 6, 5, 8, 3, 2, 1, 4 ],
#  [ 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 ] ]
#
Class(TRConv2D, TaggedNonTerminal, rec(
    abbrevs := [ img -> [img]],
    dims := self >> self.params[1].dims(),
    isReal := True,
    terminate := self >>
        TIRDFT2D(self.params[1].dims()).terminate() *
        RCDiag(FList(TReal, 1/Product(self.params[1].dims())*Flat(self.params[1].fftImage()))) *
        TRDFT2D(self.params[1].dims()).terminate(),
    normalizedArithCost := (self) >>
        TRDFT2D(self.params[1].dims()).normalizedArithCost() +
        TIRDFT2D(self.params[1].dims()).normalizedArithCost() +
        3 * Rows(TRDFT2D(self.params[1].dims())),
    symTerminate := self >> let(N := self.params[1].dims(),
        imgv := Flat(self.params[1].timeImage()),
        tf := MDDFT(N),
        ti := MDDFT(N, -1),
        d := MatSPL(tf) * imgv,
        t := ti * (1/Product(N))*Diag(d) * tf,
        MatSPL(t)),
    forwardTransform := self >> let(t := TRDFT2D(self.params[1].dims()), tags := self.getTags(),
        When(
            Length(tags) >= 1 and ObjId(tags[1]) = paradigms.vector.AVecReg, let(n := t.dims()[1], v := tags[1].v,
                TCompose([TPrm(fTensor(fId(n/(2*v)), L(2*v, 2))), t]).withTags(tags)),
            t.withTags(tags))),

    HashId := self >> let(conv := self.params[1].dims(), When(IsBound(self.tags), Concatenation(conv, self.tags), conv)),

    hashAs := meth(self)
        local conv;
        conv := Copy(self);
        conv.params[1] := self.params[1].unknownImage();
        return conv;
    end
));


#F 2D RConv Rule
NewRulesFor(TRConv2D, rec(
    TRConv2D_TRDFT2D_tSPL := rec(
        switch := true,
        applicable := (self, t) >> true,
        children := (self, t) >>
            [[
                TCompose([
                    TIRDFT2D(t.params[1].dims()),
                    TRCDiag(fPrecompute(t.params[1].freqImage())),
                    TRDFT2D(t.params[1].dims())
                ]).withTags(t.getTags())
            ]],
        apply := (self, t, C, Nonterms) >> C[1]
    )
));


#Class(TRCorr2D, TaggedNonTerminal, rec(
#    abbrevs := [ img -> [img]],
#    dims := self >> self.params[1].dims(),
#    isReal := True,
#    terminate := self >> let(tf := TRDFT2D(self.params[1].dims()), d := 1/Product(self.params[1].dims()) * Flat(tf.compute(self.params[1].element)),
#        TIRDFT2D(self.params[1].dims()).terminate() *
#        RCDiag(FList(TReal, List([1..Length(d)], j->When(IsOddInt(j), d[j], -d[j])))) *
#        tf.terminate(),
#    normalizedArithCost := (self) >>
#        TRDFT2D(self.params[1].dims()).normalizedArithCost() +
#        TIRDFT2D(self.params[1].dims()).normalizedArithCost() +
#        3 * Rows(TRDFT2D(self.params[1].dims()))),
#    symTerminate := self >> let(N := self.params[1].dims(),
#        img := self.params[1].element,
#        rimg := Reversed(List(img, Reversed)),
#        imgv := Flat(rimg),
#        tf := MDDFT(N),
#        ti := MDDFT(N, -1),
#        d := MatSPL(tf) * imgv,
#        t := ti * (1/Product(N))*Diag(d) * tf,
#        MatSPL(t))
#));
