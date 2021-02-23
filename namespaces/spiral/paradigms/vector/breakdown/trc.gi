
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################################
#   TRC rules

_VBlkMatSPL := function(mat, vtype)
   local i, n, m, nv, a, v, start, final, diff, m_zeros;
   [n, m] := DimensionsMat(mat);
   start := 1;
   a := [];
   v := vtype.size;
   nv := approx.CeilingRat(n/v);
   m_zeros := Replicate(m, vtype.t.zero().v);
   for i in [1 .. nv] do
       final := start + v - 1;
       if (final > n)  then
           diff := final - n;
           Add(a, List(TransposedMat(mat{[start .. n]} :: Replicate(diff, m_zeros)), 
		       vec -> vtype.value(vec)));
       else
           Add(a, List(TransposedMat(mat{[start .. final]}), 
		       vec -> vtype.value(vec)));
       fi;
       start := final + 1;
   od;
   return a;
end;

VectorizedMatSPL := (isa, spl) -> let(
    r   := Rows(spl),
    c   := Cols(spl),
    v   := isa.v,
    mat := Cond(IsMat(spl), spl, MatSPL(spl)),
    tt  := VBlk(_VBlkMatSPL(mat, isa.t), v) * VGath_dup(fId(c), v),
    When(Mod(r,v)=0, tt, VScat_pc(r,r,0,v)*tt)
);

NewRulesFor(TRC, rec(
#   base cases
    TRC_vect := rec(
        info := "TRC vect",
        forTransposition := false,
        applicable := t -> t.hasTag(AVecReg),
        children := t -> let(
            tags := t.getTags(),
            vecTag := t.getTag(AVecReg, 1),
            [[ t.params[1].withTags(tags).setWrap(VWrapTRC(vecTag.isa)) ]]
        ),
        apply := (t, c, nt) -> VRC(c[1], t.getTag(AVecReg, 1).v)
    ),

    TRC_VRCLR := rec(
        info := "TRC vect",
        forTransposition := false,
        applicable := t -> t.hasTags() and t.isTag(1, AVecReg),
        children := t -> [[ t.params[1].withTags(t.getTags()) ]],
        apply := (self, t, C, Nonterms) >> RC(C[1]),
        switch := false
    ),

    TRC_cplx := rec(
        forTransposition := false,
        applicable := t -> t.hasTag(AVecRegCx),
        children := t -> let(
            tags := t.getTags(),
            tag  := t.getTag(AVecRegCx, 1),
            nt   := t.params[1],
            [[ nt.withTags(tags).setWrap(VWrapTRCcplx(tag.isa)) ]]
        ),
        apply := (self, t, C, Nonterms) >> vRC(C[1]),
        switch := false
    ),

    # TRC_cplxvect will fire even when asked for real vectorization
    TRC_cplxvect := rec(
        forTransposition := false,
        applicable := t -> t.hasTag(AVecRegCx) or t.hasTag(AVecReg),
        children := t -> let(
            tags := t.getTags(),
            tag := Filtered([t.getTag(AVecReg, 1), t.getTag(AVecRegCx, 1)], i->not IsBool(i))[1],
            nt := t.params[1],
            ctag := When(ObjId(tag)=AVecRegCx, tag, AVecRegCx(tag.isa)),
            newTags := List(tags, i->When(i=tag, ctag, i)),
            [[ nt.withTags(newTags).setWrap(VWrapTRCcplx(tag.isa)) ]]
        ),
        apply := (self, t, C, Nonterms) >> vRC(C[1]),
        switch := false
    ),

    TRC_cplx_v2 := rec(
        forTransposition := false,
        applicable := t -> t.hasTag(AVecRegCx) and t.getTag(AVecRegCx, 1).isa.v = 2,
        children := t -> let(
            tags := t.getTags(),
            tag := t.getTag(AVecRegCx, 1),
            nt := t.params[1],
            newTags := Filtered(tags, i -> i<>tag),
            [[ nt.withTags(newTags).setWrap(VWrapTRCcplx(tag.isa)) ]] 
        ),
        apply := (self, t, C, Nonterms) >> vRC(VTensor(C[1], 1)),
        switch := false
    ),

    TRC_By_Def := rec(
         forTransposition := false,
	 maxSize := 128,
         applicable := (self, t) >> Rows(t) <= self.maxSize and Cols(t)<=self.maxSize and 
       	                            t.hasTags() and t.isTag(1, AVecReg) and t.free()=[],
         apply := (self, t, C, Nonterms) >> let(
	     isa := t.getTag(AVecReg, 1).isa,
	     VectorizedMatSPL(isa, RC(t.params[1]))
	 )
    )
));
