
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

NewRulesFor(DFT, rec(
	DFT_tSPL_CT_Mem := rec(
		switch := true,

		applicable := (self, nt) >> nt.params[1] > 2
			and not IsPrime(nt.params[1])
			and nt.hasTag(AMem),

		children := nt -> let(
			memtag := nt.getTag(AMem),
			Map2(
				Filtered(DivisorPairs(nt.params[1]), (m,n) -> n mod memtag.block = 0),
				(m,n) -> [
					TCompose([
						TGrp(TCompose([
							TTensorI(DFT(m, nt.params[2] mod n), n, AVec, AVec),
							TDiag(fPrecompute(Tw1(m*n, n, nt.params[2])))
						])),
						TGrp(TTensorI(DFT(n, nt.params[2] mod n), m, APar, AVec))
					]).withTags(nt.getTags())
				]
			)
		),

		apply := (nt, c, cnt) -> c[1]
	)
));
