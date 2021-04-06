
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

NewRulesFor(WHT, rec(
	WHT_tSPL_CT_Mem := rec(
		switch := true,
		applicable := (self, nt) >> nt.params[1] > 2
			and not IsPrime(nt.params[1])
			and nt.hasTag(AMem) 						# make sure we're tagged
			and not IsList(nt.getTag(AMem)) 			# make sure we have only ONE memory tag (single level of cache)
			and nt.params[1] > nt.getTag(AMem).size,	# problem size must be larger than the cache

		# atm, just handle a single AMem
		children := nt -> let(
			memtag := nt.getTag(AMem),
			Map2(
				Filtered(DivisorPairs(nt.params[1]), (m,n) -> n mod memtag.block = 0, m <= memblock.assoc),
				(m,n) -> [
					TCompose([
						TTensorI(WHT(m), n, AVec, AVec),
						TTensor(WHT(n), m, APar, APar)
					])
				]
			)
		),

		apply := (nt, c, cnt) -> c[1]
	)
));

