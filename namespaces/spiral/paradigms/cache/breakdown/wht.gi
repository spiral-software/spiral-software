
# Copyright (c) 2018-2019, Carnegie Mellon University
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

<#
NewRulesFor(TTensorI, rec(
	TTensorI_WHT_Mem := rec(
		switch := true,
		applicable := (self, nt) >> nt.params[1] > 2
			and not IsPrime
	)
));
#>
<#

NewRulesFor(WHT, rec(
	WHT_tSPL_CT_MemVec := rec(
		applicable := nt -> let(
			m := nt.getTag(AMem),
			v := nt.getTag(AVV),
			size := 2^nt.params[1]

			m
			and not IsList(m)
			and v
			and not IsList(v)
			and m.activation() = 0
			and size > 2 * v.vregs * v.vlen
		),

		children := (nt) -> let(
			v := nt.getTag(AVV),
			size := 2^nt.params[1],
			m := v.vregs / 2,
			n := size / m,

			TComplose(
				Inplace(TTensor(
					WHT(Log2Int(m)),
					AVec,
					AVec,
					n
				)),
				TTensor(
					WHT(Log2Int(n)),
					APar,
					APar,
					m
				)
			)
		),

		apply :=  (nt, c, cnt) -> [c],
			
		
	),

));


NewRulesFor(TTensor, rec(
	TTensor_WHT_Vec := rec(
		applicable := 
		children := 
		apply := 
	)
));
#>
