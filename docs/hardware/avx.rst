.. _simd:

SIMD
====

.. only:: html

   .. contents::
   
Targeting SIMD Vector Instructions
++++++++++++++++++++++++++++++++++

Simple Example: Intel AVX 4-way Double Precision
------------------------------------------------

.. code-block:: none

	opts := SIMDGlobals.getOpts(AVX_4x64f); # default: real vectorization
	t := TRC(DFT(16)).withTags(opts.tags); 
	rt := RandomRuleTree(t, opts); 
	c := CodeRuleTree(rt, opts); 
	PrintCode("AVX_DFT16", c, opts); 

Stepwise Code Generation
------------------------

.. code-block:: none

	opts.tags;				# what are the tags
	opts.tags[1].v;			# vector length
	opts.tags[1].isa;			# targeted ISA
	opts.vector;				# check out the options used
	spl := SPLRuleTree(rt);		# There are SIMD SPL objects
	s := SumsRuleTree(rt, opts);		# and a SMP ISum
	InfinityNormMat(MatSPL(s) - MatSPL(t));	# correctness check

Complex Vectorization Example
-----------------------------

.. code-block:: none

	optsc := SIMDGlobals.getOpts(AVX_4x64f, # __m256d = (re,im,re,im)
		rec(realVect := false, cplxVect := true)); 
	rtc := RandomRuleTree(t, optsc); 	# complex vectorized DFT(16)
	sc := SumsRuleTree(rtc, optsc);	
	cc := CodeRuleTree(rtc, optsc); 	# far fewer shuffle operations
	PrintCode("AVXcplx_DFT16", cc, optsc); 

	
Tags
++++

Real Vectorization
------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\tags.gi
	Class(AVecReg, AGenericTag, rec(
		isReg := true, isRegCx := false, isVec := true,
		updateParams := meth(self)
			Checked(IsSIMD_ISA(self.params[1]));
			Checked(Length(self.params)=1);
			self.v := self.params[1].v; self.isa := self.params[1];
		end,
		container := (self, spl) >> 
			paradigms.vector.sigmaspl.VContainer(spl, self.isa)
	));

Complex Vectorization
---------------------

.. code-block:: none

	Class(AVecRegCx, AVecReg, rec(
		updateParams := meth(self)
			Checked(IsSIMD_ISA(self.params[1]));
			Checked(Length(self.params)=1);
			self.v := self.params[1].v/2; self.isa := self.params[1];
		end,
		container := (self, spl) >> 
			paradigms.vector.sigmaspl.VContainer(spl, self.isa.cplx()),
		isRegCx := true
	));


Vectorization Rules
+++++++++++++++++++

Simple Vectorization Rule Example
---------------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\breakdown.gi
	NewRulesFor(TTensorI, rec(
	AxI_vec := rec(
		forTransposition := false,
		applicable := nt -> nt.hasTags() and IsVecVec(nt.params) and 
			(nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)) and 
			IsInt(nt.params[2]/nt.firstTag().v),
		children := nt -> let(r := nt.params[2] / nt.firstTag().v,
				isa := nt.firstTag().isa,
				[[ When(r = 1,
					When(nt.numTags() = 1,
						nt.params[1].setWrap(VWrap(isa)),
						nt.params[1].setWrap(
							Drop(nt.getTags(), 1)).setWrap(VWrap(isa))
					),
					TTensorI(nt.params[1].setWrap(VWrap(isa)), r, 
						AVec, AVec).withTags(Drop(nt.getTags(), 1))
				)]]
			),
			apply := (nt, c, cnt) -> VTensor(c[1], nt.firstTag().v)
	   )
	));

Formula Rewrite
---------------

Kronecker Commute
#################

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\breakdown.gi
	NewRulesFor(TTensorI, rec(
		IxA_vec := rec(forTransposition := false,
		applicable := nt -> IsParPar(nt.params) and nt.hasTags() and 
			(nt.isTag(1,AVecReg) or nt.isTag(1,AVecRegCx)) and 
			IsInt(nt.params[2]/nt.firstTag().v),
		children := nt -> let(pv := nt.getTags(), v := pv[1].v,
			isa := pv[1].isa, d := nt.params[1].dims(),
			[[
				TL(d[1]*v, v, 1, 1).withTags(pv).setWrap(VWrapId),
				When(Length(pv)=1, nt.params[1].setWrap(VWrap(isa)), 
					nt.params[1].setpv(Drop(pv, 1)).setWrap(VWrap(isa))),
				TL(d[2]*v, d[2], 1, 1).withTags(pv).setWrap(VWrapId)
			]]),
		apply := (nt, c, cnt) -> let(
			l := nt.params[2] / nt.firstTag().v,
			A := c[1] * VTensor(c[2], nt.firstTag().v) * c[3],
			NoDiagPullin(When(l=1, A , Tensor(I(l), A))))
		)
	));


Wrapping
++++++++

VWrap Transformation
--------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\vwrap.gi
	Class(VWrap, VWrapBase, rec(
		__call__ := (self,isa) >> Checked(IsSIMD_ISA(isa),
			WithBases(self, rec(operations:=PrintOps, isa:=isa))),
		wrap := (self,r,t,opts) >> let(
			isa := self.isa, v := isa.v,
			tt := When(t.isReal(),
				@_Base(paradigms.vector.sigmaspl.VTensor(r.node, v), r),
					paradigms.vector.breakdown.AxI_vec(
						TTensorI(TRC(t).withTags([AVecReg(isa)]), v, 
							AVec, AVec).withTags([AVecReg(isa)]),   
					paradigms.vector.breakdown.TRC_VRCLR(
						TRC(t).withTags([AVecReg(isa)]), r))),
		print := self >> Print(self.name, "(", self.isa, ")"),   
	));

VWrap in Context of Search
--------------------------

Used in DP to time sub-trees.

.. code-block:: none

	opts := SIMDGlobals.getOpts(AVX_4x64f); 
	t := DFT(2).setWrap(VWrap(AVX_4x64f));
	rt := RandomRuleTree(t, opts);
	wrt := t.wrap.wrap(rt, t, opts);
	SPLRuleTree(wrt);


Special Role of Stride Permutations
+++++++++++++++++++++++++++++++++++

TL: Lift Stride Permutation to Non-Terminal Level
-------------------------------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\common\nonterms.gi
	Class(TL, Tagged_tSPL_Container, rec(
		abbrevs :=  [ (size, stride) -> Checked(ForAll([size, stride], 
			IsPosIntSym), [size, stride, 1, 1]),
				(size, stride, left, right) -> 
			Checked(ForAll([size, stride, left, right], IsPosIntSym), 
				[size, stride, left, right]) ],
		__call__ := arg >> let(self := arg[1], args := Drop(arg, 1),
			Cond(args=[1,1,1,1], I(1), ApplyFunc(Inherited,args))),
		dims := self >> 
			Replicate(2, self.params[1]*self.params[3]*self.params[4]),
		terminate := self >> Tensor(I(self.params[3]), 
			L(self.params[1], self.params[2]), I(self.params[4])),
		transpose := self >> TL(self.params[1], 
			self.params[1]/self.params[2], self.params[3], 
			self.params[4]).withTags(self.getTags()),
	));

TL in Context of Search
-----------------------

Used in DP to time sub-trees.

.. code-block:: none

	t := TL(8,2,2,4);
	t.terminate();


Architecture Specific Permutations
++++++++++++++++++++++++++++++++++

Looking up vectorized Implementations for TL
--------------------------------------------

.. code-block:: none

	Import(paradigms.vector.sigmaspl);
	opts := SIMDGlobals.getOpts(AVX_4x64f); 
	opts.breakdownRules.TL;
	t := TL(16,4,1,1).withTags(opts.tags);	# a in-register perm
	rt := RandomRuleTree(t, opts);
	HashLookup(opts.baseHashes[1], t);	# the implementation is cached

	s := SPLRuleTree(rt);
	vp := Collect(s, VPerm)[1];		# SPL object carries its code
	vp.code;				# code generator refers to ISA
	AVX_4x64f.rules; 			# ISA carries implementations 
	PrintCode("", vp.code(Y, X), opts);	# for in-register perms

SIMD ISA Database and Bootstrapping a Vector Architecture
---------------------------------------------------------

.. code-block:: none

	SIMD_ISA_DB;			# central SIMD data base
	SIMD_ISA_DB.installed();	# all the ISAs supported
	Doc(AVX_4x64f);		# The ISA carries all the relevant info
	Print(SIMD_ISA_DB.buildBases);	# How the base cases are built
	AVX_4x64f.buildRules;		# bootstrapping function
	SIMD_ISA_DB.getBases(AVX_4x64f);	# all the base cases needed
	SIMD_ISA_DB.lookupBases(AVX_4x64f);	# and how they are implemented


ISA Database and Hashed Breakdowns
++++++++++++++++++++++++++++++++++

Generic Breakdown Rules to Look Up Architecture Specific Code
-------------------------------------------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\bases\tl_bases.gi
	NewRulesFor(TL, rec(
		SIMD_ISA_Bases1 := rec(
			forTransposition := false,
			applicable := (self, t) >> t.isTag(1, AVecReg) and
				let(isa := t.firstTag().isa, P:=t.params,
					isa.active and ForAny(isa.supportedTL(),
					e -> _TL_applicable(e, P[1], P[2], P[3], P[4]))),
			apply := function(nt,C,cnt)
				local isa, tl, ll, vprm, P;
				P:=nt.params;
				isa := nt.firstTag().isa;
				tl := isa.getTL(P);
				ll := P[3] / tl.perm.l;
				vprm := tl.vperm;
				return When(ll = 1, vprm, 
					BlockVPerm(ll, isa.v, vprm, tl.perm.spl));
			end,
		)
	));


Stride Permutatiopn Identities as Rules
+++++++++++++++++++++++++++++++++++++++

Generic Breakdown Rules to Look Up Architecture Specific Code
-------------------------------------------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\common\breakdown.gi
	NewRulesFor(TL, rec(
		IxLxI_kmn_n := rec (forTransposition := false,
			applicable := nt -> 
				Length(DivisorsIntDrop(nt.params[1]/nt.params[2])) > 0,
			children := nt -> let(
				N := nt.params[1], n := nt.params[2],
				km := N/n, ml := DivisorsIntDrop(km),
				l := nt.params[3], r := nt.params[4],
				List(ml, m -> let( k := km/m, [
						TL(k*n, n, l, r*m).withTags(nt.getTags()),
						TL(m*n, n, k*l, r).withTags(nt.getTags())
					]))
				),
			apply := (nt, c, cnt) -> let(
				spl := c[1] * c[2],
				When(nt.params[1] = nt.params[2]^2,
					SymSPL(spl),
					spl
				)
			)
	));


SPL And Σ-SPL Vector Objects
++++++++++++++++++++++++++++

Generate The Example
--------------------

.. code-block:: none

	Import(paradigms.vector.sigmaspl);
	opts := SIMDGlobals.getOpts(AVX_4x64f); 
	rt := RandomRuleTree(TRC(DFT(16)).withTags(opts.tags), opts); 
	s := SPLRuleTree(rt); 			# SPL Objects
	ss := SumsRuleTree(rt, opts); 	# Sigma-SPL Objects

Inspecting the Vector Objects
-----------------------------

.. code-block:: none

	Collect(s, VTensor)[1];	# Vectorized Tensor(., I(v))
	Collect(ss, VTensor)[1];	# Vectorized Tensor(., I(v))
	Collect(s, VPerm)[1];		# Vectorized Prm(.)
	Collect(s, BlockVPerm)[1];	# Vectorized Tensor(I(.), Prm(.))
	Collect(s, VContainer)[1];	# Provides context for rewriting
	Collect(s, VRC)[1];		# Carries interleaved complex format
	Collect(ss, VGath)[1];		# Tensor(Gath(.), I(v))
	Collect(ss, VScat)[1];		# Tensor(Scat(.), I(v))
	Collect(ss, VRCDiag)[1];	# Vectorized Diag(.)


Vector SPL Objects
++++++++++++++++++

Vector Tensor SPL Object
------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\sigmaspl\vtensor.gi
	Class(VTensor, Tensor, rec(
		new := (self, L) >> SPL(WithBases(self, rec(
			_children := [L[1]],
			dimensions := When(IsBound(L[1].dims), L[1].dims(), 
			L[1].dimensions) * L[2], vlen := L[2]))),
		from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.vlen),
		print := (self,i,is) >> Print(self.name, "(",
			self.child(1).print(i+is,is), ", ", self.vlen, ")"),
		toAMat := self >> Tensor(self.child(1), I(self.vlen)).toAMat(),
		sums := self >> Inherit(self, rec(_children := 
			[self.child(1).sums()])),
		isPermutation := False,
		dims := self >> self.child(1).dims() * self.vlen,
		needInterleavedLeft := False,
		needInterleavedRight := False,
		transpose := self >> VTensor(self.child(1).transpose(), self.vlen),
		isBlockTransitive := true,
		cannotChangeDataFormat := False,
	));


Vector Σ-SPL Objects
++++++++++++++++++++

Vector Gather
-------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\sigmaspl\gather.gi
	Class(VGath, BaseVGath, SumsBase, rec(
		rChildren := self >> [self.func],
		rSetChild := rSetChildFields("func"),
		from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v),
		new := (self, func, v) >> SPL(WithBases(self,
			rec(func := func, v := v))).setDims(),
		dims := self >> [self.v*self.func.domain(), 
			self.v*self.func.range()],
		transpose := self >> VScat(self.func, self.v),
		print := (self,i,is) >> Print(self.name, "(", self.func, ", ", 
			self.v,")", self.printA()),
		toAMat := self >> let(v:=self.v, n := 
			EvalScalar(v*self.func.domain()),
				N := EvalScalar(v*self.func.range()),
				func := fTensor(self.func, fId(v)).lambda(),
				AMatMat(List([0..n-1], row -> BasisVec(N, 
					EvalScalar(func.at(row).ev()))))),
	));


Vector RC Objects: Manage Data Layout
+++++++++++++++++++++++++++++++++++++

Vector RC Class
---------------

.. code-block:: none

	Class(VRC, RC, rec(
		toAMat := (self) >> AMatMat(RCMatCyc(MatSPL(self.child(1)))),
		new := meth(self, spl, v)
			local res;
			res := SPL(WithBases(self, rec(_children:=[spl], v:=v,
									dimensions := spl.dimensions)));
			res.dimensions := res.dims();
			return res;
		end,
		print := (self, i, is) >> Print(self.__name__, 
			"(\n", Blanks(i+is), self.child(1).print(i+is,is), ", ",
		#"\n", Blanks(i+is), 
		self.v, 
		#"\n", Blanks(i),
		")", self.printA()),
		unroll := self >> self,
		transpose := self >> VRC(self.child(1).conjTranspose(), self.v),
		vcost := self >> self.child(1).vcost(),
		from_rChildren := (self, rch) >> ObjId(self)(rch[1], self.v)
	));

VRC, VRCL, VRCR, VRCLR
----------------------

.. code-block:: none

	v1 := VRC(Tensor(I(2), F(2)), 4); 	# Implicit data reorganization
	MatSPL(v1);
	v2 := VRCL(Tensor(I(2), F(2)), 4);	# The L and R encodes 
	MatSPL(v2);
	v3 := VRCR(Tensor(I(2), F(2)), 4);	# which side goes from
	MatSPL(v3);
	v4 := VRCLR(Tensor(I(2), F(2)), 4);	# interleaved complex format to
	MatSPL(v4);				# block split complex format

Terminate VRC, VRCL, VRCR, VRCLR
--------------------------------

.. code-block:: none

	Import(paradigms.vector.rewrite);
	opts := SIMDGlobals.getOpts(AVX_4x64f); 
	# see how the format gets propagated down the tree
	v5 := VRC(Tensor(I(2), F(2)) * Tensor(F(2), I(2)), 4);
	RulesVRC(v5);
	# when propagated to the leftmost/rightmost tree leaves, terminate
	v6 := VRCL(VTensor(F(2), 4), 4);
	v7 := Rewrite(v6, [RulesVRCTerm], opts);
	# termnination inserts VPerms to implement local data format change
	v8 := VRCR(VTensor(F(2), 4), 4);
	v9 := Rewrite(v9, [RulesVRCTerm], opts);


Vector SumsGen and Rewriting
++++++++++++++++++++++++++++

Default Sumsgen Handles Vector SPL to Σ-SPL
-------------------------------------------

.. code-block:: none

	opts := SIMDGlobals.getOpts(AVX_4x64f); 
	opts.sumsgen;
	opts.sumsgen.VTensor;
	opts.sumsgen.VPerm;

Rewrite Rule Strategies
-----------------------

.. code-block:: none

	opts.formulaStrategies;
	opts.formulaStrategies.sigmaSpl;
	opts.formulaStrategies.preRC;
	opts.formulaStrategies.postProcess;

Compile Strategies
------------------

.. code-block:: none

	opts.compileStrategy;
	Print(opts.vector.isa.fixProblems);


Vector CodeGen
++++++++++++++
	
Default Sumsgen handles SPL to Σ-SPL.

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\vector\sigmaspl\codegen.gi
	Class(VectorCodegen, DefaultCodegen, rec(
		VContainer := (self, o, y, x, opts) >>
			self(o.child(1), y, x, CopyFields(opts, rec(
				vector := rec(
					isa  := o.isa,
					SIMD := LocalConfig.cpuinfo.SIMDname )))),
		VPrm_x_I := (self, o, y, x, opts) >> 
			self(VTensor(Prm(o.func), o.v), y, x, opts),
		VPerm := (self, o, y, x, opts) >> o.code(y, x),
		VTensor := (self, o, y, x, opts) >> let(
			CastToVect := p -> StripList(List(Flat([p]), e -> 
			tcast(TPtr(TVect(opts.vector.isa.t.t, o.vlen)), e))),
			self(o.child(1), CastToVect(y), CastToVect(x), opts)),
		VGath := (self, o, y, x, opts) >> Cond(IsUnalignedPtrT(x.t),
			self(VGath_u(fTensor(o.func, fBase(o.v, 0)), o.v), y, x, opts),
			self(VTensor(Gath(o.func), o.v), y, x, opts)),
		VScat := (self, o, y, x, opts) >> Cond(IsUnalignedPtrT(y.t),
			self(VScat_u(fTensor(o.func, fBase(o.v, 0)), o.v), y, x, opts),
			self(VTensor(Scat(o.func), o.v), y, x, opts))
	)); 


Vector Instructions
+++++++++++++++++++

Every ISA defines ISA specific instructions and polymorphic add,…

.. code-block:: none

	# spiral-core\namespaces\spiral\platforms\avx\code.gi
	# __m256d _mm256_insertf128_pd(__m256d a, __m128d b, int offset);
	Class(vinsert_2l_4x64f, VecExp_4.binary(), rec(
		ev := self >> let( 
			a := _unwrap(self.args[1].ev()),
			b := _unwrap(self.args[2].ev()),
			When( self.args[3].p[1] = 0, 
				b :: a{[3 .. 4]}, a{[1 .. 2]} :: b )),
		computeType := self >> self.args[1].t, 
	));

	# __m256d _mm256_unpackhi_pd(__m256d a, __m256d b);
	Class(vunpackhi_4x64f, VecExp_4.binary(), rec(
		semantic := (in1, in2, p) -> [in1[2], in2[2], in1[4], in2[4]],
		ev := _evpack 
	));

	#__m256 _mm256_blend_ps(__m256 m1, __m256 m2, const int mask);
	Class(vblend_8x32f,  VecExp_8.binary(), rec(
		semantic := (in1, in2, p) -> 
		List( Zip2(TransposedMat([in1, in2]), p), e -> e[1][e[2]]),
		params := self >> Replicate(8, [1,2]), ev := _evshuf2
	));


Vector Strength Reduction and Fixup Rules
+++++++++++++++++++++++++++++++++++++++++

.. code-block:: none

	# spiral-core\namespaces\spiral\platforms\avx\sreduce.gi
	RewriteRules(FixCodeAVX, rec(
		fix_noneExp := Rule( noneExp, e -> e.t.zero()),
		vpermf128_8x32f_to_vextract := Rule( [assign, [deref, @(1)], 
			[vpermf128_8x32f, @(2), @(3), @(4)]],
			e -> let( p := @(4).val.p, 
				a := [[@(2).val, [0]], [@(2).val, [1]], [@(3).val, 
				[0]], [@(3).val, [1]]],
				dst := tcast(TPtr(TVect(T_Real(32), 4)), @(1).val),
				chain(
					assign(deref(dst), 
						ApplyFunc(vextract_4l_8x32f, a[p[1]])),
					assign( deref(dst+1), 
						ApplyFunc(vextract_4l_8x32f, a[p[2]])))
			)),
		addsub_4x64f_to_mul := Rule( [addsub_4x64f, _0, @(1)],
			e -> mul(e.t.value([-1,1,-1,1]), @(1).val)),
		addsub_8x32f_to_mul := Rule( [addsub_8x32f, _0, @(1)],
			e -> mul(e.t.value([-1,1,-1,1,-1,1,-1,1]), @(1).val)),
		avx_add_addsub_vzero := Rule([add, @(1), [@(2, [addsub_4x64f,
			addsub_8x32f]), _0, @(3)]], 
			e -> ObjId(@(2).val)(@(1).val, @(3).val)), 
	));


Vector Unparser
+++++++++++++++

Polymorphic Unparsing for standard icode, adds special instructions

.. code-block:: none

	# spiral-core\namespaces\spiral\platforms\avx\unparse.gi
	Class(AVXUnparser, SSEUnparser, rec(
		TVect := (self, t, vars, i, is) >> let(
			ctype := self.ctype(t, _isa(self)),
			Print(ctype, " ", self.infix(vars, ", "))),
		vpack := (self, o, i, is) >> Cond( _avxT(o.t, self.opts),
				Print("_mm256_set_", self.ctype_suffix(o.t, _isa(self)), 
			"(", self.infix(Reversed(o.args), ", "), ")"),        
			Inherited(o, i, is)),
		sub := (self, o, i, is) >> Cond( _avxT(o.t, self.opts), let(
			isa := _isa(self),
			sfx := self.ctype_suffix(o.t, isa),
			saturated := When(isa.isFixedPoint and  
				isa.saturatedArithmetic, "s", ""),
				self.printf("_mm256_sub$1_$2($3, $4)", [saturated, sfx, 
					o.args[1], o.args[2]])),
			Inherited(o, i, is)),
		vextract_2l_4x64f := (self, o, i, is) >> 
			self.prefix("_mm256_extractf128_pd", o.args),
		vstore_2l_4x64f := (self, o, i, is) >> 
			self.prefix("_mm256_extractf128_pd", o.args),    
	));


ISA Definition
++++++++++++++

ISA Definition file ties everyting together

.. code-block:: none

	# spiral-core\namespaces\spiral\platforms\avx\isa.gi
	Class(AVX_4x64f, AVX_Intel, rec(
		includes     := () -> ["<include/omega64.h>"] :: _AVXINTRIN(),
		v       := 4,
		t       := TVect(T_Real(64), 4),
		ctype   := "double",
		instr   := [ vunpacklo_4x64f, vunpackhi_4x64f, vshuffle_4x64f, 
			vperm2_4x64f, vpermf128_4x64f, vperm_4x64f, vblend_4x64f ],
		mul_cx := (self, opts) >>
			((y, x, c) -> let( u1 := self.freshU(), u2 := self.freshU(), 
				u3 := self.freshU(),
				decl([u1, u2, u3], chain(
					assign(u1, mul(x, vunpacklo_4x64f(c, c))),
					assign(u2, vshuffle_4x64f(x, x, [2,1,2,1])),
					assign(u3, mul(u2, vunpackhi_4x64f(c, c))),
					assign(y, addsub_4x64f(u1, u3)))))),
		svload_init := (vt) -> [
		[ y,x,opts) -> let(u1 := var.fresh_t("U", TVect(vt.t, 2)), 
				decl([u1], chain(
					assign(u1, vload1sd_2x64f(x[1].toPtr(vt.t))),
					assign(y, vinsert_2l_4x64f(vt.zero(), u1, [0]))))), 
		...
	));


Vector Benchmarking Infrastructure
++++++++++++++++++++++++++++++++++

LocalConfig provides unit tests.

.. code-block:: none

LocalConfig.bench;

	Create a test and run it.

.. code-block:: none

	dpbench := LocalConfig.bench.AVX().4x64f.1d.dft_ic.medium();
	dpbench.runAll();












	

