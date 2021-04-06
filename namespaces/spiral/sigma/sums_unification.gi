
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(PropagateUnification);

list_dup := (n, a) -> Cond(IsList(a), Checked(Length(a)=n, a), Replicate(n, a));

getXTypeOpts := (t, opts) -> When(IsBound(t.a.t_in),  List(t.a.t_in,  e -> TPtr(e)), list_dup(t.arity()[2], opts.XType));
getYTypeOpts := (t, opts) -> When(IsBound(t.a.t_out), List(t.a.t_out, e -> TPtr(e)), list_dup(t.arity()[1], opts.YType));

# TypeOpts(<sums>, <opts>) returns options with 'XType' and 'YType' fields set acording 
# to 't_in' and 't_out' attributes (when available). 
# Playing ping-pong between SumsUnification and TypeOpts allows to perform search for OL and 
# multi data type formulas on existing DP search infrastructure.

TypeOpts := (sums, opts) -> Cond( not IsBound(opts.XType) or not IsBound(opts.YType), opts,
    CopyFields(opts, rec(
        XType := StripList(getXTypeOpts(sums, opts)),
        YType := StripList(getYTypeOpts(sums, opts))
)));

# SumsUnification sets data type 't_in' and 't_out' attributes on sums tree..
# It propagates 't_in' types of sums root object (if available) or opts.XType.
SumsUnification := function(sums, opts)
    if IsBound(opts.doSumsUnification) and opts.doSumsUnification then
        sums := PropagateUnification(sums, List(getXTypeOpts(sums, opts), e -> e.t));
    fi;
    return sums;
end;

Class(PropagateUnification, HierarchicalVisitor, rec(

     # makes sure the object is not a polymorphic function/SPL like L,
     # which would cause trouble because they are handled below via
     # FuncClass, and thus .a.t_in and t_out are not set
     _isSPL  := x -> IsSPL(x) and not IsFunction(x),

     _unify  := (t1, t2) -> Cond( IsList(t1) and IsList(t2) and Length(t1)=Length(t2), 
                                   List([1..Length(t1)], i -> UnifyPair(t1[i], t2[i])),
                               IsList(t1) and IsType(t2),
                                   List([1..Length(t1)], i -> UnifyPair(t1[i], t2)),
                               IsType(t1) and IsType(t2),
                                   UnifyPair(t1, t2),
                               Error("PropagateUnification: type dimensions mismatch")),
     _to_cplx := (lst) -> List(lst, e -> T_Complex(e)),
     _to_real := (lst) -> List(lst, e -> e.realType()),

     TCast := meth(self, o, inputs)
         o.params[3] := UnifyTypes(inputs :: [o.params[3]]);
         return o.withA( t_in  => [o.params[3]],
                         t_out => [o.params[2]]); 
     end,

     TCvt := (self, o, inputs) >>
         o.withA( t_in  => [o.isa_from().t.base_t()],
                  t_out => [o.isa_to().t.base_t()  ] ),

     Cvt := ~.TCvt,

     Compose := (self, o, inputs) >> let(
         ch    := DropLast(ScanR( o.rChildren(), (l, e) -> When(IsList(l), self(e, l), self(e, l.a.t_out)), inputs ), 1),
         o.from_rChildren(ch).withA( t_in  => Last(ch).a.t_in,
                                     t_out => ch[   1].a.t_out)),

     RC := ( self, o, inputs) >> let(
         ch := List(o.rChildren(), e -> When( self._isSPL(e), self(e, self._to_cplx(inputs)), e)),
         o.from_rChildren(ch).withA(t_in => self._to_real(ch[1].a.t_in), t_out => self._to_real(ch[1].a.t_out))),
     GRC := ~.RC,
     TRC := ~.RC,

     BaseMat := (self, o, inputs) >> let(
         dmn := List(o.dmn(), e -> e.t.base_t()),
         rng := List(o.rng(), e -> e.t.base_t()),
         inp := List(Zip2(dmn, inputs), UnifyTypes),
         # we have no idea how to propagate data types for objects which arity is other than [1, 1],
         # let's hope object defined it's rng() and dmn() data types or uses same data type everywhere
         # or data type propagated horizontally in the case of same arity on the left and on the right.
         rnginp := Replicate( Length(rng), UnifyTypes(inp) ),
         out := List(Zip2(rng, Cond( Length(dmn)=Length(rng), inp, rnginp )), UnifyTypes),
         o.withA( t_in => inp, t_out => out )),

     ClassSPL := (self, o, inputs) >> let(
         ch := List(o.rChildren(), e -> When( self._isSPL(e), self(e, inputs), e )),
         spl_ch := Filtered(ch, self._isSPL),
         When( spl_ch <> [],
             o.from_rChildren(ch).withA(
                 t_in  => FoldL1(List(spl_ch, x->x.a.t_in ), self._unify),
                 t_out => FoldL1(List(spl_ch, x->x.a.t_out), self._unify)),
             self.BaseMat(o, inputs))),

     Cross := meth (self, o, inputs) 
         local ch, i, j, d;
         ch := o.rChildren();
         j  := 1;
         for i in [1..Length(ch)] do
             d     := Length(ch[i].dmn());
             ch[i] := self(ch[i], Flat(inputs{[j..j+d-1]}));
             j     := j + d;
         od;
         return o.from_rChildren(ch).withA(
                     t_in  => Flat(List(ch, e -> e.a.t_in)),
                     t_out => Flat(List(ch, e -> e.a.t_out)));
     end,

     SMPBarrier := (self, o, inputs) >> let( ch := self(o.child(1), inputs),
         ObjId(o)(o.nthreads, o.tid, ch).withA( t_in => ch.a.t_in, t_out => ch.a.t_out)),

     VContainer := ~.ClassSPL, 
     SumsBase   := ~.ClassSPL,
     VPerm      := ~.BaseMat,
     
      
     TaggedNonTerminal := meth( self, o, inputs)
         local tag, tL, tR, tM;
         tag := FirstDef(o.getTags(), t -> Global.autolib.IsACVec(t), false);
         if tag=false then
             return self.ClassSPL(o, inputs);
         else
             tL := List(Flat([tag.isaL()]), e->e.t.base_t()); 
             tR := List(Flat([tag.isaR()]), e->e.t.base_t());
             tM := List(list_dup(o.arity()[2], tag.isaM()), e->e.t.base_t());
             if IsComplexT(inputs[1]) then
                 tL := List(tL, e -> T_Complex(e));
                 tR := List(tR, e -> T_Complex(e));
                 tM := List(tM, e -> T_Complex(e));
             fi;
             return self.ClassSPL(o, tM).withA(t_in => tR, t_out => tL);
         fi;
     end,

     GTBase := ~.TaggedNonTerminal, 

     FuncClass     := Ignore,
     FuncClassOper := Ignore,
));

