
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(ISA_Bridge);

IsISABridge := (obj) -> IsRec(obj) and IsBound(obj._ISABridge) and obj._ISABridge=true;

#F VCvt(<n>, <ISA_Bridge>)
#F data type conversion from one ISA into another.
#F
# OBSOLETE, will be removed
Class(VCvt, TCast, rec(
    abbrevs := [],
    def := (n, brige) -> Perm((), n),
    
    dmn := self >> [ TArray(self.params[2].isa_from.gt(), self._spl().dims()[2]) ],
    rng := self >> [ TArray(self.params[2].isa_to.gt(),   self._spl().dims()[1]) ],

    toAMat := self >> self._spl().toAMat(),

    _spl := self >> self.params[2].splVCvt(self.params[1]),
));



Class(Cvt, SumsBase, Sym, rec(
    abbrevs := [  (isa_bridge) -> Checked(IsISABridge(isa_bridge), [isa_bridge]) ],
    def := (isa_bridge) -> Perm((), isa_bridge.granularity()),

    n         := self >> self.params[1].granularity(),
    isa_to    := self >> self.params[1].isa_to,
    isa_from  := self >> self.params[1].isa_from,

    dmn := self >> [ TArray(self.isa_from().t.base_t(), self.n()) ],
    rng := self >> [ TArray(self.isa_to().t.base_t(), self.n()) ],

    isPermutation := False,

    toAMat := self >> self.params[1].toAMat(),

    #doNotMeasure := true
));


Class(TCvt, Tagged_tSPL, rec(
    abbrevs := [
        (n, isa_to, isa_from) -> Checked(IsPosIntSym(n), IsISA(isa_to), IsISA(isa_from), [n, isa_to, isa_from, []]),
        (n, isa_to, isa_from, props) -> Checked(IsPosIntSym(n), IsISA(isa_to), IsISA(isa_from), IsList(props), [n, isa_to, isa_from, props]),
    ],

    #def := (n, isa_to, isa_from) -> Perm((), n),

    n         := self >> self.params[1],
    isa_to    := self >> self.params[2],
    isa_from  := self >> self.params[3],
    props     := self >> self.params[4],

    dmn := self >> [ TArray(self.isa_from().t.base_t(), self.n()) ],
    rng := self >> [ TArray(self.isa_to().t.base_t(), self.n()) ],

    toAMat := self >> I(self.n()).toAMat(),

    isReal := self >> true,

    isAuxNonTerminal := true,

    hashAs := self >> ObjId(self)(8*Lcm(self.isa_to().v, self.isa_from().v), self.isa_to(), self.isa_from(), self.props()).takeTA(self),
));


#
# ISA_Bridge: represents conversion from one SIMD_ISA to another.
# Each descendant implements particular conversion case and registered in 
# the ISA_Bridge object by [source ISA, destination ISA] pair.
# 

# <props> field 
# "wraparound" | "saturation"
# "trunc" | "round" | "ceil" | "floor"


Class(ISA_Bridge, rec(
    _ISABridge := true,
    _table     := rec(),

    __call__ := meth(self) 
        local inst;
        inst := WithBases(self);
        if not IsBound(self._table.(inst.isa_from.id())) then
            self._table.(inst.isa_from.id()) := rec();
        fi;
        if not IsBound(self._table.(inst.isa_from.id()).(inst.isa_to.id())) then
            self._table.(inst.isa_from.id()).(inst.isa_to.id()) := [];
        fi;
        Add(self._table.(inst.isa_from.id()).(inst.isa_to.id()), inst);
        return inst;
    end,

    add := (self, class) >> class(),

    printAvailableBridges := meth(self)
        local all;
        all := Flat(List(UserRecValues(self._table), r -> UserRecValues(r)));
        Sort(all, (a,b) -> let(a1 := StringPrint(a.isa_from), b1 := StringPrint(b.isa_from), When(a1=b1, StringPrint(a.isa_to)<StringPrint(b.isa_to), a1<b1)));
        Print(PrintPad("From", 27), PrintPad("To", 27), "Props\n", Replicate(60, '-'), "\n");
        DoForAll(all, e -> Print(PrintPad(StringPrint(e.isa_from), 26), " ", PrintPad(StringPrint(e.isa_to), 26), " ", e.props, "\n"));
    end,

    _applicable_cvt := (props, min_range, range_in, bridge) -> let(
        range_from  := bridge.isa_from.t.range(),
        range_to    := bridge.isa_to.t.range(),
        range_mid   := bridge.range(),

        range       := Cond( range_in=false, range_from, range_from * range_in),
        range_req   := Cond( range_in=false, min_range,  min_range  * range_in),
        range_mul   := range_to * range_mid * range_from,

        class_range := Cond( range.min >= range_to.min and range.max <= range_to.max and
                             range.min >= range_mid.min and range.max <= range_mid.max, [], ["saturation", "wraparound"]),
        class_prec  := Cond( range_from.eps = range_to.eps and range_from.eps = range_mid.eps,    [], ["round", "trunc", "ceil", "floor"]),
        bridge_class_range  := Intersection(class_range, props),
        bridge_class_prec   := Intersection(class_prec,  props),
        
        Intersection(bridge_class_range, bridge.props) = bridge_class_range
        and Intersection(bridge_class_prec, bridge.props) = bridge_class_prec
        and range_mul.min <= range_req.min and range_mul.max >= range_req.max
        and range_mul.eps <= range_req.eps
    ),

    # bridge: returns ISA_Bridge objects list which implement conversion from isa_from to isa_to

    find := (arg) >> let(
        self     := arg[1],
        isa_to   := arg[2],
        isa_from := arg[3],
        props    := arg[4],
        min_range:= Cond( Length(arg)>4, arg[5], isa_to.t.range() * isa_from.t.range() ),
        range_in := Cond( Length(arg)>5, arg[6], isa_from.t.range()),

        Cond( IsBound(self._table.(isa_from.id())) and IsBound(self._table.(isa_from.id()).(isa_to.id())),
            Filtered(self._table.(isa_from.id()).(isa_to.id()), 
                e -> self._applicable_cvt(props, min_range, range_in, e)),
            [] )
    ),
    
    #######################
    # descendants should define the fields listed below.
    
    # isa_from: source SIMD_ISA 
    isa_from := false,
    # isa_to: destination SIMD_ISA 
    isa_to   := false,

    # minimal range
    range := (self) >> self.isa_to.t.range() * self.isa_to.t.range(),
    
    # NOTE: bridge conversion properties, explain
    # For example float-to-int conversion can be implemented with different rounding methods bla bla bla
    props       := [],

    # code(<y>, <x>, <opts>): called from codegen to get code that implements Cvt(...);
    code        := (self, y, x, opts) >> Error("not implemented"),

    # granularity: Cvt object height and width (which are the same).
    granularity := self >> Lcm(self.isa_to.v, self.isa_from.v),

    # toAMat: implements Cvt.toAMat()
    toAMat      := abstract(), 

    # toSpl: conversion block formula NOTE: explain
    toSpl       := abstract(), 

    ##############################################
    # helpers

    _x := (self, x, offs) >> vtref(self.isa_from.t, x, offs),
    _y := (self, y, offs) >> vtref(self.isa_to.t,   y, offs),

    _mkTL       := meth(self, isa, n, s)
        local t, hentry, sums;
        t := TL(n, s).withTags(isa.getTags());
        hentry := HashLookup(SIMD_ISA_DB.getHash(), t);
        if hentry=false then
            #return Error("SIMD_ISA_DB lookup failed, tried ", t); 
            return Prm(L(n, s));
	else
            sums := _SPLRuleTree(hentry[1].ruletree).sums();
            sums := When(IsBound(sums.unroll), sums.unroll(), sums);
            return sums;
	fi;
    end,

));


# ISA_Bridge base class for bridges for which Cvt object is identity
Class(ISA_Bridge_I, ISA_Bridge, rec(
    toAMat      := (self) >> I(self.granularity()).toAMat(),
    toSpl       := (self) >> Cvt(self),
));

Class(ISA_Bridge_VvL, ISA_Bridge, rec(
    toAMat      := (self) >> L(self.granularity(), self.isa_from.v).toAMat(),
    toSpl       := (self) >> Cvt(self)*self._mkTL(self.isa_to, self.granularity(), div(self.granularity(), self.isa_from.v)),
));

Class(ISA_Bridge_VvR, ISA_Bridge, rec(
    toAMat      := (self) >> L(self.granularity(), self.isa_from.v).toAMat(),
    toSpl       := (self) >> self._mkTL(self.isa_to, self.granularity(), div(self.granularity(), self.isa_from.v)) * Cvt(self),
));

# plain conversion from one scalar data type to another, 
# assuming isa_to type range includes isa_from type range
Class(ISA_Bridge_tcast, ISA_Bridge_I, rec(
    code  := (self, y, x, opts) >> assign( self._y(y,0), tcast(self.isa_to.t, self._x(x,0)) ),
));

# reinterpretation
Class(ISA_Bridge_wrap, ISA_Bridge_I, rec(
    range := (self) >> self.isa_to.t.range(),
    props := ["wraparound"],
    code  := (self, y, x, opts) >> assign( self._y(y,0), tcast(self.isa_to.t, self._x(x,0)) ),
));


_debug_print_cvt_chains := false;

DebugCVT := function(value)
    _debug_print_cvt_chains := value;
end;

# _cvt_build_chains( <TCvt non terminal> ) returns all data conversion chains available for given
#   TCvt object.
#

Declare(_cvt_build_chains_rec);
_cvt_build_chains_rec := function(chain, isa, min_range, range, props, available_isa)
    local ifrom, fr, r, b, dbg_print;
    ifrom := Last(chain).isa_to;
    if ifrom = isa then
        if _debug_print_cvt_chains then
            Print(GreenStr("CVT: "), ConcatSepList(chain, e -> e.__name__, " -> "), "\n");
        fi;
        return [chain];
    else
        r  := [];
        fr := ConcatList( available_isa, isa -> ISA_Bridge.find( isa, ifrom, props, min_range, range ));
        dbg_print := fr = [] and _debug_print_cvt_chains;
        for b in fr do
            Append(r, _cvt_build_chains_rec(chain :: [b], isa, min_range, b.range()*range, props, RemoveList(available_isa, b.isa_to)));
        od;
    fi;
    if dbg_print then
        Print(RedStr("Dead CVT: "), ConcatSepList(chain, e -> e.__name__, " -> "), "\n");
    fi;
    return r;
end;

_cvt_build_chains := function(t)
    local available_isa, r, range, fr, b, min_range;
    available_isa := Set(t.firstTag().params[1] :: [t.isa_from(), t.isa_to()]);#Set(RemoveList(t.firstTag().params[1], t.isa_from()) :: [t.isa_to()]);
    r         := [];
    range     := StripList(t.getA("r_in", [t.isa_from().t.range()]));
    min_range := t.isa_to().t.range() * t.isa_from().t.range();
    fr        := ConcatList( available_isa, isa -> ISA_Bridge.find( isa, t.isa_from(), t.props(), min_range, range ));
    for b in fr do
        Append(r, _cvt_build_chains_rec([b], t.isa_to(), min_range, b.range()*range, t.props(), RemoveList(available_isa, b.isa_to)));
    od;
    return Sort(r);
end;


NewRulesFor( TCvt, rec(
   TCvt_Term := rec(
       forTransposition := false,
       
       applicable := t -> t.firstTagIs(AMultiVec),
       freedoms   := t -> [ _cvt_build_chains(t) ],
       child      := (t, fr) -> [InfoNt(Reversed(fr[1]))],
       apply      := (self, t, C, nt) >> let(
           # merge adjasent loops when possible
           l := self._merge_loops(List(nt[1].params[1], b -> rec( 
                   # NOTE: need ESReduce here to deal with symbolic expressions
                   #        this way can figure out divisibility at least some times
                   its := ESReduce(div(t.n(), b.granularity()), spiral.SpiralDefaults), bridge := b
                ))),
           
           Compose(List(l, e -> self._to_ISum(e)))           
       ),
       
       _to_ISum := (self, r) >> let(
           ch   := Cond( IsBound(r.bridge), [r.bridge.toSpl()], List(r.children, e -> self._to_ISum(e))),
           cols := Cols(Last(ch)),
           rows := Rows(ch[1]),
           i    := Ind(r.its),
           ISum(i, Compose( [Scat(H(r.its*rows, rows, rows*i, 1))] :: ch :: [Gath(H(r.its*cols, cols, cols*i, 1))]))
       ),

       _merge_loops := meth(self, l)
           local c, e, new_ch, r;
           r := [];
           c := l[1];
           for e in Drop(l, 1) do
               if c.its <> 1 and _divides(c.its, e.its) then
                   new_ch := rec(its := ESReduce(div(e.its, c.its), spiral.SpiralDefaults), bridge := e.bridge);
                   if IsBound(c.bridge) then
                       # it's a single bridge, creating new node and making this one child
                       c := rec(
                           its      := c.its,
                           children := [rec(its := 1, bridge := c.bridge), new_ch]
                       );
                   else
                       Add(c.children, new_ch);
                   fi;
               elif e.its<>1 and _divides(e.its, c.its) then
                   new_ch := rec(its := ESReduce(div(c.its, e.its), spiral.SpiralDefaults));
                   if IsBound(c.bridge) then
                       new_ch.bridge := c.bridge;
                   else
                       new_ch.children := c.children;
                   fi;
                   c := rec(
                       its      := e.its,
                       children := [new_ch, rec(its := 1, bridge := e.bridge)]
                   );
               else
                   Add(r, c);
                   c := e;
               fi;
           od;
           
           Add(r, c);
      
           # recurse on children
           return List(r, e -> Cond(IsBound(e.children), CopyFields(e, rec(children := self._merge_loops(e.children))), e));
       end,
           
   ),
));

