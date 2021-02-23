
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(paradigms.vector);

Class(LSKernel, SumsBase, BaseContainer, rec(
    doNotMarkBB:= true,
    abbrevs := [ ch -> [ch,0], (ch,ops) -> [ch,ops] ],
    new := (self, ch, ops) >> SPL(WithBases(self, rec(
        info := Cond(
            IsInt(ops) or IsRat(ops) 
                or IsValue(ops) or IsExp(ops), rec(
                    opcount := When(IsValue(ops),ops.v,ops),
                    free := Set([]),
                    loadFunc := fId(ch.dimensions[2]),
                    storeFunc := fId(ch.dimensions[1])
                ),

            IsRec(ops), ops,

            Error("unknown info")
        ),

        _children := [ch],

        dimensions := ch.dimensions
    ))),

    rChildren := self >> [self._children[1], self.info],

    rSetChild := meth(self, n, what)
        if n=2 then self.info := what;
        elif n=1 then self._children[1] := what;
        else Error("<n> must be in [1..2]");
        fi;
    end,

    mergeInfo := (self, r1, r2) >> rec(
        opcount := r1.opcount + r2.opcount,
        free := Concat(r1.free, r2.free),
        loadFunc := r2.loadFunc,
        storeFunc := r1.storeFunc
    ),

    print := meth(self, indent, indentStep)
        local s,ch,first,newline;

        ch := [self.child(1),self.info];
        if self._short_print or ForAll(ch, x->IsSPLSym(x) or IsSPLMat(x)) then 
            newline := Ignore;
        else 
            newline := self._newline;
        fi;

        first:=true;
        Print(self.__name__, "(");
        for s in ch do
            if(first) then first:=false;
            else Print(", "); fi;
            newline(indent + indentStep);
            When(IsSPL(s) or (IsRec(s) and IsBound(s.print) and NumGenArgs(s.print)=2),
                s.print(indent + indentStep, indentStep), Print(s));
        od;
        newline(indent);
        Print(")");
        self.printA();
    end,
));

Class(DMAGath, Gath, rec(doNotMarkBB:= true));

Class(DMAScat, Scat, rec(doNotMarkBB:= true));

Class(DMAFence, Buf, rec(doNotMarkBB:= true));

Class(SWPSum, ISum, rec(doNotMarkBB := true));

Class(DMAGathV, VGath, rec(doNotMarkBB := true));
Class(DMAScatV, VScat, rec(doNotMarkBB := true));

