
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(CScratchUnparserProg, CUnparser, rec(
    dma_signal := (self,o,i,is) >> Print(Blanks(i), self.opts.dmaSignal(self.opts), self.pinfix(o.args, ", "), ";\n"),
    dma_wait := (self,o,i,is) >> Print(Blanks(i), self.opts.dmaWait(self.opts), "", self.pinfix(o.args, ", "), ";\n"),

    cpu_signal := (self,o,i,is) >> Print(Blanks(i), self.opts.cpuSignal(self.opts), self.pinfix(o.args, ", "), ";\n"),
    cpu_wait := (self,o,i,is) >> Print(Blanks(i), self.opts.cpuWait(self.opts), "", self.pinfix(o.args, ", "), ";\n"),

    dma_fence := (self,o,i,is) >> Print(Blanks(i), self.opts.dmaFence(self.opts), "();\n"),

    dma_load := (self,o,i,is) >> Print(Blanks(i), self.opts.dmaLoad(self.opts),
        "(", self(o.loc,i,is), ", ", self(o.exp,i,is), ", ", self(o.size,i,is), ");\n"),

    dma_store := (self,o,i,is) >> Print(Blanks(i), self.opts.dmaStore(self.opts),
        "(", self(o.loc,i,is), ", ", self(o.exp,i,is), ", ", self(o.size,i,is), ");\n"),

    par_exec := (self,o,i,is) >> Print(Blanks(i), "parallel {\n", DoForAll(o.cmds, c -> self(c, i+is, is)), Blanks(i), "}\n"),

    decl := meth(self,o,i,is)
        local arrays, memarrays, scratcharrays, romarrays, other, l, arri, myMem;
        [arrays, other] := SplitBy(o.vars, x->IsArray(x.t));
        [memarrays, arrays] := SplitBy(arrays, i->IsBound(i.t.qualifiers) and self.opts.memModifier in i.t.qualifiers);
        [scratcharrays, arrays] := SplitBy(arrays, i->IsBound(i.t.qualifiers) and self.opts.scratchModifier in i.t.qualifiers);
        [romarrays, arrays] := SplitBy(arrays, i->IsBound(i.t.qualifiers) and self.opts.romModifier in i.t.qualifiers);

        if Length(arrays) > 0 then
            DoForAll(arrays, v -> Print(Blanks(i), self.opts.arrayBufModifier, " ", self.declare(v.t, v, i, is), ";\n"));
        fi;
        if Length(memarrays) > 0 then
            DoForAll(memarrays, v -> Print(Blanks(i), self.opts.arrayBufModifier, " ", self.opts.memModifier, " ", self.declare(v.t, v, i, is), ";\n"));
        fi;
        if Length(scratcharrays) > 0 then
            DoForAll(scratcharrays, v -> Print(Blanks(i), self.opts.arrayBufModifier, " ", self.opts.scratchModifier, " ", self.declare(v.t, v, i, is),";\n"));
        fi;
        if Length(romarrays) > 0 then
            DoForAll(romarrays, v -> Print(Blanks(i), self.opts.arrayBufModifier, " ", self.opts.romModifier, " ", self.declare(v.t, v, i, is), ";\n"));
        fi;

        if (Length(other)>0) then
            other:=SortRecordList(other,x->x.t);
            for l in other do
               Sort(l, (a,b)->a.id < b.id);
               Print(Blanks(i), self.declare(l[1].t, l, i, is), ";\n");
            od;
        fi;

        self(o.cmd, i, is);

        #Pop arena for this decl
        if IsBound(self.opts.useMemoryArena) and self.opts.useMemoryArena and Length(arrays) > 0 and arrays[1].id[1] <> 'D' then
          myMem := 0;
          for arri in arrays do
             # Account for vector allocations in memory arena (which is scalar)
             myMem := myMem + (arri.t.size * When(IsBound(arri.t.t) and ObjId(arri.t.t)=TVect, arri.t.t.size, 1));
          od;
          if ObjId(myMem) = Value then myMem := myMem.v; fi;
          Print(Blanks(i));
          Print("arenalevel += ", myMem, ";\n" );
        fi;
    end,

    swp_loop := (self, o, i, is) >> let(v := o.var, lo := o.range[1], hi := Last(o.range),
        Print(When(IsBound(self.opts.looppragma), self.opts.looppragma(o,i,is)),
          Blanks(i), "for(int ", v, " = ", lo, "; ", v, " <= ", hi, "; ", v, "++) { // SWP loop\n",
          self(o.cmd,i+is,is),
          Blanks(i), "}\n")),
));
