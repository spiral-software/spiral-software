
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F threadId() 
#F 
Class(threadId, Exp, rec(
    computeType := self >> TInt
));

#F barrier(<nthreads>, <tid>, <barrier-name>)
#F
#F Example of mapping from SMPBarrier object <o>: 
#F    barrier(o.nthreads, o.tid, "&GLOBAL_BARRIER")
#F
Class(barrier, call, rec(
    visitAs := call
));

#F smp_fork(<nthreads>, <cmd>)
#F NOTE: this should take in thread id 'tid' for consistency, but it doesn't!
#F
Class(smp_fork, Command, rec(
    rChildren := self >> [self.nthreads, self.cmd],
    rSetChild := rSetChildFields("nthreads", "cmd"),

    __call__ := (self, nthreads, cmd) >> WithBases(self,
       rec(operations := CmdOps,
           nthreads   := Checked(IsInt(nthreads) or IsScalar(nthreads), nthreads),
           cmd        := Checked(IsCommand(cmd), cmd))),

    print := (self,i,is) >> Print(
         self.__name__, "(", self.nthreads, ",\n",
         Blanks(i+is), self.cmd.print(i+is, is), "\n",
         Blanks(i), ")"
    )
));

#F smp_loop(<nthreads>, <tidvar>, <tidexp>, <loopvar>, <range>, <cmd>)
#F
#F nthreads - # of threads
#F tidvar - thread id variable (will be set to tid value), necessary for nested parallelism
#F tidexp - thread id value 
#F
Class(smp_loop, loop_base, rec(
   __call__ := meth(self, nthreads, tidvar, tidexp, loopvar, range, cmd) 
       local result;
       Constraint(IsVar(loopvar)); 
       Constraint(IsCommand(cmd)); 
       range := toRange(range);
       if range = 0 then
           return skip();      
       else 
           loopvar.setRange(range);
           loopvar.isLoopIndex := true;
           return WithBases(self, rec(
                   operations := CmdOps, 
                   nthreads := nthreads, 
                   cmd := cmd, 
                   var := loopvar, 
                   tidvar := Checked(IsLoc(tidvar), tidvar),
                   tidexp := toExpArg(tidexp),
                   range := range));
       fi;
   end,

   rChildren := self >> [self.nthreads, self.tidvar, self.tidexp, self.var, self.range, self.cmd],
   rSetChild := rSetChildFields("nthreads", "tidvar", "tidexp", "var", "range", "cmd"),

   print := (self, i, is) >> Print(self.name, "(", self.nthreads, ", ", 
       self.tidvar, ", ", self.tidexp, ", ", self.var, ", ", 
       self.range, ",\n", Blanks(i+is),
       self.cmd.print(i+is, is),
       Print("\n", Blanks(i), ")")),

   DContainer := (self, o, y, x, opts) >> self(o.child(1), y, x, opts),

   free := self >> Difference(self.cmd.free(), [self.var, self.tidvar])
));

# Class(smp_chain, chain, rec(
#    __call__ := meth(arg)
#        local self, nthreads, cmds;
#        [self, nthreads, cmds] := [arg[1], arg[2], Flat(Drop(arg, 2))];
#        return WithBases(self, rec(
#                nthreads   := nthreads,
#                operations := CmdOps,
#                cmds       := Checked(ForAll(cmds, IsCommand), cmds)));
#    end,

#    print := (self,i,is) >> When(Length(self.cmds)=0,
#        Print(self.name, "(", self.nthreads, ")"),
#        Print(self.name, "(", self.nthreads, ",\n", self.printCmds(i+is, is), Blanks(i), ")"))
# ));

Class(SMPCodegenMixin, Codegen, rec(
    SMPBarrier := (self, o, y, x, opts) >> chain(
        self(o.child(1), y, x, opts), 
        barrier(o.nthreads, o.tid, "&GLOBAL_BARRIER")),

    SMPSum := (self, o, y, x, opts) >> let(
        outer_tid     := When(IsBound(opts._tid), opts._tid, 0),
        outer_num_thr := When(IsBound(opts._tid), opts._tid.range, 1),
        tid := var.fresh("tid", TInt, o.nthreads * outer_num_thr),
        smp_loop(o.nthreads, tid, (outer_tid * outer_num_thr) + o.tid,
                 o.var, o.domain, 
                 self(o.child(1), y, x, CopyFields(opts, rec(_tid := tid))))
    )
));


_PullBuffersSMP := function(expr, type_predicate, nthreads, tid)
    local ch, i, t, data, pullv, stayv;
    if ObjId(expr)=assign then return [[], expr];
    else
        # Implemented as a recursive tree walk
	data := [];

        if ObjId(expr)=decl then
            [pullv, stayv] := SplitBy(expr.vars, x->type_predicate(x.t));
            data := List(pullv, x->[x, nthreads, tid]);
            expr := decl(stayv, expr.cmd);
        elif ObjId(expr)=smp_loop then 
            nthreads := nthreads * expr.nthreads; 
            tid := expr.tidvar;
        fi;

        ch := _children(expr);
        for i in [1..Length(ch)] do
            t := _PullBuffersSMP(ch[i], type_predicate, nthreads, tid);
            Append(data, t[1]);
            _setChild(expr, i, t[2]);
        od;
        return [data, expr];
    fi;
end;

#F PullBuffersSMP(code, type_predicate)
#F
#F  Pulls out declarations (from decl(..)) that satisfy type_predicate,
#F  and return declared variables as triplets [var, nthreads, tid]
#F
#F  This function handled nested parallelism, thats why the implementation is
#F  not so straightforward. 
#F
PullBuffersSMP := (code, type_predicate) -> _PullBuffersSMP(code, type_predicate, 1, 0);


 
