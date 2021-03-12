
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


memo := function(context, prefix, exp)
   local v, isum;
   exp := SReduce(toExpArg(exp), SpiralDefaults); ##Pass the real opts instead
   if IsValue(exp) then return exp.v;
   elif IsVar(exp) then return exp;
   else
       if IsBound(context.ISum) and context.ISum <> [] then
       isum := Last(context.ISum);
       v := var.fresh_t(prefix, TInt);
       if not IsBound(isum.memos) then isum.memos := []; fi;
       Add(isum.memos, [v, exp]);
       #Print(v.id , " -> ", exp, " [", isum.var, " in ", isum.domain, "]\n");
       v.mapping := exp;
       return v;
       else 
       return exp;
       fi;
   fi;
end;

nomemo := (cx,pfx,exp) -> exp;

memo := nomemo; # memo feature is currently broken -- disable it

Class(ProcessMemos, RuleSet);
RewriteRules(ProcessMemos, rec(
   # pull out cmemos
   PullOutCmemo := Rule(cmemo, 
     function(e,cx) 
         local target;
	 target := cx.(e.target);
         if not IsBound(target.memos) then 
	     target.memos := [ e.args[1], e.mapping ]; 
	 else Add(target.memos, [ e.args[1], e.mapping ]);
	 fi;
	 return e.args[1];
     end),

   # process memos at ISum
   ProcessISumMemos := Rule(@(1, [ISum, ISumAcc], e->IsBound(e.memos)), 
     e -> ObjId(@(1).val)(e.var, e.domain,
	      FoldL(Reversed(e.memos), 
		  (ex,m) -> Data(m[1], m[2], ex)), e.child(1)))
));
