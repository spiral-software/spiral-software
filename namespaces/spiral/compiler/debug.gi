
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details


# ------------------------------------------------------------------------------
# DEBUGGING
# ------------------------------------------------------------------------------
<# commented out as broken long ago
dim_bug := zz -> 
    Collect(zz, @.cond(e->IsRec(e) and IsBound(e.dimensions) and IsBound(e.dims) 
    and Try(e.dims())[1]=true and e.dimensions <> e.dims()));

checkrules := function(sums, nrules, verif_mat, ruleset)
     local sums, norm, verif_mat, dimbugs, cx;
     cx := limit_cx(1);
     cx.applied := 1;

     while nrules > 0 and cx.applied > 0 do
         cx.applied := 0;
	 cx.rlimit := 1;

         DebugRewriting(true);
	 sums := _ApplyAllRulesTopDown(sums, cx, ruleset);
         DebugRewriting(false);

	 norm := inf_norm(MatSPL(sums) - verif_mat );
	 Print(norm, "\n");
	 if norm > 0.001 then return [sums, "BROKEN"]; fi;

         #dimbugs := dim_bug(sums);
         #if dimbugs <> [] then return [sums, "DIM BUG"]; fi;
         nrules := nrules - 1;
      od;
      return [sums, "OK"];
end;

_Bug := function(ind, spl)
    local c, res, bugspl;
    spl := toSPL(spl);
    Print(Blanks(ind), ObjId(spl), " - ");
    res := Try(Compile(spl));
    Print(res[1], "\n");

    if res[1] = true then return true; # no bug
    else
	for c in spl.children() do
	   bugspl := _Bug(ind+3, c);
	   if bugspl <> true then return bugspl; fi; # bug is in the child
	od;
	return spl; # bug is here and not in the children
    fi;
end;

Bug := spl -> _Bug(0,spl);

WipeRoots := sums -> DoForAll(Collect(sums, @),
    function(s)
        if IsRec(s) then Unbind(s.root); fi;
	return s;
    end
);
#>
DebugCompile := function(switch)
    Constraint(IsBool(switch));
    if switch then
        Compile.timingStatus := (i, stage, t) -> Print(i, ": ", stage, "\ntime: ", t, "\n");
    else
        Compile.timingStatus := Ignore;
    fi;
end;

