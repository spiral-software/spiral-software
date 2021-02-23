
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


CompareCodeMat := (sums, opts) ->
    let(me := Try(CMatrix(CodeSums(Copy(sums), opts), opts)),
        them := MatSPL(sums),
        When(me[1] = false, false,
             InfinityNormMat(me[2]-them) < 1e-5));


_RecursiveFindBug := function(ind, spl, opts, verify_func)
    local c, ch, res, bugspl, origspl, seg, i, len;
    Print(Blanks(ind), ObjId(spl), " - ");
    res := verify_func(spl, opts);
    Print(res, "\n");

    # no bug, verification succeeded
    if res = true then return true; 

    # bug, verification failed, now we will recurse and verify the children
    else
        origspl := spl;
        # loops need to be unrolled to get rid of index variables
        if IsBound(spl.unroll) then spl := spl.unroll(); fi;
        for c in spl.children() do
            bugspl := _RecursiveFindBug(ind+3, c, opts, verify_func);
            if bugspl <> true then return bugspl; fi; # bug is in the child
        od;

        # verification succeeded for all of the children, but we will still try to 
        # narrow down the problem if the current node is Compose
        if ObjId(spl)=Compose and spl.numChildren() > 2 then
            ind := ind+3;
            PrintLine(Blanks(ind), "Problematic multifactor Compose. Narrowing down the problem");
            ch := spl.children();
            while Length(ch) > 2 do
                len := Length(ch)-1;
                i:=1;
                res := true;
                # the loop terminates when res = false (verif failed) OR everything succeeded (ie no smaller bug)
                while res and i <= Length(ch)-len+1 do
                   seg := ch{[i..i+len-1]}; 
                   res := verify_func(Compose(seg), opts);
                   PrintLine(Blanks(ind), "len=", len, " i=", i, " ", res);
                   i := i+1;
                od;
                if res=true then return Compose(ch); 
                else ch := seg;
                fi;
            od;
            return Compose(ch);
        else
            return origspl; # bug is here and not in the children, and we could not narrow it down further
        fi;
    fi;
end;

# verify_func(spl, opts)
#
RecursiveFindBug := (spl, opts, verify_func) -> _RecursiveFindBug(0, spl, opts, verify_func);



_RecursiveFindBugExp := function(ind, spl, opts, verify_func)
    local c, ch, res, bugspl, origspl, seg, i, len;
    Print(Blanks(ind), ObjId(spl), " - ");
    res := verify_func(spl, opts);
    Print(res, "\n");

    # no bug, verification succeeded
    if res = true then return true; 

    # bug, verification failed, now we will recurse and verify the children
    else
        origspl := spl;
        # loops need to be unrolled to get rid of index variables
        if IsBound(spl.unroll) then spl := spl.unroll(); fi;
        for c in spiral.rewrite._children(spl) do
            bugspl := _RecursiveFindBugExp(ind+3, c, opts, verify_func);
            if bugspl <> true then return bugspl; fi; # bug is in the child
        od;

        return origspl; # bug is here and not in the children
    fi;
end;

RecursiveFindBugExp := (spl, opts, verify_func) -> _RecursiveFindBugExp(0, spl, opts, verify_func);



RecursiveFindBugRT := function (rt, opts)
    local chk, s;

    chk := (spl, opts) -> InfinityNormMat(CMatrix(CodeSumsOpts(spl, opts), opts) - MatSPL(spl)) < 1e-4;
    s := SumsRuleTreeOpts(rt, opts);

    return RecursiveFindBug(s, opts, chk);
end;
