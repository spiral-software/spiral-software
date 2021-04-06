
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################
# These functions work on formula level
########################################

MarkBlock := function(sums) sums.isBlock := true; return sums; end;

IsMarkedBlock := o -> IsRec(o) and IsBound(o.isBlock) and o.isBlock;

DoNotMark := s -> (IsBound(s.doNotMarkBB) and s.doNotMarkBB = true) or 
                  (IsIterative(s) and IsSymbolic(s.domain));

IsBlockTransitive := o -> IsBound(o.isBlockTransitive) and o.isBlockTransitive;

Compose.isBlockTransitive := true;
SUM.isBlockTransitive := true;
SUMAcc.isBlockTransitive := true;
Data.isBlockTransitive := true;

markBlocks := function(bk, sums, criterion_func)
    local doNotMark, c;
    if not IsSPL(sums) then return sums; fi;

    #Error(">>>>>>>------", ObjId(sums), "------>>>>>>>>>");
    doNotMark := false;

    for c in sums.rChildren() do
        markBlocks(bk, c, criterion_func);
        # if any of our children cannot be marked BB, we cannot be marked BB
        if IsSPL(c) and DoNotMark(c) then
          doNotMark := true;

          sums.doNotMarkBB := true; # To communicate this up to parent.
          # This is a slight HACK. Maybe we should mark these with a different
          # label (like sums.childIsNoMarkBB) and check for this, so we don't confuse
          # these with preset .doNotMarkBB

          #Error("Not marking: ", sums.__name__, " because of child: ", c.__name__);
        fi;
    od;

    if DoNotMark(sums) then
      doNotMark := true;
      #Error("Not marking ", sums.__name__, " because of doNotMarkBB");
    fi;

    #Error("<<<<<<<------", ObjId(sums), "------<<<<<<<<<");

    if not doNotMark then
        if IsBlockTransitive(ObjId(sums)) then
            if ForAll(sums.children(), IsMarkedBlock) then
               MarkBlock(sums);
            fi;
        elif criterion_func(sums, bk) then
            MarkBlock(sums);
        fi;
    fi;

    return sums;

end;

standard_criterion_func := (sums, bk) -> not IsBound(sums._perm) 
    and not (IsIterative(sums) and IsSymbolic(sums.domain))
    and not ForAll(Flat(sums.dims()), IsSymbolic) 
    and     ForAny(Flat(sums.dims()), d -> d<=bk);

MarkBlocksDimSums := (bk,sums) -> SubstBottomUp(sums, @, 
    e -> Cond(not IsSPL(e) or (IsBound(e.doNotMarkBB) and e.doNotMarkBB), e,
          not IsBound(e._perm) and not (IsIterative(e) and IsSymbolic(e.domain)) and
             (((not IsSymbolic(Cols(e))) and Cols(e)<=bk) or # finish this
              ((not IsSymbolic(Rows(e))) and Rows(e)<=bk)),    MarkBlock(e),
          IsBlockTransitive(ObjId(e)) and ForAll(e.children(), IsMarkedBlock), MarkBlock(e),
          e));

MarkBlocksAreaSums := (bk,sums) -> SubstBottomUpRules(sums, rec(
    blockAreaSums := Rule(@, 
        e -> Cond(not IsSPL(e) or not IsBound(e.area),    e,
              not IsSymbolic(e.area()) and e.area() <= bk,  MarkBlock(e),
              IsBlockTransitive(ObjId(e)) and ForAll(e.children(), IsMarkedBlock), MarkBlock(e),
              e))));

MarkBlocksOps := (bk,sums) -> SubstBottomUp(sums, @, 
    e -> Cond(not IsSPL(e), e,
          not IsBound(e._perm) and e.numops()<=bk,    MarkBlock(e),e));

##
#MarkBlocksSums := MarkBlocksDimSums;
MarkBlocksSums := (bk, sums) -> markBlocks(bk, sums, standard_criterion_func);
##

Class(PostProcessBBs, RuleSet);
RewriteRules(PostProcessBBs, rec(
    mergeBBsinsideCompose := ARule(Compose, [@(1, BB), @(2, BB)], e -> [BB(Compose(@(1).val.rChildren()[1], @(2).val.rChildren()[1]))]),
    kickOutNestedBBs := Rule( @@(1, BB, (x, cx) -> cx.isInside(BB)), e -> e.child(1)),
));

#F BlockSums(<bksize>, <sums>) 
#F     Uses MarkBlocksSums with given block size <bksize> to partition the 
#F     Sigma-SPL formula <sums> into unrolled blocks. 
#F
#F     MarkBlocksSums = MarkBlocksDimSums || MarkBlocksAreaSums.
#F     ..DimSums partitions by the smallest dimensions
#F     ..AreaSums partitions by the total dense area of a (potentially sparse) matrix
#F
BlockSums := function(bk, sums)
    sums := MarkBlocksSums(bk, sums);
    sums := SubstTopDownNR(sums, @.cond(IsMarkedBlock), e->When(ObjId(e)=BB, e, BB(e)));
    return sums;
end;

BlockSumsOpts := function(sums, opts)
    if IsBound(opts.markBlock) then
        sums := opts.markBlock(opts.globalUnrolling, sums);
    else
        sums := MarkBlocksSums(opts.globalUnrolling, sums);
    fi;
    sums := SubstTopDownNR(sums, @.cond(IsMarkedBlock), e->When(ObjId(e)=BB, e, BB(e)));
    return PostProcessBBs(sums);
end;

# Alternative implementation can use rules
#RuleSet(RulesBlockSums);
#Rule(@.cond(IsMarkedBlock), e -> BB(e));
#ARule(Compose, [..., @(1, BB), @(2, BB), ...], e -> BB(Compose(@(1).val, @(2).val));
#ARule(SUM,     [..., @(1, BB), @(2, BB), ...], e -> BB(SUM    (@(1).val, @(2).val)));
#ARule(SUMAcc,  [..., @(1, BB), @(2, BB), ...], e -> BB(SUMAcc (@(1).val, @(2).val)));

####################################
# These functions work on code level
####################################

#F MarkForUnrolling(<bbnum>, <code>, [<opts>])
MarkForUnrolling := (arg) -> Cond( 
    # if has options and no SimFlexKernelFlag then go without kern()
    Length(arg)=3 and not IsBound(arg[3].SimFlexKernelFlag), 
       unroll_cmd(arg[2]),
    # else
       kern(arg[1], unroll_cmd(arg[2])));

IsMarkedForUnrolling := code -> ObjId(code) = unroll_cmd; 

#F BlockCollect(<code>)
#F
#F Returns a list of blocks marked for unrolling (field doUnroll:=true must be set)
#F
BlockCollect := c -> Collect(c, @.cond(IsMarkedForUnrolling));

Declare(_BlockUnroll);

LAST_BLOCKUNROLL := 0;
LAST_BLOCKUNROLLOPTS := 0;
#F BlockUnroll(<code>, <opts>)
#F
#F Generates unrolled code for marked blocks (field doUnroll:=true must be set)
#F
BlockUnroll := function(c, opts)
    LAST_BLOCKUNROLL := Copy(c);
    LAST_BLOCKUNROLLOPTS := opts;
    return _BlockUnroll(c, opts, 1);
end;

_BlockUnroll := function(c, opts, ldepth)
    local i, last, ch, newch, tocompile;

    if IsMarkedForUnrolling(c) then return Compile(c.cmds[1], opts); # strip unroll_cmd
    elif not IsCommand(c) then return c;
    else
    # a chain with some subcommands to be unrolled
    if ObjId(c) = chain and ForAny(c.cmds, IsMarkedForUnrolling) then
        newch := [];
        i := 1;
        # group each sequence of marked subcommands and unroll together
        # to facilitate copy propagation in-between 
        while i <= Length(c.cmds) do
            ch := c.cmds[i];
           # recurse on blocks that aren't marked 
            if not IsMarkedForUnrolling(ch) then 
                Add(newch, _BlockUnroll(ch, opts, ldepth));
                # otherwise find maximal sequence of marked blocks and unroll them together
            else 
                last := i;
                tocompile := [ch.cmds[1]];
                while last < Length(c.cmds) and IsMarkedForUnrolling(c.cmds[last+1]) do
                    last := last+1; 
                    ch := c.cmds[last];
                    Add(tocompile, ch.cmds[1]); # strip unroll_cmd
                od;
                Add(newch, Compile(chain(tocompile), opts));
                i := last;
            fi;
            i := i+1;
        od;
        return chain(newch);
    # for an unmarked command (that is not a chain) simply recurse on subcommands
    else
        # mark loop variables with their loop nesting depth
        if ObjId(c) in [loop, loopn] then 
            c.var.ldepth := ldepth; 
            ldepth := ldepth + 1;
        fi;

        i := 1;
        for ch in c.rChildren() do
            c.rSetChild(i, _BlockUnroll(ch, opts, ldepth));
            i := i+1;
        od;
        return c;
    fi;
    fi;
end;

# enumerate the blocks. useful for when you want to turn blocks on and off at
# the code level.

EnumBlocks := function(sums, opts)
    local blocks, numbered, notnumb, max, i;
    blocks := Collect(sums, BB);
    
    numbered := Filtered(blocks, e -> IsBound(e.bbnum));
    notnumb := Filtered(blocks, e -> not IsBound(e.bbnum));

    if numbered <> [] then
        max := Maximum(List(numbered, e -> e.bbnum));
    else
        max := 0;
    fi;

    for i in notnumb do
        max := max + 1;
        i.bbnum := max;
    od;

    return sums;
end;
