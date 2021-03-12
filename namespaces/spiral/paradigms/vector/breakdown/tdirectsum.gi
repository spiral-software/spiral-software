
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


########################################################################
#   (A + B) rules
NewRulesFor(TDirectSum, rec(
#   (A + B) terminate
    A_dirsum_B_delayed := rec(
        forTransposition := false,
        children := t -> let(tags := t.getTags(), 
            [[ t.params[1].withTags(tags), t.params[2].withTags(tags) ]]
        ),
#D        children := (self, t) >> let (tags:=GetTags(t),
#D            [[ AddTag(t.params[1], tags), AddTag(t.params[2], tags) ]]),
        apply := (t, C, Nonterms) -> DelayedDirectSum(C)
    )
));
