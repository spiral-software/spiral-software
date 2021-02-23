
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(JoinDirectSums, RuleSet);
RewriteRules(JoinDirectSums, rec(
    combine := ARule(Compose, [@(1, DelayedDirectSum), @(2, DelayedDirectSum,
        e -> let(c1:=@(1).val.children(), c2:=e.children(), Length(c1) = Length(c2) and ForAll([1..Length(c1)], i->c1[i].dims()[2] = c2[i].dims()[1]))) ],
        e -> let(c1:=@(1).val.children(), c2:=@(2).val.children(), [ DelayedDirectSum(List([1..Length(c1)], i -> c1[i] * c2[i])) ])
    )
));

Class(TerminateDirectSums, RuleSet);
RewriteRules(TerminateDirectSums, rec(
    terminate := Rule(@(1, DelayedDirectSum), e-> DirectSum(@(1).val.children()).sums())
));
