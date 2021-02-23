
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# idea : Restrict by dims(@(2)) <= codelet size
#
RewriteRules(RulesSums, rec(
    CommutePrmScatInplace := ARule(Compose, [@(1, [Prm,Scat]), [Inplace, @(2)]],
	e -> [ Inplace(@(1).val * @(2).val * @(1).val.transpose()), Copy(@(1).val) ]),

    CommuteInplaceGath := ARule(Compose, [[Inplace, @(2)], @(1, Gath)],
	e -> [ Copy(@(1).val), Inplace(@(1).val.transpose() * @(2).val * @(1).val) ])
));



Class(RulesInplace, RuleSet);
RewriteRules(RulesInplace, rec(
    CommutePrmScatInplace := RulesSums.rules.CommutePrmScatInplace,
    CommuteInplaceGath := RulesSums.rules.CommuteInplaceGath,

    InplaceSUM := Rule([Inplace, @(1,SUM)], e -> SUM(List(@(1).val.children(), Inplace)))
));
