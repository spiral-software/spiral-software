
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(RulesSMP, RuleSet);

RewriteRules(RulesSMP, rec(
    PSD_SMPSumRight := ARule(Compose,  [ @(1, [RCDiag, Diag, Prm, Scat, FormatPrm]), @(2, SMPSum) ],
     e -> let(s:=@(2).val, [ SMPSum(s.nthreads, s.tid, s.var, s.domain, @(1).val * s.child(1)) ])),

    PGD_SMPSumLeft  := ARule(Compose, [ @(1, SMPSum), @(2, [Prm, Gath, Diag, RCDiag, FormatPrm]) ],
     e -> let(s:=@(1).val, [ SMPSum(s.nthreads, s.tid, s.var, s.domain, s.child(1) * @(2).val) ])),


    PSD_SMPBarrierRight := ARule(Compose,  [ @(1, [RCDiag, Diag, Prm, Scat, FormatPrm]), @(2, SMPBarrier) ],
     e -> let(s:=@(2).val, [ SMPBarrier(s.nthreads, s.tid, @(1).val * s.child(1)) ])),

    PGD_SMPBarrierLeft  := ARule(Compose, [ @(1, SMPBarrier), @(2, [Prm, Gath, Diag, RCDiag, FormatPrm]) ],
     e -> let(s:=@(1).val, [ SMPBarrier(s.nthreads, s.tid, s.child(1) * @(2).val) ])),

    Drop_GrpSPSum := Rule([Grp, @(1, SMPSum)], e -> @(1).val),
    Drop_GrpISum := Rule([Grp, @(1, ISum)], e -> @(1).val),

    SMP_ISum := Rule([SMP, @(1), @(2), @(3, ISum)], e ->
        SMPSum(@(1).val, @(2).val, @(3).val.var, @(3).val.domain, @(3).val.child(1))),
));

RewriteRules(RulesRC, rec(
    RC_SMPSum := Rule([RC, @(1, SMPSum)],
        e -> let(s:=@(1).val, SMPSum(s.nthreads, s.tid, s.var, s.domain, RC(s.child(1))))),

    RC_SMPBarrier := Rule([RC, @(1, SMPBarrier)],
        e -> let(s:=@(1).val, SMPBarrier(s.nthreads, s.tid, RC(s.child(1))))),
));
