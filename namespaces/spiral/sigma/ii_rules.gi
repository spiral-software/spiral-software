
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# ================================================================
# Simplification of Intervals
# ================================================================

full_II := (int, size) -> 
    [int.target(II), size, 0, @.cond(e->e=size.val)];

empty_II := (int, size) -> 
    [int.target(II).cond(e->e.params[2]=e.params[3]), size, @, @];

left_II := (int, size, endd) -> 
    [int.target(II), size, 0, endd];

right_II := (int, size,start) -> 
    [int.target(II), size, start, @.cond(e->e=size.val)];

Class(RulesII, RuleSet);

RewriteRules(RulesII, rec(
    # full II o index mapping func @(3) = II with domain of @(3)
    # II(n, 0, n) o f = II(domain(f), 0, domain(f))
    FullII_f := ARule(fCompose, [ full_II(@, @(1)), @(2) ],  e -> [ II(@(2).val.domain()) ]),
    EmptyII_f := ARule(fCompose, [ empty_II(@, @(1)), @(2) ],  e -> [ II(@(2).val.domain(),0,0) ]),

    # partial II o dirsum of perms
    # II(n, 0, k) o fDirsum(perm_k, perm_?) -> II(n,0,k)
    LeftII_fDirsum := ARule(fCompose, 
        [ left_II(@(0), @(1), @(2)), [fDirsum, @(3).cond(e -> is_perm(e) and e.range()=@(2).val), ...]],
        e -> [ @(0).val ]),

    # partial II o dirsum of perms
    # II(n, n-k, n) o fDirsum(perm_?, perm_k) -> II(n,n-k,n)
    RightII_fDirsum := ARule( fCompose, 
        [ right_II(@(0), @(1), @(2)), [fDirsum, ..., @(4).cond(e -> is_perm(e) and e.range() = @(1).val-@(2).val)]],
        e -> [ @(0).val ]),

    # fTensor(IIn, IIk) -> II(n*k)
    fullII_tensor_fullII := ARule(diagTensor, [full_II(@(0), @(1)), full_II(@(0), @(2))], 
        e -> [ II(@(1).val * @(2).val) ]),

    # fDirsum(IIn, IIk) -> II(n+k)
    fullII_dirsum_fullII := ARule(diagDirsum, [full_II(@(0), @(1)), full_II(@(0), @(2))], 
        e -> [ II(@(1).val + @(2).val) ]),

    diagTensor_emptyII := Rule([diagTensor, ..., empty_II(@(1), @(2)), ...], e -> II(e.domain(),0,0)),
    diagTensor_fullII_Id1 := ARule(diagTensor, [@(0), full_II(@(1), @(2).cond(e->e=1))], e -> [ @(0).val ]),
    diagTensor_fullII_Id2 := ARule(diagTensor, [full_II(@(1), @(2).cond(e->e=1)), @(0)], e -> [ @(0).val ]),


    II_dirsum_emptyII := ARule(diagDirsum, [[II, @(1), @(2), @(3)], empty_II(@(4), @(5))], 
        e -> [ II(@(1).val + @(5).val), @(2).val, @(3).val ]),

    emptyII_dirsum_II := ARule(diagDirsum, [empty_II(@(4), @(5)), [II, @(1), @(2), @(3)]], 
        e -> [ II(@(1).val + @(5).val), @(5).val + @(2).val, @(5).val + @(3).val ]),

    # II Shifting, we support shift left (shift <= start), and right (shift >= end)
    # II o Z = shifted II# 
    II_Z := ARule(fCompose, [[II, @(1), @(2), @(3)], [Z, @(4), @(5).cond(e -> (e<=@(2).val) or (e >= @(3).val))]],
        e -> [II(@(1).val, (@(2).val-@(5).val) mod @(1).val, 
               # this makes 0->size, because size mod size = 0
                let(last:=(@(3).val-@(5).val) mod @(1).val,  
                    When(last=0, @(1).val, last))) ]),

    DiagFullII_toId := Rule([Diag, full_II(@(1), @(2))], e -> I(@(2).val)),
    Diag_fConst_toId := Rule([Diag, [fConst, @(2), @(1), 1]], e -> I(@(1).val)),
    Diag_fConstV_toId := Rule([Diag, [fConst, @(2), @(1), _1]], e -> I(@(1).val)),

    COND_fullII := Rule([COND, full_II(@(1), @(2)), @(3), @(4)], e -> @(3).val),
    COND_emptyII := Rule([COND, full_II(@(1), @(2)), @(3), @(4)], e -> @(4).val)
));
