
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

RewriteRules(RulesStrengthReduce, rec(
    re_vpack := Rule([re, vpack], x -> ApplyFunc(vpack, List(x.args[1].args, re))),
    im_vpack := Rule([im, vpack], x -> ApplyFunc(vpack, List(x.args[1].args, im))),

    velem_vpack := Rule([velem, @(1, vpack), @(2, Value)], x -> @(1).val.args[1+@(2).val.v]),

    velem_vdup := Rule([velem, @(1, vdup), @(2)], x -> @(1).val.args[1]), 

    fStretch_Drop := Rule( @(1, fStretch, x -> x.num = x.den), e -> e.func),
));
