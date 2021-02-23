
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



_macro_binop_ev := rec(
    ev := self >> self.t.value(self.semantic(self._vval(1), self._vval(2), [])).ev(),
);

################################
#         MACRO_2xf
#

Class(vunpacklo_2xf, _macro_binop_ev, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 2, 1)));

Class(vunpackhi_2xf, _macro_binop_ev, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 2, 1)));

Class(vpacklo_2xf, _macro_binop_ev, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [1, 1], 2, 1)));

Class(vpackhi_2xf, _macro_binop_ev, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [2, 2], 2, 1)));

Class(vmulcx_2xf, VecExp_2.binary());
Class(vswapcx_2xf, VecExp_2.unary());

################################
#         MACRO_4xf
#

Class(vunpacklo_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 4, 1)));

Class(vunpackhi_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 4, 1)));

Class(vunpacklo2_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 4, 2)));

Class(vunpackhi2_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 4, 2)));

Class(vpacklo_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [1, 3, 1, 3], 4, 1)));

Class(vpackhi_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [2, 4, 2, 4], 4, 1)));

Class(vpacklo2_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [1, 1], 4, 2)));

Class(vpackhi2_4xf, _macro_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [2, 2], 4, 2)));

Class(vmulcx_4xf, VecExp_4.binary());
Class(vswapcx_4xf, VecExp_4.unary());

################################
#         MACRO_8xf
#

Class(vunpacklo_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 8, 1)));

Class(vunpackhi_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 8, 1)));

Class(vunpacklo2_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 8, 2)));

Class(vunpackhi2_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 8, 2)));

Class(vpacklo_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [1, 3, 5, 7, 1, 3, 5, 7], 8, 1)));

Class(vpackhi_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [2, 4, 6, 8, 2, 4, 6, 8], 8, 1)));

Class(vpacklo2_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [1, 3, 1, 3], 8, 2)));

Class(vpackhi2_8xf, _macro_binop_ev, VecExp_8.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [2, 4, 2, 4], 8, 2)));

Class(vtrcx_8xf, VecExp_8.unary(), rec(
    semantic := (in1, p) -> shuffle(in1, in1, [1, 2, 5, 6, 3, 4, 7, 8], 8, 1),
    ev := self >> self.t.value(self.semantic(self._vval(1), [])).ev()
));

Class(vmulcx_8xf, VecExp_8.binary());
Class(vswapcx_8xf, VecExp_8.unary());

#################################
# Three Op version
#
ThreeOpFromBinOp(
    Class(vunpacklo_2xf_cmd,  assign_cmd, rec( exp_op := vunpacklo_2xf )),
    Class(vunpackhi_2xf_cmd,  assign_cmd, rec( exp_op := vunpackhi_2xf )),
    Class(vpacklo_2xf_cmd,    assign_cmd, rec( exp_op := vpacklo_2xf   )),
    Class(vpackhi_2xf_cmd,    assign_cmd, rec( exp_op := vpackhi_2xf   )),
    Class(vmulcx_2xf_cmd,     assign_cmd, rec( exp_op := vmulcx_2xf    )),

    Class(vunpacklo_4xf_cmd,  assign_cmd, rec( exp_op := vunpacklo_4xf  )),
    Class(vunpackhi_4xf_cmd,  assign_cmd, rec( exp_op := vunpackhi_4xf  )),
    Class(vunpacklo2_4xf_cmd, assign_cmd, rec( exp_op := vunpacklo2_4xf )),
    Class(vunpackhi2_4xf_cmd, assign_cmd, rec( exp_op := vunpackhi2_4xf )),
    Class(vpacklo_4xf_cmd,    assign_cmd, rec( exp_op := vpacklo_4xf    )),
    Class(vpackhi_4xf_cmd,    assign_cmd, rec( exp_op := vpackhi_4xf    )),
    Class(vpacklo2_4xf_cmd,   assign_cmd, rec( exp_op := vpacklo2_4xf   )),
    Class(vpackhi2_4xf_cmd,   assign_cmd, rec( exp_op := vpackhi2_4xf   )),
    Class(vmulcx_4xf_cmd,     assign_cmd, rec( exp_op := vmulcx_4xf     )),

    Class(vunpacklo_8xf_cmd,  assign_cmd, rec( exp_op := vunpacklo_8xf  )),
    Class(vunpackhi_8xf_cmd,  assign_cmd, rec( exp_op := vunpackhi_8xf  )),
    Class(vunpacklo2_8xf_cmd, assign_cmd, rec( exp_op := vunpacklo2_8xf )),
    Class(vunpackhi2_8xf_cmd, assign_cmd, rec( exp_op := vunpackhi2_8xf )),
    Class(vpacklo_8xf_cmd,    assign_cmd, rec( exp_op := vpacklo_8xf    )),
    Class(vpackhi_8xf_cmd,    assign_cmd, rec( exp_op := vpackhi_8xf    )),
    Class(vpacklo2_8xf_cmd,   assign_cmd, rec( exp_op := vpacklo2_8xf   )),
    Class(vpackhi2_8xf_cmd,   assign_cmd, rec( exp_op := vpackhi2_8xf   )),
    Class(vmulcx_8xf_cmd,     assign_cmd, rec( exp_op := vmulcx_8xf     ))
);

Class(vswapcx_4xf_cmd,    assign_cmd, rec( exp_op := vswapcx_4xf    ));
Class(vswapcx_2xf_cmd,    assign_cmd, rec( exp_op := vswapcx_2xf    ));
Class(vswapcx_8xf_cmd,    assign_cmd, rec( exp_op := vswapcx_8xf    ));

Class(vtrcx_8xf_cmd, assign_cmd, rec(exp_op := vtrcx_8xf));

Class(vstore_cmd, assign_cmd);
Class(vload_cmd,  assign_cmd);

RewriteRules(ThreeOpRuleSet, rec(
    assign_to_store := Rule([assign, @(1, nth), @(2)], e -> vstore_cmd(@(1).val.loc, @(1).val.idx, @(2).val)),
    assign_to_load  := Rule([assign, @(1, var), @(2, nth)], e -> vload_cmd(@(1).val, @(2).val.loc, @(2).val.idx)),
));


