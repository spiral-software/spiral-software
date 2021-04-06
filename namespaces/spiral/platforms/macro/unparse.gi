
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



Class(MACROUnparser, ThreeOpMacroUnparser_Mixin, CMacroUnparserProg, rec(

    data   := (self, o, i, is) >> When(IsVecT(o.var.t), 
                                     Print(Blanks(i), self.prefixT("VDECLC", o.var.t, [o.var] :: o.value.v), ";\n", self(o.cmd, i, is)),
                                     Inherited(o, i, is)),

    vpack  := (self, o, i, is) >> self.prefixT("VPK",  o.t, o.args),
    vdup   := (self, o, i, is) >> self.prefixT("VDUP", o.t, o.args[1]),

    assign_cmd := (self, o, i, is) >> Print(Blanks(i), self.(o.exp_op.__name__)(o, i, is), ";\n"),

    vunpacklo_2xf  := (self, o, i, is) >> self.prefix_T("VUNPKLO", o.args),
    vunpacklo_4xf  := ~.vunpacklo_2xf,
    vunpacklo_8xf  := ~.vunpacklo_2xf,

    vunpackhi_2xf  := (self, o, i, is) >> self.prefix_T("VUNPKHI", o.args),
    vunpackhi_4xf  := ~.vunpackhi_2xf,
    vunpackhi_8xf  := ~.vunpackhi_2xf,

    vunpacklo2_4xf := (self, o, i, is) >> self.prefix_T("VUNPKLO2", o.args),
    vunpacklo2_8xf := ~.vunpacklo2_4xf,

    vunpackhi_2xf  := (self, o, i, is) >> self.prefix_T("VUNPKHI", o.args),
    vunpackhi_4xf  := ~.vunpackhi_2xf,
    vunpackhi_8xf  := ~.vunpackhi_2xf,

    vunpackhi2_4xf := (self, o, i, is) >> self.prefix_T("VUNPKHI2", o.args),
    vunpackhi2_8xf := ~.vunpackhi2_4xf,

    vpacklo_2xf  := (self, o, i, is) >> self.prefix_T("VPKLO", o.args),
    vpacklo_4xf  := ~.vpacklo_2xf,
    vpacklo_8xf  := ~.vpacklo_2xf,

    vpackhi_2xf  := (self, o, i, is) >> self.prefix_T("VPKHI", o.args),
    vpackhi_4xf  := ~.vpackhi_2xf,
    vpackhi_8xf  := ~.vpackhi_2xf,

    vpacklo2_4xf := (self, o, i, is) >> self.prefix_T("VPKLO2", o.args),
    vpacklo2_8xf := ~.vpacklo2_4xf,

    vpackhi_2xf  := (self, o, i, is) >> self.prefix_T("VPKHI", o.args),
    vpackhi_4xf  := ~.vpackhi_2xf,
    vpackhi_8xf  := ~.vpackhi_2xf,

    vpackhi2_4xf := (self, o, i, is) >> self.prefix_T("VPKHI2", o.args),
    vpackhi2_8xf := ~.vpackhi2_4xf,

    vtrcx_8xf    := (self, o, i, is) >> self.prefix_T("VTRCX", o.args),

    vmulcx_2xf := (self, o, i, is) >> self.prefix_T("VMULCX", o.args),
    vmulcx_4xf := ~.vmulcx_2xf,
    vmulcx_8xf := ~.vmulcx_2xf,

    vswapcx_2xf := (self, o, i, is) >> self.prefix_T("VSWAPCX", o.args),
    vswapcx_4xf := ~.vswapcx_2xf,
    vswapcx_8xf := ~.vswapcx_2xf,

    fma  := (self, o, i, is) >> self.prefixTTT("FMA", o.args[1].t, o.args[2].t, o.args[3].t, o.args),
    fms  := (self, o, i, is) >> self.prefixTTT("FMS", o.args[1].t, o.args[2].t, o.args[3].t, o.args),
    nfma := (self, o, i, is) >> self.prefixTTT("NFMA", o.args[1].t, o.args[2].t, o.args[3].t, o.args),

    vstore_cmd := (self, o, i, is) >> Print(Blanks(i), self.prefixT("VSTORE", o.args[3].t, o.args), ";\n"),
    vload_cmd  := (self, o, i, is) >> Print(Blanks(i), self.prefixT("VLOAD", o.args[1].t, o.args), ";\n"),
));
