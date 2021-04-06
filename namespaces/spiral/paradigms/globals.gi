
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(paradigms.common, paradigms.vector.breakdown);

#   NOTE: Phase out!!
Enable_tSPL := function()
    Print("\n\nImportant change: \"Enable_tSPL\" is to be phased out -- use \"tSPL_Globals.getDPOpts()\" and ask Franz\n\n");
    SwitchRulesQuiet(DFT, []);
    SwitchRulesQuiet(WHT, []);
    SwitchRulesQuiet(MDDFT, []);
    SwitchRulesName([DFT_Base, DFT_Rader, DFT_SplitRadix, DFT_tSPL_CT, DFT_CT, WHT_Base, WHT_tSPL_BinSplit, MDDFT_Base, MDDFT_tSPL_RowCol, WHT_tSPL_Pease], true);
    SwitchRulesName([DFT_vecrec, IxA_L_vecrec, DFT_vecrec_T, L_IxA_vecrec], USE_VECREC);
    SwitchRulesName([IxA_L_vec, L_IxA_vec], true);
end;


Class(tSPL_Globals, rec(
    getDPOpts := self >> rec(breakdownRules := self.breakdownRules()),
    breakdownRules := self >>
        rec(
            DFT := [ DFT_Base, DFT_Rader, DFT_SplitRadix, DFT_CT_Mincost, DFT_GoodThomas, DFT_tSPL_CT, DFT_tSPL_Rader, DFT_CT ],
            WHT := [ WHT_Base, WHT_tSPL_BinSplit, WHT_tSPL_Pease ],
            MDDFT := [ MDDFT_Base, MDDFT_tSPL_RowCol ],
            TTensor := [ AxI_IxB, IxB_AxI ],
            TTensorI := [ IxA_base, AxI_base, IxA_L_base, L_IxA_base ],
            TL := [ L_base ],
#D            TTag := [ TTag_down ],
            TRC := [TRC_tag ],
            TDiag := [ TDiag_tag ],
            TRaderMid:= [ TRaderMid_tag ],
            TDirectSum := [ A_dirsum_B ],
            TCompose := [ TCompose_tag ],
            TICompose := [ TICompose_tag ],
            TGrp := [ TGrp_tag ],
            TDR := [TDR_base ]
        )
));


#    [DFT_vecrec, IxA_L_vecrec, DFT_vecrec_T, L_IxA_vecrec], USE_VECREC
#    [IxA_L_vec_LS, L_IxA_vec_LS], USE_LOOPSPLITTING
