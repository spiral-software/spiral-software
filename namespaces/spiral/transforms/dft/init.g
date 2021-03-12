
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

_hhi := (N, n, b, s, spl) -> Scat(H(N,n,b,s)) * spl * Gath(H(N,n,b,s));
_hho := (N, n, scat_b, scat_s, gath_b, gath_s, spl) -> Scat(H(N,n,scat_b,scat_s)) * spl * Gath(H(N,n,gath_b,gath_s));

Include(dft);
Include(sincosdft);
Include(dft_rules);
Include(skewdft);
Include(dft234);
Include(modperms);
Include(inplace);
Include(multidim);
Include(pd);
Include(udft);
Include(dft_srreg);
Include(gauss);
Include(prune);
Include(mdprune);

SwitchRulesByNameQuiet(DFT, [DFT_Base, DFT_Base1, DFT_CT, DFT_Rader, DFT_PD]);
