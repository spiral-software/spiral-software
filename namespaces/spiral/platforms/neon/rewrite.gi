
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RewriteRules(RulesStrengthReduce, rec(

################## HALF ####################
    vunpacklo_half := Rule( vunpacklo_half,
        e -> vextract_half(vtrnq_half(e.args[1], e.args[2]), [0])),
    vunpackhi_half := Rule( vunpackhi_half,
        e -> vextract_half(vtrnq_half(e.args[1], e.args[2]), [1])),


################## NEON ####################
    vpacklo_neon := Rule( vpacklo_neon,
        e -> vextract_neon_4x32f(vuzpq_32f(e.args[1], e.args[2]), [0])),
    vpackhi_neon := Rule( vpackhi_neon,
        e -> vextract_neon_4x32f(vuzpq_32f(e.args[1], e.args[2]), [1])),
    vunpacklo_neon := Rule( vunpacklo_neon,
        e -> vextract_neon_4x32f(vzipq_32f(e.args[1], e.args[2]), [0])),
    vunpackhi_neon := Rule( vunpackhi_neon,
        e -> vextract_neon_4x32f(vzipq_32f(e.args[1], e.args[2]), [1])),
    vtransposelo_neon := Rule( vtransposelo_neon,
        e -> vextract_neon_4x32f(vtrnq_32f(e.args[1], e.args[2]), [0])),
    vtransposehi_neon := Rule( vtransposehi_neon,
        e -> vextract_neon_4x32f(vtrnq_32f(e.args[1], e.args[2]), [1])),
    vunpacklolo2_neon := Rule( vunpacklolo2_neon,
        e -> vextract_neon_4x32f(vtrnq_32f(vpacklo_neon(e.args[1], e.args[2]), vpackhi_neon(e.args[1], e.args[2])), [0])),
    vunpackhihi2_neon := Rule( vunpackhihi2_neon,
        e -> vextract_neon_4x32f(vtrnq_32f(vpacklo_neon(e.args[1], e.args[2]), vpackhi_neon(e.args[1], e.args[2])), [1])),

));

