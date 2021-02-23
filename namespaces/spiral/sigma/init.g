
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#
# Sigma-SPL Optimization and Rewriting Rules
# ------------------------------------------
#
# This package defines common rewriting rules and strategies for
# Sigma-SPL that will be useful for most transforms.
#
# A transform might define additional transform specific rewriting
# rules.
#@P

Import(code, rewrite, formgen, spl);

Include(spl2sums);
Include(ol);
Include(sumsgen);
Include(memo);
Include(sums_rules);

Include(merge_tensors);
Include(func_rules);
Include(ii_rules);

Include(dot_rules);
Include(diag_rules);
Include(modp_rules);
Include(hfunc_rules);
Include(rc_rules);

Include(precompute);
Include(strategy);
Include(sums_unification);
Include(sums_ruletree);

Include(mon_rules);
Include(latex);
Include(directsum);
Include(smap);
