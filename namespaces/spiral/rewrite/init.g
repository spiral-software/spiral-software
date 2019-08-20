
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details

# Rewriting Engine
# ----------------
# This package contains function that deal with pattern matching,
# and tree transformations and rewriting rules.
#
# In the future, the rewrite rule compiler should be implemented
# here.
#@P

Declare(RuleTrace, RuleStatus, @, Rule, IsRewriteRule, map_children_safe);

Include(attr);
Include(rules);
Include(ruleset);
Include(visitor);
Include(condpat);

Include(object);
Include(debug);
