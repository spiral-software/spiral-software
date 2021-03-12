
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Rewriting Engine
# ----------------
# This module contains function that deal with pattern matching,
# tree transformations, and rewrite rules.


Declare(RuleTrace, RuleStatus, @, Rule, IsRewriteRule, map_children_safe);

Include(attr);
Include(rules);
Include(ruleset);
Include(visitor);
Include(condpat);

Include(object);
Include(debug);
