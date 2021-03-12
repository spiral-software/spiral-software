
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Sigma-SPL to code translation and basic block compiler
# -----------------------------------------------------
# This package contains functions for translation Sigma-SPL to code
# (in-place and regular), and compiling and optimizing the basic
# blocks in the code.
#@P

Import(rewrite, code, spl, formgen, sigma);

Declare(SuccLoc, Compile);

Include(ssa);
Include(cse);
Include(copyprop);

Include(newunroll);
Include(loopind);
Include(cse_rand);
Include(dag);

Declare(DefFrontier);

Include(binsplit);
Include(frontier);
Include(sched);

Include(fma);
Include(cxfma);

Include(compile);
Include(hoister);
Include(simpleloop);
Include(unrblock);
Include(sums2code);
Include(sums2ipcode);
Include(c);
Include(fp);
Include(top);
Include(codegen);
Include(rollingptr);

# codegen strategy
Class(CodegenStrat);
Include(cgwrap);
Include(cgslab);
Include(cgstrat);

Include(unparse);

Include(datatype);
Include(bug);
Include(vref);
Include(regalloc);
Include(two_op);
Include(x86);
Include(io);

Include(strategy);

Include(three_op);
Include(inttab);


