
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(compiler, sigma, spl, formgen, rewrite);
ImportAll(transforms);
Import(code);
Import(search);
Import(paradigms.common); # IsTSPL
Import(paradigms.smp);    # SMP mixins, used in reccodegen.gi and unparse.gi
ImportAll(paradigms.vector);

RecursStep.isBlockTransitive := true; #!!!
RC        .isBlockTransitive := true; #!!!

Include(signature);
Include(codelet);
Include(codegen);
Include(hacks);

Include(dpbench); # NOTE: move out, problem: uses CodeletName/CodeletShape
Include(testbench);

Include(defaults);
Include(recgt);
Include(recvector);

