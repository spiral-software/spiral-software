
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Include(codegen);

StreamDefaults := Copy(SpiralDefaults);
StreamDefaults.generateInitFunc := false;
StreamDefaults.codegen := HDLCodegen;
StreamDefaults.sumsgen := sigma.LegacySumsGen;
StreamDefaults.compileStrategy:=conservativeCompileSSA;
StreamDefaults.globalUnrolling := 1024;
StreamDefaults.useDeref := false;

Import(paradigms.common);
Import(paradigms.common.id);
Import(paradigms.loops); 

Include(sort);
Include(sortvec);
Include(bitperms);
Include(streambitperms);
Include(streampermsnonbit);
Include(sums);
Include(nonterms);
Include(real);
Include(rules);
Include(dct);
Include(hacks);
Include(webscr);
Include(scripts);
Include(permnetwork);
Include(sw);
Include(sort_explore);
Include(explore_2ddft);
