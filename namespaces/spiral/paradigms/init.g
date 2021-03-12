
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(rewrite, code, compiler, search, spl, formgen, sigma);
ImportAll(transforms);

[USE_LOOPSPLITTING, USE_VECREC] := [false, false];

Declare(Enable_tSPL);

Load(spiral.paradigms.common);
Load(spiral.paradigms.loops);
Load(spiral.paradigms.smp);
Load(spiral.paradigms.vector);
Load(spiral.paradigms.distributed);
Load(spiral.paradigms.multibuffer);
Load(spiral.paradigms.cache);
Load(spiral.paradigms.stream);
Load(spiral.paradigms.scratchpad);
Load(spiral.paradigms.dram);

Include(globals);
