
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(paradigms.vector);
Import(paradigms.vector.sigmaspl);
Import(platforms.sse);

Declare(NEON);
Declare(NEON_HALF);

Include(misc);
Include(code);
Include(unparse);
Include(bench);
Include(rewrite);
Include(isa);
