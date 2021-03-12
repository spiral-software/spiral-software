
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(paradigms.vector);
Import(paradigms.vector.sigmaspl);

Include(misc);
Include(code);
Declare(SSE_2x64f, SSE_2x32f, SSE_4x32f, SSE_8x16i, SSE_16x8i, SSE_4x32i, SSE_2x64i);
Include(unparse);
Include(sreduce);
Include(isa);
Include(cvt);
Include(bench);
Include(bitcode);
