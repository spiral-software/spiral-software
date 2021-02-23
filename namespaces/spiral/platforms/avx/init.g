
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(paradigms.vector);
Import(paradigms.vector.sigmaspl);
Import(platforms.sse);

Declare(AVX_4x64f, AVX_8x32f);

Include(misc);
Include(code);
Include(unparse);
Include(isa);
Include(bench);
Include(sreduce);
Include(cvt);
