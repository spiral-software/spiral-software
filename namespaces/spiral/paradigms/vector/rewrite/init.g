
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(rewrite, paradigms.vector.sigmaspl, paradigms.vector.bases, paradigms.vector.breakdown);

canFuse := e -> IsBound(e.fuse);

Include(sreduce);
Include(vectorize);
Include(propagate);
Include(diag);
Include(vrc);
Include(rc);
Include(conj);
Include(rader);
Include(fadd);
Include(strategy);
