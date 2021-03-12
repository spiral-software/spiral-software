
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#   Helper Functions
DivisorsIntDrop := n -> let(l:=DivisorsInt(n), Sublist(l, [2..Length(l)-1]));

Load(spiral.paradigms.common.id);

Include(sigmaspl);
Include(tags);
Include(nonterms);
Include(gt);
Include(gtdft);
Include(breakdown);
Include(twiddle);
Include(dft);
Include(dftpease);
Include(wht);
Include(dct);
Include(dst);
Include(rdft);
Include(mdrdft);
Include(mdconv);
Include(dht);
Include(mdct);
Include(prune);
Include(interpolate);
