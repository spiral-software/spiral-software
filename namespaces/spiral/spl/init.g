
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Import(rewrite, code, approx);
Declare(I, fId, fConst, Gath, Perm, Diag, IsNonTerminal, IsSPLSym, IsSPLMat, IDirSum, BaseOperation);
Declare(Conjugate, Compose, Scale, SUM, Blk, InertConjTranspose);

# How to deal with 2 x 2 rotation matrices
#   1. sums * diag * sums  (3 adds, 3 mults)
#   2. Three lifting steps (fused mult/adds)  (3 adds, 3 mults)
#   3. By definition (2 adds, 4 mults)
#
EXPAND_ROTATIONS := 1;

Include(objhash);
Include(splfunc);
Include(print);

Include(SPL);
Include(BaseMat);
Include(Sym);
Include(NonTerminal);
Include(Mat);
Include(Diag);

# BaseOperation
Include(BaseOperation);
Include(Tensor);
Include(Compose);
Include(DirectSum);
Include(Conjugate);

# BaseMat
Include(Sparse);
Include(Perm);

# BaseOperation
Include(BaseOverlap);
Include(RowColTensor);
Include(RowDirectSum);
Include(ColDirectSum);
Include(Stack);

# BaseIterative
Include(BaseIterative);
Include(Iter);

# BaseContainer
Include(BaseContainer);
Include(Scale);

# Sigma-SPL
Include(sums);
Include(PermClass);
Include(perms);
Include(perms2);
Include(hfunc);
Include(transpose);


Declare(ToeplitzMat);
Include(diags);
Include(diags2);
Include(ij);
Include(symbols);
Include(export); # expand rotations
Include(Mon);
Include(interval);
# OL
Include(ol);
# OL extensions
Include(ext_ol);
# Various sparse matrices
Include(matrices);
# SPLAMat
Include(amat);
# Other
Include(auxil);
Include(random);
# Index-free Sigma-SPL
Include(gtfuncs);
Include(gtsums);
Include(gtrules);
Include(tags);
## hashing
Include(hashspl);
Include(vwrap);
# extensions
Include(ext_bitperms);

pm := x->PrintMat(When(IsSPL(x), MatSPL(x), x));
