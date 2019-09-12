# -*- Mode: shell-script -*-
# Decomposition of Matrices with Symmetry
# MP, 12.12.97 - 2.8.98, GAPv3.4

# Literature:
#   T. Minkwitz: PhD. Thesis, University of Karlsruhe, 1993
#   S. Egner   : PhD. Thesis, University of Karlsruhe, 1997
#   M.Pueschel : PhD. Thesis, University of Karlsruhe, 1998

if not IsBound(InfoAlgogen) then
  InfoAlgogen := Ignore;  # switch it on if you like
fi;
if not IsBound(RuntimeAlgogen) then
  RuntimeAlgogen := Ignore;  # switch it on to get runtime profile
fi;

# Nach Submission (AREP 1.1)
# --------------------------
# 11.09.98: Bei MatrixDecompositionByPermPermSymmetry und
#           MatrixDecompositionByMonMonSymmetry den hint "noOuter"
#           eingef"uhrt um schnellere Zerlegung m"oglich zu machen.
#           Bei der Haar Transformation macht das einen deutlichen
#           Unterschied.
# 30.04.99: Kleiner Bug in MatrixDecompositionByPermPermSymmetry:
#           Wenn die Permutationsdarstellungen nicht aequivalent
#           sind, hat er AR nicht berechnet.
#
# AREP 1.3
# --------
# 07.01.02: MatrixDecompositionByMon@IrredSymmetry
#           Runtime profoling eingefuehrt (siehe RuntimeAlgogen oben)

#F MatrixDecompositionByPermPermSymmetry( <mat/amat> [, <hint> ])
#F   decomposes <mat/amat> into a product of sparse
#F   matrices according to the perm-perm symmetry.
#F   An amat is returned representing the product.
#F   If the hint "noOuter" is supplied, then the symmetry
#F   is decomposed without using OuterTensorProductDecompositionMonRep.
#F   This may speed up the decomposition but may yield a suboptimal
#F   matrix decomposition.
#F

MatrixDecompositionByPermPermSymmetry := function ( arg )
  local M, hint, Rs, RL, RR, DL, DR, P, AL, AR, A, t1, t2, t3, t4;

  if Length(arg) = 1 then
    M    := arg[1];
    hint := "no hint";
  elif Length(arg) = 2 then
    M    := arg[1];
    hint := arg[2];
  else
    Error(
      "usage: ",
      "  MatrixDecompositionByPermPermSymmetry( <mat/amat> [, <hint> ] )"
    );
  fi;

  if not ( IsMat(M) or IsAMat(M) ) then
    Error("<M> must be a <mat/amat>");
  fi;
  if not hint in ["no hint", "noOuter"] then
    Error("<hint> must be \"noOuter\"");
  fi;

  # calculate the permperm symmetry
  t1 := Runtime();
  InfoAlgogen("#I CALCULATING THE PERMPERM SYMMETRY\n");
  Rs := PermPermSymmetry(M);
  RL := Rs[1];
  RR := Rs[2];

  # decompose the reps
  # if the reps are equivalent by 
  # a permutation, only one of 
  # them has to be decomposed
  InfoAlgogen("#I DECOMPOSING THE REPS\n");
  t2 := Runtime();
  DL := DecompositionMonRep(RL, hint);
  AL := DL.conjugation.element;
  if IsEquivalentARep(RL, RR) then
    P := ConjugationPermReps(RL, RR); 
    if P <> false then
      if IsIdentityMat(P) then
	AR := AL;
      else
        AR := P * AL;
      fi;
    else
      DR := DecompositionMonRep(RR, hint);
      AR := DR.conjugation.element;
    fi;
  else
    DR := DecompositionMonRep(RR, hint);
    AR := DR.conjugation.element;
  fi;

  # the block matrix
  InfoAlgogen("#I SPECIALIZING\n");
  t3 := Runtime();
  if IsMat(M) then
    M := AMatMat(M);
  fi;
  A := 
    AL *
    AMatSparseMat(MatAMat(InverseAMat(AL) * M * AR), false) *
    InverseAMat(AR);

  # avoid that A is checked for monomiality
  # by SimplifyAMat
  A.isMonMat := IsMonMat(M);
  A  := SimplifyAMat(A);
  t4 := Runtime();
  RuntimeAlgogen(
    "finding symmetry: ", t2 - t1, "\n",
    "decomposing symmetry: ", t3 - t2, "\n",
    "specializing: ", t4 - t3, "\n",
    "total: ", t4 - t1, "\n"
  );
  return A;
end;


#F MatrixDecompositionByMonMonSymmetry( <mat/amat> [, <hint> ] )
#F   decomposes <mat/amat> into a product of sparse
#F   matrices according to the mon-mon symmetry.
#F   An amat is returned representing the product.
#F   If the hint "noOuter" is supplied, then the symmetry
#F   is decomposed without using OuterTensorProductDecompositionMonRep.
#F   This may speed up the decomposition but may yield a suboptimal
#F   matrix decomposition.
#F

MatrixDecompositionByMonMonSymmetry := function ( arg )
  local M, hint, Rs, RL, RR, DL, DR, AL, AR, A, t1, t2, t3, t4;

  if Length(arg) = 1 then
    M    := arg[1];
    hint := "no hint";
  elif Length(arg) = 2 then
    M    := arg[1];
    hint := arg[2];
  else
    Error(
      "usage: ",
      "  MatrixDecompositionByMonMonSymmetry( <mat/amat> [, <hint> ] )"
    );
  fi;

  if not ( IsMat(M) or IsAMat(M) ) then
    Error("<M> must be a <mat/amat>");
  fi;
  if not hint in ["no hint", "noOuter"] then
    Error("<hint> must be \"noOuter\"");
  fi;

  # calculate the monmon symmetry
  InfoAlgogen("#I CALCULATING THE MONMON SYMMETRY\n");
  t1 := Runtime();
  Rs := MonMonSymmetry(M);
  RL := Rs[1];
  RR := Rs[2];

  # decompose the reps
  InfoAlgogen("#I DECOMPOSING THE SYMMETRY\n");
  t2 := Runtime();
  DL := DecompositionMonRep(RL, hint);
  DR := DecompositionMonRep(RR, hint);
  AL := DL.conjugation.element;
  AR := DR.conjugation.element;

  # the block matrix
  InfoAlgogen("#I SPECIALIZING\n");
  t3 := Runtime();
  if IsMat(M) then
    M := AMatMat(M);
  fi;
  A := 
    AL *
    AMatSparseMat(MatAMat(InverseAMat(AL) * M * AR), false) *
    InverseAMat(AR);

  # avoid that A is checked for monomiality
  # by SimplifyAMat
  A.isMonMat := IsMonMat(M);
  A  := SimplifyAMat(A);
  t4 := Runtime();
  RuntimeAlgogen(
    "finding symmetry: ", t2 - t1, "\n",
    "decomposing symmetry: ", t3 - t2, "\n",
    "specializing: ", t4 - t3, "\n",
    "total: ", t4 - t1, "\n"
  );
  return A;
end;


#F MatrixDecompositionByPermIrredSymmetry( <mat/amat> [, <maxblocksize> ] )
#F   decomposes <mat/amat> into a product of sparse
#F   matrices according to the perm-irred symmetry
#F   returned by the function PermIrredSymmetry1.
#F   An amat is returned representing the product.
#F   Only those symmetries where all irreducibles 
#F   are of degree <= <maxblocksize> are considered. 
#F   The default for <maxblocksize> is 2.
#F   Among all symmetries [RL, RR] the best is chosen 
#F   according to the following measure.
#F   If
#F     RL ~= RR ~= directsum R_i^(n_i), and Q = sum n_i^2 * d_i^2,
#F   with d_i = deg(R_i), then the best symmetry has
#F   the smallest value of Q.
#F

MatrixDecompositionByPermIrredSymmetry := function ( arg )
  local M, max, dim, Rs, Qs, min, pos, R, D, A, t1, t2, t3, t4;

  # decode and check arguments
  if Length(arg) = 1 then
    M   := arg[1];
    max := 2;
  elif Length(arg) = 2 then
    M   := arg[1];
    max := arg[2];
  else
    Error(
      "usage: \n", 
      "  MatrixDecompositionByPermIrredSymmetry( ",
      "    <mat/amat> [, <maxblocksize> ]  )"
    );
  fi;
  if IsAMat(M) then
    M := MatAMat(M);
  fi;
  if not IsMat(M) then
    Error("<M> must be a matrix");
  fi;
  if not ( IsInt(max) and max >= 1 ) then
    Error("<max> must be a posiive integer");
  fi;

  dim := DimensionsMat(M);
  if dim[1] <> dim[2] then
    return AMatMat(M);
  fi;

  # calculate the symmetry
  # use PermIrredSymmetry1 for time reasons
  InfoAlgogen("#I CALCULATING THE PERMIRRED SYMMETRY\n");
  t1 := Runtime();
  Rs := PermIrredSymmetry1(M, max);
  if Length(Rs) > 0 then
    
    # take the best pair for decomposition,
    # let R = directsum R_i^(n_i) be the decomposition
    # of the right (or left) side of the perm-irred symmetry
    # into irreducibles, then the quality is given by 
    # a small value of 
    #   sum n_i^2 * d_i^2, 
    # where d_i = deg(R_i)
    InfoAlgogen("#I CHOOSING SYMMETRY\n");
    Qs := 
      List(
        Rs, 
        p -> 
          Sum(
	    List(
	      Collected(
		List(p[2].rep.summands, r -> CharacterARep(r))
	      ),
	      cn -> Degree(cn[1])^2 * cn[2]^2
	    )
          )
      );
    min := Minimum(Qs);
    pos := PositionProperty(Qs, q -> q = min);
    R   := Rs[pos];

    InfoAlgogen("#I DECOMPOSING THE SYMMETRY\n");
    t2 := Runtime();
    D := DecompositionMonRep(R[1]).conjugation.element;

    # the block matrix
    t3 := Runtime();
    InfoAlgogen("#I SPECIALIZING\n");
    M := AMatMat(M);
    A := 
      D *
      AMatSparseMat(
        MatAMat(InverseAMat(D) * M * InverseAMat(R[2].conjugation)), 
        false
      ) *
      R[2].conjugation;

    # avoid that A is checked for monomiality
    # by SimplifyAMat
    A.isMonMat := IsMonMat(M);
    A  := SimplifyAMat(A);
    t4 := Runtime();
    RuntimeAlgogen(
      "finding symmetry: ", t2 - t1, "\n",
      "decomposing symmetry: ", t3 - t2, "\n",
      "specializing: ", t4 - t3, "\n",
      "total: ", t4 - t1, "\n"
    );
    return A;
  fi;

  # decomposition failed
  return AMatMat(M);

end;


#F MatrixDecompositionByMon2IrredSymmetry( <mat/amat> [, <maxblocksize> ] )
#F   decomposes <mat/amat> into a product of sparse
#F   matrices according to the mon2-irred symmetry
#F   returned by the function Mon2IrredSymmetry1.
#F   The Mon2IrredSymmetry considers all monomial representations
#F   with entries +/-1.
#F   An amat is returned representing the product.
#F   Only those symmetries where all irreducibles
#F   are of degree <= <maxblocksize> are considered.
#F   The default for <maxblocksize> is 2.
#F   Among all symmetries [RL, RR] the best is chosen
#F   according to the following measure.
#F   If
#F     RL ~= RR ~= directsum R_i^(n_i), and Q = sum n_i^2 * d_i^2,
#F   with d_i = deg(R_i), then the best symmetry has
#F   the smallest value of Q.
#F

MatrixDecompositionByMon2IrredSymmetry := function ( arg )
  local M, max, dim, Rs, Qs, min, pos, R, D, A, t1, t2, t3, t4;

  # decode and check arguments
  if Length(arg) = 1 then
    M   := arg[1];
    max := 2;
  elif Length(arg) = 2 then
    M   := arg[1];
    max := arg[2];
  else
    Error(
      "usage: \n",
      "  MatrixDecompositionByMon2IrredSymmetry( ",
      "    <mat/amat> [, <maxblocksize> ]  )"
    );
  fi;
  if IsAMat(M) then
    M := MatAMat(M);
  fi;
  if not IsMat(M) then
    Error("<M> must be a matrix");
  fi;
  if not ( IsInt(max) and max >= 1 ) then
    Error("<max> must be a posiive integer");
  fi;

  dim := DimensionsMat(M);
  if dim[1] <> dim[2] then
    return AMatMat(M);
  fi;

  # calculate the symmetry
  # use Mon2IrredSymmetry1 for time reasons
  InfoAlgogen("#I CALCULATING THE MON@IRRED SYMMETRY\n");
  t1 := Runtime();
  Rs := Mon2IrredSymmetry1(M, max);
  if Length(Rs) > 0 then

    # take the best pair for decomposition,
    # let R = directsum R_i^(n_i) be the decomposition
    # of the right (or left) side of the perm-irred symmetry
    # into irreducibles, then the quality is given by
    # a small value of
    #   sum n_i^2 * d_i^2,
    # where d_i = deg(R_i)
    InfoAlgogen("#I CHOOSING SYMMETRY\n");
    Qs :=
      List(
        Rs,
        p ->
          Sum(
            List(
              Collected(
                List(p[2].rep.summands, r -> CharacterARep(r))
              ),
              cn -> Degree(cn[1])^2 * cn[2]^2
            )
          )
      );
    min := Minimum(Qs);
    pos := PositionProperty(Qs, q -> q = min);
    R   := Rs[pos];

    InfoAlgogen("#I DECOMPOSING THE SYMMETRY\n");
    t2 := Runtime();
    D := DecompositionMonRep(R[1]).conjugation.element;

    # the block matrix
    InfoAlgogen("#I SPECIALIZING\n");
    t3 := Runtime();
    M := AMatMat(M);
    A :=
      D *
      AMatSparseMat(
        MatAMat(InverseAMat(D) * M * InverseAMat(R[2].conjugation)),
        false
      ) *
      R[2].conjugation;

    # avoid that A is checked for monomiality
    # by SimplifyAMat
    A.isMonMat := IsMonMat(M);
    A  := SimplifyAMat(A);
    t4 := Runtime();
    RuntimeAlgogen(
      "finding symmetry: ", t2 - t1, "\n",
      "decomposing symmetry: ", t3 - t2, "\n",
      "specializing: ", t4 - t3, "\n",
      "total: ", t4 - t1, "\n"
    );
    return A;
  fi;

  # decomposition failed
  return AMatMat(M);

end;

