# -*- Mode: shell-script -*-
# Automatic loading for AREP-Library
# MP, 12.03.98, converted to GAP v3.4.4
# SE, 21.01.98, as 'eplib.g' for EPLib
# SE, 22.07.98, AMatSparseMat, IdentityAMat added
# MP, 15.08.99, freezing AREP 1.1

# Nach Submission (AREP 1.1)
# --------------------------
# 12.09.98: Neue Funktionen aus amat.g eingef"ugt:
#           PrintAMatToTexHeader, PrintAMatToTexEnd,
#           PrintAMatToTex, AppendAMatToTex, UnscrambleDFT.
#           (siehe amat.g oben).
# 16.10.98: Banner ge"andert

# AREP 1.2
# --------
# 13.03.00: Neue Funktionen aus transf.g
# 28.03.00: Neue Funktionen aus transf.g
# 15.03.01: Neue Funktion in permperm.g
#
# AREP 1.3
# --------
# 08.01.02: neue files fuer mon2-irred symmetry:
#   mon2mat.g, mon2blk.g; entsprechende Funktionen
#   in symmetry.g, algogen.g
# 20.06.03: neue Funktionen in transf.g
# 23.08.06: neue Funktion in amat.g und in der neuen Datei real.g
# 10.12.06: neue Funktion in amat.g

# Print the logo (not for spiral)

#if BANNER and not QUIET then
#  Print("\n");
#  
#  # Use the PR function from GAP's init.g
#  PR("  ___   ___   ___  ___   ");
#  PR(" |   | |   | |    |   |   Version 1.2, 15 August 2000"); 
#  PR(" |___| |___| |___ |___|   ");
#  PR(" |   | |  \\  |    |       by Sebastian Egner  "); 
#  PR(" |   | |   \\ |___ |          Markus Pueschel  ");
#  PR(" ");
#  PR("Abstract REPresentations");
#  Print("\n");
#
#fi;

# The exported symbols of the package AREP
# ========================================

AUTO( ReadPkg("arep", "lib/algogen"),
  MatrixDecompositionByPermPermSymmetry,
  MatrixDecompositionByMonMonSymmetry,
  MatrixDecompositionByPermIrredSymmetry,
  MatrixDecompositionByMon2IrredSymmetry
);

AUTO( ReadPkg("arep", "lib/amat"),
  AMatOps,
  IsAMat,
  IsInvertibleMat,
  IsIdentityMat,
  IdentityPermAMat,
  IdentityMonAMat,
  IdentityMatAMat,
  IdentityAMat,
  AllOneAMat,
  NullAMat,
  DiagonalAMat,
  DFTAMat,
  RDFTAMat,
  RotationAMat,
  SORAMat,
  AMatPerm,
  AMatMon,
  AMatMat,
  ScalarMultipleAMat,
  PowerAMat,
  ConjugateAMat,
  DirectSumAMat,
  TensorProductAMat,
  GaloisConjugateAMat,
  InverseAMat,
  PseudoInverseAMat,
  TransposedAMat,
  DeterminantAMat,
  TraceAMat,
  RankAMat,
  IsPermMat,
  IsMonMat,
  PermAMat,
  MonAMat,
  MatAMat,
  PermAMatAMat,
  MonAMatAMat,
  MatAMatAMat,
  UnscrambleDFT,
  UnscrambleRotation,
  SimplifyAMatFirstParse,
  SimplifyAMat,
  kbsAMat,
  SubmatrixAMat,
  kbsDecompositionAMat,
  LinearComplexityAMat,
  AMatSparseMat,
  PrintAMatToTexHeader,
  PrintAMatToTexEnd,
  PrintAMatToTex,
  AppendAMatToTex,
  RecognizeCosPi
);

AUTO( ReadPkg("arep", "lib/arep"),
  ARepOps,
  IsARep,
  TrivialPermARep,
  TrivialMonARep,
  TrivialMatARep,
  RegularARep,
  NaturalARep,
  ARepByImages,
  ARepByHom,
  ARepByCharacter,
  ConjugateARep,
  DirectSumARep,
  InnerTensorProductARep,
  OuterTensorProductARep,
  GaloisConjugateARep,
  RestrictionARep,
  InductionARep,
  ExtensionARep,
  GroupWithGenerators,
  ImageARep,
  IsEquivalentARep,
  CharacterARep,
  IsIrreducibleARep,
  KernelARep,
  IsFaithfulARep,
  ARepWithCharacter,
  GeneralFourierTransform,
  IsPermRep,
  IsMonRep,
  PermARepARep,
  MonARepARep,
  MatARepARep,
  RandomMonRep
);

AUTO( ReadPkg("arep", "lib/arepfcts"),
  IsRestrictedCharacter,
  AllExtendingCharacters,
  OneExtendingCharacter,
  IntertwiningSpaceARep,
  IntertwiningNumberARep,
  UnderlyingPermARep,
  IsTransitiveMonRep,
  IsPrimitiveMonRep,
  TransitivityDegreeMonRep,
  OrbitDecompositionMonRep,
  TransitiveToInductionMonRep,
  InsertedInductionARep,
  ConjugationPermLists,
  ConjugationPermReps,
  ConjugationTransitiveMonReps,
  TransversalChangeInductionARep,
  OuterTensorProductDecompositionMonRep,
  InnerConjugationARep,
  RestrictionInductionARep,
  AllMaximalNormalSubgroupsBetween,
  OneMaximalNormalSubgroupBetween,
  RestrictionToSubmoduleARep,
  kbsARep,
  kbsDecompositionARep,
  ExtensionOnedimensionalAbelianRep,
  DecompositionMonRep
);

AUTO( ReadPkg("arep", "lib/complex"),
  ImaginaryUnit,
  Conjugate,
  Re,
  Im,
  AbsSqr,
  Sqrt,
  ExpIPi,
  CosPi,
  SinPi,
  TanPi,
  ReducingRatFactorCyc,
  ReducingCycFactor
);

AUTO( ReadPkg("arep", "lib/mon"),
  IsMon,
  IsPermMon,
  IsDiagMon,
  MonOps,
  Mon,
  MatMon,
  MonMat,
  PermMon,
  DegreeMon,
  CharacteristicMon,
  OrderMon,
  TransposedMon,
  DeterminantMon,
  TraceMon,
  GaloisMon,
  DirectSumMon,
  DirectSumMon,
  TensorProductMon,
  TensorProductMon,
  CharPolyCyclesMon
);

AUTO( ReadPkg("arep", "lib/monmon"),
  MonMonSym,
  MonMonSymL,
  MonMonSymR
);

AUTO( ReadPkg("arep", "lib/mon2blk"),
  kbsMon2M,
  LessMon2BlockSym,
  CompletedMon2BlockSym,
  DisplayMon2BlockSym,
  BlockMon2BlockSym,
  Mon2BlockSymBySubsets
);

AUTO( ReadPkg("arep", "lib/mon2mat"),
  Mon2Encode,
  Mon2Decode,
  FullMon2Group,
  Mon2MatSym,
  Mon2MatSymL,
  Mon2MatSymR
);

AUTO( ReadPkg("arep", "lib/permblk"),
  HasseEdgesList,
  BlockOfPartition,
  IsRefinedPartition,
  MeetPartition,
  JoinPartition,
  PartitionIndex,
  kbs,
  kbsM,
  LessPermBlockSym,
  CompletedPermBlockSym,
  DisplayPermBlockSym,
  PermBlockSymL,
  PermBlockSymR,
  PermBlockSymByPermutations,
  PermBlockSymBySubsets
);

AUTO( ReadPkg("arep", "lib/permmat"),
  SelectBaseFromList,
  PermMatSym,
  PermMatSymL,
  PermMatSymR,
  PermMatSymNormalL
);

AUTO( ReadPkg("arep", "lib/permperm"),
  FewGenerators,
  PermPermSym,
  PermPermSymL,
  PermPermSymR
);

AUTO( ReadPkg("arep", "lib/real"),
  IsRealMon,
  IsRealRep,
  RealDecompositionMonRep,
  RestrictToAbelianSymmetry,
  PRealMatrixDecompositionByPermPermSymmetry,
  PRealMatrixDecompositionByMonMonSymmetry,
  RealMatrixDecompositionByPermIrredSymmetry,
  RealMatrixDecompositionByMon2IrredSymmetry,
  PRealMatrixDecomposition
);

AUTO( ReadPkg("arep", "lib/summands"),
  DirectSummandsPermutedMat
);

AUTO( ReadPkg("arep", "lib/symmetry"),
  PermPermSymmetry,
  MonMonSymmetry,
  PermIrredSymmetry,
  PermIrredSymmetry1,
  Mon2IrredSymmetry,
  Mon2IrredSymmetry1
);

AUTO( ReadPkg("arep", "lib/tools"),
  PartitionIndex,
  PartitionRefinement,
  DirectProductSymmetricGroups,
  DiagonalMat,
  DirectSumMat,
  TensorProductMat,
  MatPerm,
  PermMat,
  PermutedMat,
  DirectSumPerm,
  TensorProductPerm,
  MovedPointsPerm,
  NrMovedPointsPerm,
  PermOfCycleType,
  SupportVector,
  RowBlockStructureMat,
  ColumnBlockStructureMat,
  BlockStructureMat
);

AUTO( ReadPkg("arep", "lib/transf"),
  DiscreteFourierTransform,
  DiscreteFourierTransform,
  InverseDiscreteFourierTransform,
  InverseDiscreteFourierTransform,
  DiscreteHartleyTransform,
  InverseDiscreteHartleyTransform,
  DiscreteCosineTransform,
  InverseDiscreteCosineTransform,
  DiscreteSineTransform,
  InverseDiscreteSineTransform,
  DiscreteCosineTransformIV,
  InverseDiscreteCosineTransformIV,
  DiscreteSineTransformIV,
  InverseDiscreteSineTransformIV,
  DiscreteCosineTransformI,
  InverseDiscreteCosineTransformI,
  DiscreteSineTransformI,
  InverseDiscreteSineTransformI,
  ModifiedCosineTransform,
  InverseModifiedCosineTransform,
  WalshHadamardTransform,
  InverseWalshHadamardTransform,
  SlantTransform,
  InverseSlantTransform,
  HaarTransform,
  InverseHaarTransform,
  RationalizedHaarTransform,
  InverseRationalizedHaarTransform,
  DFT,
  InvDFT,
  RDFT,
  DHT,
  InvDHT,
  DCT,
  InvDCT,
  DST,
  InvDST,
  DCT_IV,
  InvDCT_IV,
  DST_IV,
  InvDST_IV,
  DCT_I,
  InvDCT_I,
  DCT_II,
  DCT_III,
  DST_I,
  InvDST_I,
  DST_II,
  DST_III,
  MDCT,
  IMDCT,
  WHT,
  InvWHT,
  ST,
  InvST,
  HT,
  InvHT,
  RHT,
  InvRHT,
  CosDFT,
  SinDFT,
  DCT_Iunscaled,
  DCT_IIunscaled,
  DCT_IIIunscaled,
  DCT_IVunscaled,
  DCT_Vunscaled,
  DCT_VIunscaled,
  DCT_VIIunscaled,
  DCT_VIIIunscaled,
  DCTunscaled,
  DST_Iunscaled,
  DST_IIunscaled,
  DST_IIIunscaled,
  DST_IVunscaled,
  DST_Vunscaled,
  DST_VIunscaled,
  DST_VIIunscaled,
  DST_VIIIunscaled,
  DSTunscaled,
  PolynomialDTT,
  normalizeCosine,
  zerosT,
  SkewDCT_IIIunscaled,
  SkewDCT_IVunscaled,
  SkewDST_IIIunscaled,
  SkewDST_IVunscaled,
  IMDCTunscaled
);


