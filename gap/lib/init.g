# -*- Mode: shell-script -*-
# Changes:
# - calls to sml/, tbl/, thr/, tom/, two/ removed
# - several changes for startup
# - in the end load arep and formgen

#############################################################################
##
#A  init.g                      GAP library                  Martin Schoenert
##
##
#Y  Copyright (C) 2018-2021, Carnegie Mellon University
#Y  All rights reserved.  See LICENSE for details.
#Y  
#Y  This work is based on GAP version 3, with some files from version 4.  GAP is
#Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
##


Global.LoadStack := [];

_basesRecname := RecName("__bases__");
_opsRecname := RecName("operations");

_pos := function(lst, elt)
    local i;
    for i in [1..Length(lst)] do
        if Same(lst[i], elt) then return i; fi;
    od;
    return false;
end;
        
# moves the field __bases__ in a record to the topmost position,
# for faster lookup of fields that live in one of the __bases__
_moveBasesUp := function(r)
    local i,ch,t1,t2, ofs;
    if not IsRec(r) then return Error("<r> must be a record"); fi;
    ch := Children(r);
    ch := ch{[2..Length(ch)]};
    ofs := 1;

    i := _pos(ch, _basesRecname);
    if i<>false then 
        r[1] := ch[i];
        r[2] := ch[i+1];
        r[i] := ch[1];
        r[i+1] := ch[2];
        ofs := 3;
    fi;

    ch := Children(r);
    ch := ch{[2..Length(ch)]};
    i := _pos(ch, _opsRecname);
    if i<>false then 
        r[ofs] := ch[i];
        r[ofs+1] := ch[i+1];
        r[i] := ch[ofs];
        r[i+1] := ch[ofs+1];
    fi;
    return r;
end;
_moveBasesDown := function(r)
    local i,ch,t1,t2,len;
    if not IsRec(r) then return Error("<r> must be a record"); fi;
    ch := Children(r);
    ch := ch{[2..Length(ch)]};
    len := Length(ch);

    i := _pos(ch, _basesRecname);
    if i<>false then 
        r[len-1] := ch[i];
        r[len] := ch[i+1];
        r[i] := ch[len-1];
        r[i+1] := ch[len];
    fi;
    
    ch := Children(r);
    ch := ch{[2..Length(ch)]};
    i := _pos(ch, _opsRecname);
    if i<>false then 
        r[1] := ch[i];
        r[2] := ch[i+1];
        r[i] := ch[1];
        r[i+1] := ch[2];
    fi;

    return r;
end;

FileManager := rec(
    files := [],
    addFile := (self,f) >> Add(self.files, f)
);

ProgressBar := rec(
     __call__ := (self, events_per_dot) >> 
         _moveBasesUp(rec(__bases__ := [self], n:=0, events_per_dot:=events_per_dot, done:=false)),

    advance := meth(self)
        self.n := self.n + 1;
	if not self.done and (self.n mod self.events_per_dot = 0) then
	    PrintTo("*errout*", "."); 
	fi;
    end
);
     
_fileProgress := ProgressBar(10);

Add(HooksAfterOpenInput, arg -> _fileProgress.advance());
Add(HooksAfterOpenInput, function(arg) CurrentFile().id := false; end);

WarnUndefined := function(path, pkg)
    local vars, v, res;
    vars := NSFields(pkg);
    path := PathRelativeToSPIRAL(path);
    for v in vars do
        res := Try(IsVoid(pkg.(v)));
        if res[1]=false then Print("Warning: ", path, ": '", v, "' is broken, ", res[2], "\n"); fi;
        if res[2]=true then Print("Warning: ", path, ": '", v, "' is declared but not initialized\n"); fi;
    od;
end;

Add(HooksBeforeCloseInput,
function(arg)
    local cf;
    cf := CurrentFile();
    #WarnUndefined(cf.pkg);
    FileManager.addFile( rec( file := PathRelativeToSPIRAL(cf.fileName),
                          lines := cf.lineNumber,
                  pkg   := cf.pkg,
                  id    := When(IsBound(cf.id), cf.id, false) ) );
    ClearCommentBuffer();
end);

##  CVSID := function(cvs_id)
##      CurrentFile().id := cvs_id;
##  end;

VERLIB := "v3r4p4 1997/04/18";

PATH_SEP := config_val_t_strval_get(config_demand_val("path_sep"));

Global.SpiralVersion := Version();

PrintBannerSpiral := function()
    Print("\n");
    Print("     _____       _            __   \n");
    Print("    / ___/____  (_)________ _/ /  \n");
    Print("    \\__ \\/ __ \\/ / ___/ __ `/ / \n");
    Print("   ___/ / /_/ / / /  / /_/ / /  \n");
    Print("  /____/ .___/_/_/   \\__,_/_/  \n");
    Print("      /_/                                              \n");
    Print("\n");
    Print("  http://www.spiral.net                                   \n");
    Print("  Spiral ", SpiralVersion, "\n");
    Print("----------------------------------------------------------   \n");
end;

PrintBannerShortSPIRAL := function()
    Print(" Spiral ", SpiralVersion, "\n");
    Print("----------------------------------------------------------   \n");
end;

if not QUIET and BANNER then
    PrintBannerSpiral();
fi;

ReadIndent := "";

if not IsBound( InfoRead1 )  then InfoRead1 := Ignore;  fi;
if not IsBound( InfoRead2 )  then InfoRead2 := Ignore;  fi;

ReplacedString := function ( string, old, new )
    local  res,  i,  k,  l;
    res := [];
    k := 1;
    l := false;
    for i  in [1..Length(string)]  do
        if string{[i..i+Length(old)-1]} = old  then
            l := i;
        fi;
        if string[i] = ';'  then
            if l <> false  then
                Append( res, string{[k..l-1]} );
                Append( res, new );
                Append( res, string{[l+Length(old)..i]} );
            else
                Append( res, string{[k..i]} );
            fi;
            k := i + 1;
            l := false;
        fi;
    od;
    if l <> false  then
        Append( res, string{[k..l-1]} );
        Append( res, new );
        Append( res, string{[l+Length(old)..Length(string)]} );
    else
        Append( res, string{[k..Length(string)]} );
    fi;
    return res;
end;

ReadPath := function ( path, name, ext, infomsg )
    local   readIndent, i, k, file, found;
    readIndent := Copy( ReadIndent );
    Append( ReadIndent, "  " );
    InfoRead1( "#I",ReadIndent,infomsg,"( \"", name, "\" )\n" );
    i := 1;
    found := false;
    while not found  and i <= Length(path)+1 do
        k := Position( path, ';', i-1 );
        if k = false  then k := Length(path)+1;  fi;
        file := path{[i..k-1]};  Append( file, name );  Append( file, ext );
        InfoRead2("#I  trying '",file,"'\n");
        found := READ( file );
        i := k + 1;
    od;
    ReadIndent := readIndent;
    if found and ReadIndent = ""  then
        InfoRead1( "#I  ",infomsg,"( \"", name, "\" ) done\n" );
    fi;
    return found;
end;

Read := function ( name )
    if not ReadPath( "", name, "", "Read" )  then
     Error("the file '",name,"' must exist and be readable");
    fi;
end;

ReadLib := function ( name )
    if not ReadPath( LIBNAME, name, ".g", "ReadLib" )  then
     Error("the library file '",name,"' must exist and be readable");
    fi;
end;

GRPNAME := ReplacedString( LIBNAME, "lib", "grp" );

ReadGrp := function ( name )
    if not ReadPath( GRPNAME, name, ".grp", "ReadGrp" )  then
     Error("the group library file '",name,"' must exist and be readable");
    fi;
end;

TWONAME := ReplacedString( LIBNAME, "lib", "two" );

ReadTwo := function ( name )
    if not ReadPath( TWONAME, name, ".grp", "ReadTwo" )  then
     Error("the 2-group library file '",name,"' must exist and be readable");
    fi;
end;

THRNAME := ReplacedString( LIBNAME, "lib", "thr" );

ReadThr := function ( name )
    if not ReadPath( THRNAME, name, ".grp", "ReadThr" )  then
     Error("the 3-group library file '",name,"' must exist and be readable");
    fi;
end;

SMLNAME := ReplacedString( LIBNAME, "lib", "sml" );

ReadSml := function ( name )
    if not ReadPath( SMLNAME, name, "", "ReadSml" )  then
        Error("the group table file '",name,"' must exist and be readable");
    fi;
end;

TBLNAME := ReplacedString( LIBNAME, "lib", "tbl" );

ReadTbl := function ( name )
    if not ReadPath( TBLNAME, name, ".tbl", "ReadTbl" )  then
     Error("the character table file '",name,"' must exist and be readable");
    fi;
end;

TOMNAME := ReplacedString( LIBNAME, "lib", "tom" );

ReadTom := function ( name )
    if not ReadPath( TOMNAME, name, ".tom", "ReadTom" )  then
     Error("the table of marks file '",name,"' must exist and be readable");
    fi;
end;


#N  1995/11/26 mschoene this should be fixed in the kernel
OnLeftAntiOperation := OnLeft;
Unbind( OnLeft );
OnLeftInverse := function ( pnt, g ) return g^-1 * pnt; end;

#N  1995/12/19 mschoene this should be fixed in the kernel
OLDBlistList := BlistList;
_UnprotectVar(BlistList);
BlistList := function ( list, sub )
    if IsSet( sub )  then
        return OLDBlistList( list, sub );
    else
        return OLDBlistList( list, Set(sub) );
    fi;
end;


# Changes to avoid errors wenn calls to sml/, tbl/, thr/, tom/, two/
# are done, this is preferred rather than changing source code in lib/
CharTableLibrary := Ignore;
BrauerTable := Ignore;


AUTO( ReadLib( "abattoir" ),
  LengthString, SubString, ConcatenationString, Edit, ProductPol, ValuePol,
  MergedRecord, UnionBlist, IntersectionBlist, DifferenceBlist, SetPrintLevel,
  Save, SetPkgname, PKGNAME, LOADED_PACKAGES, ReadPkg, ExecPkg, LoadPackage,
  RequirePackage, IsOperationsRecord, OpsOps, OperationsRecord, EXEC,
  False, True, PrintFactorsInt );

AUTO( ReadLib("sys_conf"),
    Conf );

AUTO( ReadLib( "agcent" ),
  MainEntryCSAgGroup, CentralCaseCentAgGroup, GeneralCaseCentAgGroup,
  MainEntryCentAgGroup, CentralCaseECAgWords, GeneralCaseECAgWords,
  MainEntryECAgWords, MainEntryACAgWords );

AUTO( ReadLib( "agclass" ),
  InHomSolutions, CommutatorGauss, CorrectedStabilizer, AffineOrbitsAgGroup,
  MinimalVectorAgOrbit, CentralCaseECSAgGroup, GeneralCaseECSAgGroup,
  MainEntryECSAgGroup, FusionsECSAgGroup, Fusions2ECSAgGroup,
  SubEntryNECSAgGroup, MainEntryNECSAgGroup, SubEntrySECSAgGroup,
  MainEntrySECSAgGroup, StructureConjugacyClasses, ConjugacyClassAgGroupOps );

AUTO( ReadLib( "agcomple" ),
  BaseSteinitz, AffineBlocksCO, NextCentralizerCO, NextCocyclesCO,
  NextCentralCO, NextNormalCO, NextComplementsCO, ComplementsCO,
  ComplementsCO2, Complementclasses, Complement );

AUTO( ReadLib( "agcoset" ),
  RightCosetAgGroupOps, LeftCosetAgGroupOps, AGDoubleCosets, FactorAgSubgroup,
  ElementVector, MainEntryCCEAgGroup, GenRelOrdersAgGroup );

AUTO( ReadLib( "agctbl" ),
  AgGroupClassMatrixColumn, IdentificationAgGroup, InitAgConjugacyTest );

AUTO( ReadLib( "aggroup" ),
  AgGroupOps, AgSubgroup, ChangeCollector, RefinedAgSeries, SiftedAgWord,
  Exponents, FactorArgOps, FactorArg, AgGroupPcp, MatGroupAgGroup,
  PermGroupAgGroup, AgGroupFpGroup, DirectProductAgGroupOps,
  SemidirectProductAgGroupOps, SemidirectMatProduct, CollectorlessFactorGroup,
  CanonicalAgWord7, CanonicalAgWord4, CanonicalAgWord5, CanonicalAgWord6a,
  CanonicalAgWord6b, CayleyInputAgGroup, GapInputAgGroup, SogosInputAgGroup,
  CGSInputAgGroup, MergeOperationsEntries );

AUTO( ReadLib( "aghall" ),
  GS_LIMIT, ConjugatingWordGS, ConjugatingWordCN, ComplementConjugatingAgWord,
  CoprimeComplement, HallEAS, HallComposition, HallSubgroup,
  HallConjugatingAgWord );

AUTO( ReadLib( "aghomomo" ),
  AbstractIgs, HomomorphicIgs, KernelHomomorphismAgGroupPermGroup,
  KernelHomomorphismAgGroupAgGroup, AgGroupHomomorphismOps,
  AgGroupHomomorphismByImagesOps, CompositionHomomorphismOps,
  CompositionFactorGroup, HomomorphismsSeries, IsomorphismAgGroup );

AUTO( ReadLib( "aginters" ),
  GS_SIZE, ExtendedIntersectionSumAgGroup, SumFactorizationFunctionAgGroup,
  GlasbyCover, GlasbyShift, GlasbyStabilizer, IntersectionSumAgGroup,
  SumAgGroup );

AUTO( ReadLib( "agnorm" ),
  NormalizeIgsMod, NormalizeIgsModLess, StabilizerOp1, StabilizerOp2,
  Stabilizer1NO, Stabilizer2NO, StabilizerNO, GlasbyNO, AbstractBaseMat,
  CoboundsNO, LinearNO, NormalizerNO, ConjugacyClassSubgroupsAgGroupOps );

AUTO( ReadLib( "agprops" ),
  IsAgGroup, IsElementAgSeries, IsConsistent, IsElementaryAbelianAgSeries,
  PiPowerSubgroupAgGroup, IsPNilpotent, FactorsAgGroup,
  SmallGeneratingSetAgGroup );

AUTO( ReadLib( "agsubgrp" ),
  MergedIgs, MergedCgs, Igs, Cgs, Normalize, CopyAgGroup, PRump,
  CompositionSubgroup, MeltElementaryAbelianSeriesAgGroup,
  ElementaryAbelianSeriesThrough, RefinedSubnormalSeries, AgOrbitStabilizer,
  LinearOperation, AffineOperation );

AUTO( ReadLib( "agwords" ),
  WordList, LetterInt, AgWords, AgWordsOps, CanonicalAgWord, CentralWeight,
  CompositionLength, Depth, LeadingExponent, MappedAgWord, RelativeOrder,
  FactorGroupAgWord, FactorGroupAgWordOps );

AUTO( ReadLib( "algebra" ),
  IsAlgebraElement, AlgebraElementsOps, AlgebraElements, AlgebraElementOps,
  AlgebraString, IsAlgebra, IsUnitalAlgebra, AlgebraOps, Algebra,
  UnitalAlgebra, Subalgebra, UnitalSubalgebra, TrivialSubalgebra, AsAlgebra,
  AsUnitalAlgebra, MaintainedAlgebraInfo, AsSubalgebra, AsUnitalSubalgebra,
  IsSubalgebra, UnitalAlgebraOps );

AUTO( ReadLib( "algfac" ),
  TragerNorm, TragerFact, ChaNuPol, AlgebraicPolynomialModP, UPrep,
  TransferedExtensionPol, EuclideanLattice, OrthogonalityDefect,
  AlgExtSquareHensel, DecomPoly );

AUTO( ReadLib( "algfld" ),
  AlgebraicExtension, QuotRemPolList, NewDenominator,
  RationalsAlgebraicExtensionsPolynomialOps,
  AlgebraicExtensionsPolynomialRingOps,
  RationalsAlgebraicExtensionsPolynomialRingOps,
  FiniteFieldAlgebraicExtensionsPolynomialOps,
  FiniteFieldAlgebraicExtensionsPolynomialRingOps, IsAlgebraicElement,
  AlgebraicExtensionElmOps, AlgebraicExtensionsOps, IsAlgebraicExtension,
  MinpolFactors, IsNormalExtension, GaloisMappingOps, ExtensionAutomorphism,
  RationalsAlgebraicExtensionsOps, FiniteFieldAlgebraicExtensionsOps,
  AlgExtElm, AlgExtInvElm, RationalsAlgebraicExtensionElmOps,
  FiniteFieldAlgebraicExtensionElmOps, RootOf, DefectApproximation,
  GaloisType );

AUTO( ReadLib( "algfp" ),
  IsFpAlgebraElement, MappedExpression, FFLISTS, FFList, ElementAlgebra,
  NumberAlgebraElement, FpAlgebraElementOps, FpAlgebraElement, FpAlgebraOps,
  FpAlgebraElementsOps, IsFpAlgebra, FreeAlgebra, FpAlgebra,
  PrintDefinitionFpAlgebra );

AUTO( ReadLib( "alghomom" ),
  IsAlgebraHomomorphism, IsUnitalAlgebraHomomorphism,
  KernelAlgebraHomomorphism, AlgebraHomomorphismOps,
  CompositionAlgebraHomomorphismOps, IdentityAlgebraHomomorphismOps,
  AlgebraHomomorphismByImagesOps, AlgebraHomomorphismByImages,
  UnitalAlgebraHomomorphismOps, OperationHomomorphismAlgebraOps,
  OperationHomomorphismUnitalAlgebraOps );

AUTO( ReadLib( "algmat" ),
  IsMatAlgebra, MatAlgebraOps, Fingerprint, Nullity, BasisMatAlgebraOps,
  SemiEchelonBasisMatAlgebraOps, StandardBasisMatAlgebraOps, MatAlgebra,
  EmptyMat, NullAlgebraOps, NullAlgebra, IsNullAlgebra, UnitalMatAlgebraOps );

AUTO( ReadLib( "cdaggrp" ),
  con_col_list, f2_orbit_priv, f2_orbits_priv, ls_orbit_priv, ls_orbits_priv,
  omega_1_priv, kernel_priv_ag_char, ProjectiveCharDegAgGroup, CharDegAgGroup,
  char_sec_prev, char_sec, CharTableSSGroup );

AUTO( ReadLib( "chartabl" ),
  CharTableOps, PrintCharTable, PreliminaryLatticeOps, BrauerTableOps );

AUTO( ReadLib( "classfun" ),
  IsClassFunction, ClassFunctionsOps, ClassFunctions, ClassFunctionOps,
  ClassFunction, GlobalPartitionClasses, PermClassesHomomorphism,
  VirtualCharacterOps, VirtualCharacter, IsVirtualCharacter, CharacterOps,
  Character, IsCharacter, NormalSubgroupClasses, ClassesNormalSubgroup,
  FactorGroupNormalSubgroupClasses, CharacterString, Irr, InertiaSubgroup,
  OrbitChar, OrbitsCharacters, OrbitRepresentativesCharacters );

AUTO( ReadLib( "combinat" ),
  Factorial, Binomial, Bell, Stirling1, Stirling2, CombinationsA,
  CombinationsK, Combinations, NrCombinationsK, NrCombinations, ArrangementsA,
  ArrangementsK, Arrangements, NrArrangementsA, NrArrangementsK,
  NrArrangements, UnorderedTuplesK, UnorderedTuples, NrUnorderedTuples,
  TuplesK, Tuples, NrTuples, PermutationsListK, PermutationsList,
  NrPermutationsList, DerangementsK, Derangements, NrDerangementsK,
  NrDerangements, Permanent2, Permanent, PartitionsSetA, PartitionsSetK,
  PartitionsSet, NrPartitionsSet, PartitionsA, PartitionsK, Partitions,
  NrPartitions, OrderedPartitionsA, OrderedPartitionsK, OrderedPartitions,
  NrOrderedPartitions, RestrictedPartitionsA, RestrictedPartitionsK,
  RestrictedPartitions, NrRestrictedPartitionsK, NrRestrictedPartitions,
  SignPartition, AssociatedPartition, PowerPartition, PartitionTuples, Lucas,
  Fibonacci, Bernoulli2, Bernoulli );

AUTO( ReadLib( "ctautoms" ),
  FamiliesOfRows, MatAutomorphismsFamily, MatAutomorphisms,
  TableAutomorphisms, TransformingPermutationFamily, TransformingPermutations,
  TransformingPermutationsCharTables );

AUTO( ReadLib( "ctbasic" ),
  IsCharTable, InitClassesCharTable, InverseClassesCharTable, PrintToCAS,
  TestCharTable, ClassNamesCharTable, CharTable, DisplayCharTable,
  ClassMultCoeffCharTable, ClassStructureCharTable,
  MatClassMultCoeffsCharTable, RealClassesCharTable, ClassOrbitCharTable,
  NrPolyhedralSubgroups, ClassRootsCharTable, SortCharactersCharTable,
  SortClassesCharTable, SortCharTable );

AUTO( ReadLib( "ctcharac" ),
  KernelChar, DeterminantChar, CentreChar, CentralChar, ScalarProduct,
  MatScalarProducts, InverseMatMod, PadicCoefficients,
  LinearIndependentColumns, DecompositionInt, IntegralizedMat, Decomposition,
  DecompositionMatrix, LaTeXStringDecompositionMatrix, Tensored, Reduced,
  ReducedOrdinary, Symmetrisations, Symmetrizations, SymmetricParts,
  AntiSymmetricParts, MinusCharacter, RefinedSymmetrisations,
  OrthogonalComponents, SymplecticComponents, PrimeBlocks,
  IrreducibleDifferences, CoefficientTaylorSeries, SummandMolienSeries,
  MolienSeriesOps, MolienSeries, ValueMolienSeries );

AUTO( ReadLib( "ctclfhlp" ),
  WhichClm, Findmi, Findeo1, o1m1, FindRow, CompleteClm, CompleteRows, NUMBER,
  CentralizerOrbits, CentralizerOrbitsMultiple,
  AdaptCOMatricesToCliffordTable );

AUTO( ReadLib( "ctcliffo" ),
  IsCliffordRec, CliffordRecordOps, TestCliffordRec, DetermFusions, ClmInit,
  CliffordRecords, PrintCliffordRec, IsCliffordTable, PrintCliffordTable,
  MakeHead, CliffordTableOps, CliffordTable, SplitClass, SplitCliffordTable );

AUTO( ReadLib( "ctfilter" ),
  StepModGauss, ModGauss, ContainedDecomposables, ContainedCharacters );

AUTO( ReadLib( "ctgapmoc" ),
  FieldInfo, Subfields, MAKElb11, StructureConstants, PowerInfo, ScanMOC,
  MOCChars, GAPChars, MOCTableOps, MOCTable, MOCTable0, MOCTableP, PrintToMOC );

AUTO( ReadLib( "ctgeneri" ),
  CharTableRegular, CharTableDirectProduct, CharTableFactorGroup,
  CharTableNormalSubgroup, CharTableSplitClasses, CharTableCollapsedClasses,
  CharTableIsoclinic, CharTableQuaternionic, GEN_Q_P, PrimeBase,
  CharTableSpecialized );

AUTO( ReadLib( "ctlattic" ),
  LLLReducedBasis, LLLReducedGramMat, LLL, ShortestVectors, Extract,
  Decreased, OrthogonalEmbeddings, OrthogonalEmbeddingsSpecialDimension,
  DnLattice, DnLatticeIterative, LLLint );

AUTO( ReadLib( "ctmapcon" ),
  CharString, UpdateMap, NonnegIntScalarProducts, IntScalarProducts,
  ContainedSpecialVectors, ContainedPossibleCharacters,
  ContainedPossibleVirtualCharacters, InitFusion, CheckPermChar, ImproveMaps,
  CommutativeDiagram, CheckFixedPoints, TransferDiagram, TestConsistencyMaps,
  InitPowermap, Congruences, ConsiderKernels, ConsiderSmallerPowermaps,
  PowermapsAllowedBySymmetrisations, Powermap, ConsiderTableAutomorphisms,
  OrbitFusions, OrbitPowermaps, RepresentativesFusions,
  RepresentativesPowermaps, FusionsAllowedByRestrictions, SubgroupFusions );

AUTO( ReadLib( "ctmapusi" ),
  InverseMap, CompositionMaps, ProjectionMap, Indeterminateness,
  PrintAmbiguity, Parametrized, ContainedMaps, Indirected, GetFusionMap,
  StoreFusion, ElementOrdersPowermap, Restricted, Inflated, Induced,
  CollapsedMat, Powmap, InducedCyclic, Power, Indicator );

AUTO( ReadLib( "ctpermch" ),
  SubClass, TestPerm1, TestPerm2, TestPerm3, Inequalities, Permut, PermBounds,
  PermComb, PermCandidates, PermCandidatesFaithful, PermChars, PermCharInfo );

AUTO( ReadLib( "ctpgrp" ),
  RepresentationsPGroup, MatRepresentationsPGroup, CharTablePGroup );

AUTO( ReadLib( "ctsymmet" ),
  BetaSet, CentralizerWreath, PowerWreath, InductionScheme,
  MatCharsWreathSymmetric, CharValueSymmetric, CharTableSymmetric,
  CharTableAlternating, CharValueWeylB, CharTableWeylB, CharTableWeylD,
  CharValueWreathSymmetric, CharTableWreathSymmetric );

AUTO( ReadLib( "cyclotom" ),
  IntCyc, RoundCyc, CoeffsCyc, CycList, Atlas1, EB, EC, ED, EE, EF, EG, EH,
  NK, Atlas2, EY, EX, EW, EV, EU, ET, ES, EM, EL, EK, EJ, ER, EI, StarCyc,
  Quadratic, GeneratorsPrimeResidues, GaloisMat, RationalizedMat );

AUTO( ReadLib( "dispatch" ),
  AbelianInvariants, AbsoluteIrreducibilityTest, AsRing, AutomorphismGroup,
  AsVectorSpace, CanonicalBasis, CanonicalRepresentative, Centre,
  CharacterDegrees, ChiefSeries, CommutatorFactorGroup, CompositionFactors,
  CompositionSeries, ConjugacyClasses, ConjugacyClassesPerfectSubgroups,
  ConjugacyClassesSubgroups, Constituents, DerivedSeries, DerivedSubgroup,
  Dimension, Exponent, FittingSubgroup, FpGroup, FrattiniSubgroup,
  GaloisGroup, Generators, IdentityMapping, InvariantSubspace, InverseMapping,
  IrreducibilityTest, IsAbelian, IsAutomorphism, IsBijection, IsBijective,
  IsCyclic, IsElementaryAbelian, IsEndomorphism, IsEpimorphism, IsFaithful,
  IsHomomorphism, IsInjective, IsIsomorphism, IsMonomorphism, IsNilpotent,
  IsNormalized, IsParent, IsPerfect, IsSimple, IsSolvable, IsSurjective,
  IsTrivial, KernelGroupHomomorphism, Lattice, LatticeSubgroups,
  LowerCentralSeries, MaximalNormalSubgroups, MaximalElement,
  MinimalGeneratingSet, NormalSubgroups, Normalized, Omega, One, Radical,
  RationalClasses, Representative, RepresentativesPerfectSubgroups,
  SizesConjugacyClasses, SmallestGenerators, SupersolvableResiduum,
  SylowComplements, SylowSystem, TrivialSubgroup, UpperCentralSeries, Zero,
  Determinant, Dimensions, Rank, Transposed, IsMonomial, Components, Basis,
  StandardBasis, Display, IsIrreducible, IsEquivalent, Kernel,
  FusionConjugacyClasses, KroneckerProduct, Closure, Centralizer, IsCentral,
  Base, IsZero, Eigenvalues );

AUTO( ReadLib( "domain" ),
  IsDomain, Domain, DomainOps, Elements, IsFinite, Size, IsSubset,
  Intersection, IntersectionSet, Union, UnionSet, CartesianProduct,
  Difference, Random, DefineName );

AUTO( ReadLib( "field" ),
  IsField, FieldOps, Conjugates, Norm, Trace, CharPol, MinPol,
  FieldElementsOps, FieldElements, Field, DefaultField, IsFieldHomomorphism,
  KernelFieldHomomorphism, FieldHomomorphismOps,
  CompositionFieldHomomorphismOps, IdentityFieldHomomorphismOps );

AUTO( ReadLib( "finfield" ),
  IsFiniteField, FiniteFieldOps, OrderFFE, GF, FiniteField, GaloisField,
  FiniteFieldElementsOps, FiniteFieldElements, IsBaseFF,
  IsFrobeniusAutomorphism, FrobeniusAutomorphism, FrobeniusAutomorphismI,
  FrobeniusAutomorphismOps );

AUTO( ReadLib( "fpgrp" ),
  FreeGroup, Words, WordsOps, IsFpGroup, FpGroupOps, RelatorRepresentatives,
  RelsSortedByStartGen, SortRelsSortedByStartGen, CosetTableFpGroup,
  MostFrequentGeneratorFpGroup, GeneratorsCosetTable, OperationCosetsFpGroup,
  FpGroupHomomorphismByImagesOps, IsIdenticalPresentationFpGroup,
  LowIndexSubgroupsFpGroup );

AUTO( ReadLib( "fpsgpres" ),
  PresentationSubgroupMtc, AugmentedCosetTableMtc, CheckCosetTableFpGroup,
  IsStandardized, PresentationSubgroupRrs, PresentationSubgroup,
  PresentationNormalClosureRrs, PresentationNormalClosure,
  AugmentedCosetTableRrs, SpanningTree, RenumberTree, RewriteSubgroupRelators,
  PresentationAugmentedCosetTable, AbelianInvariantsSubgroupFpGroupMtc,
  RelatorMatrixAbelianizedSubgroupMtc, AbelianInvariantsSubgroupFpGroupRrs,
  AbelianInvariantsSubgroupFpGroup, RelatorMatrixAbelianizedSubgroupRrs,
  RelatorMatrixAbelianizedSubgroup, AbelianInvariantsNormalClosureFpGroupRrs,
  AbelianInvariantsNormalClosureFpGroup,
  RelatorMatrixAbelianizedNormalClosureRrs,
  RelatorMatrixAbelianizedNormalClosure, RewriteAbelianizedSubgroupRelators,
  CanonicalRelator, ReducedRrsWord );

AUTO( ReadLib( "fptietze" ),
  TZ_NUMGENS, TZ_NUMRELS, TZ_TOTAL, TZ_GENERATORS, TZ_INVERSES, TZ_RELATORS,
  TZ_LENGTHS, TZ_FLAGS, TZ_MODIFIED, TZ_NUMREDUNDS, TZ_STATUS,
  TZ_LENGTHTIETZE, TR_TREELENGTH, TR_PRIMARY, TR_TREENUMS, TR_TREEPOINTERS,
  TR_TREELAST, PresentationOps, AbstractWordTietzeWord, AddGenerator,
  AddRelator, DecodeTree, FpGroupPresentation, PresentationFpGroup,
  RelsViaCosetTable, PresentationViaCosetTable, RemoveRelator,
  SimplifiedFpGroup, TietzeWordAbstractWord, TzCheckRecord, TzEliminate,
  TzEliminateFromTree, TzEliminateGen, TzEliminateGen1, TzEliminateGens,
  TzFindCyclicJoins, TzGeneratorExponents, TzGo, SimplifyPresentation, TzGoGo,
  TzHandleLength1Or2Relators, TzInitGeneratorImages, TzMostFrequentPairs,
  TzNewGenerator, TzPrint, TzPrintGenerators, TzPrintGeneratorImages,
  TzPrintLengths, TzOptionNames, TzRecordOps, TzPrintOptions, TzPrintPairs,
  TzPrintPresentation, TzPrintRelators, TzPrintStatus, TzRecoverFromFile,
  TzRemoveGenerators, TzSearch, TzSearchEqual, TzSort, TzSubstitute,
  TzSubstituteCyclicJoins, TzSubstituteWord, TzUpdateGeneratorImages );

AUTO( ReadLib( "galois" ),
  RevSortFun, PowersumsElsyms, ElsymsPowersums, SumRootsPolComp, SumRootsPol,
  ProductRootsPol, Tschirnhausen, TwoSeqPol, SetResolvent, DiffResolvent,
  PartitionsTest, UnParOrbits, GetProperty, ShapeFrequencies,
  ProbabilityShapes, GrabCodedLengths, Galois );

AUTO( ReadLib( "gaussian" ),
  IsGaussInt, GaussianIntegersOps, GaussianIntegers, TwoSquares,
  GaussianIntegersAsAdditiveGroupOps, IsGaussRat, GaussianRationalsOps,
  GaussianRationals, GaussianRationalsAsRingOps );

AUTO( ReadLib( "group" ),
  GroupString, GroupOps, Group, AsGroup, IsGroup, Parent, MaintainedGroupInfo,
  Subgroup, AsSubgroup, CommutatorSubgroup, Core, NormalClosure,
  NormalIntersection, Normalizer, NumberConjugacyClasses, PCore,
  SylowSubgroup, ElementaryAbelianSeries, DisplayCompositionSeries,
  PCentralSeries, SubnormalSeries, IsConjugate, IsNormal, IsSubgroup,
  IsSubnormal, Index, Agemo, AgemoAbove, JenningsSeries,
  DimensionsLoewyFactors, IsomorphismTypeFiniteSimpleGroup, IsConjugacyClass,
  ConjugacyClassGroupOps, ConjugacyClass, IsRationalClass,
  RationalClassGroupOps, RationalClass, ConjugateSubgroup, ConjugateSubgroups,
  AbstractElementsGroup, Factorization, AgGroup, PermGroup,
  IrreducibleGeneratingSet );

AUTO( ReadLib( "grpcoset" ),
  Transversal, RightTransversal, LeftTransversal, IsRightCoset, IsCoset,
  RightCoset, Coset, RightCosetGroupOps, RightCosets, Cosets, IsLeftCoset,
  LeftCoset, LeftCosetGroupOps, LeftCosets, IsDoubleCoset, DoubleCoset,
  DoubleCosetGroupOps, DoubleCosets, CalcDoubleCosets, AscendingChain,
  RefinedChain, Extension, CanonicalRightTransversal, CanonicalCosetElement,
  OnCanonicalCosetElements, PermutationCharacter, IsFactorGroupElement,
  FactorGroupElement, FactorGroupElementOps, FactorGroupElements,
  FactorGroupElementsOps, FactorGroup, NaturalHomomorphism,
  NaturalHomomorphismOps, FactorGroupOps );

AUTO( ReadLib( "grpctbl" ),
  USECTPGROUP, IsLargeGroup, CharTableDixonSchneider, DixonRecord,
  DixonRecordOps, DixonInit, RegisterNewCharacter, DixontinI, SortDixonRecord,
  DixonSplit, OrbitSplit, CombinatoricSplit, SplitCharacters,
  IncludeIrreducibles, DxLinearCharacters, ClassComparison, DxCalcPowerMap,
  DxPowerClass, SplitStep, SplitTwoSpace, DxLiftCharacter,
  GeneratePrimeCyclotomic, ModProduct, ModularCharacterDegree,
  DegreeCandidates, FrobSchurInd, BestSplittingMatrix, SplitDegree,
  CharacterMorphismGroup, AsCharacterMorphismFunction,
  CharacterMorphismOrbits, GaloisOrbits, RootsOfPol, ModRoots,
  ModularValuePol, BMminpol, KrylovSequence, Eigenbase, ActiveCols, PadicInt,
  ClassElementLargeGroup, ClassElementSmallGroup, DoubleCentralizerOrbit,
  StandardClassMatrixColumn, IdentificationGenericGroup, DxAbelianPreparation,
  AbelianNormalSubgroups );

AUTO( ReadLib( "grpelms" ),
  IsGroupElement, GroupElements, GroupElementsOps, GroupElementOps, Order,
  LeftNormedComm, RightNormedComm );

AUTO( ReadLib( "grphomom" ),
  IsGroupHomomorphism, GroupHomomorphismOps, CompositionGroupHomomorphismOps,
  IdentityGroupHomomorphismOps, ConjugationGroupHomomorphism,
  ConjugationGroupHomomorphismOps, InnerAutomorphism,
  GroupHomomorphismByImages, GroupHomomorphismByImagesOps,
  GroupHomomorphismByFunction, GroupHomomorphismByFunctionOps,
  IsomorphismGroups );

AUTO( ReadLib( "grplatt" ),
  ShallowCopyNoSC, IsConjugacyClassSubgroups, ConjugacyClassSubgroups,
  ConjugacyClassSubgroupsGroupOps, LatticeSubgroupsOps,
  PrintClassSubgroupLattice, IsomorphismPerfectGroupHelp,
  IsomorphismPerfectGroup, Zuppos );

AUTO( ReadLib( "grpprods" ),
  IsDirectProductElement, DirectProductElement, DirectProductElementOps,
  IsDirectProduct, DirectProduct, DirectProductOps, EmbeddingDirectProductOps,
  ProjectionDirectProductOps, SubdirectProduct, SubdirectProductOps,
  ProjectionSubdirectProductOps, IsSemidirectProductElement,
  SemidirectProductElement, SemidirectProductElementOps, IsSemidirectProduct,
  SemidirectProduct, SemidirectProductOps, EmbeddingSemidirectProductOps,
  ProjectionSemidirectProductOps, IsWreathProductElement,
  WreathProductElement, WreathProductElementOps, IsWreathProduct,
  WreathProduct, WreathProductOps );

AUTO( ReadLib( "integer" ),
  IntegersOps, Integers, NrBitsInt, Primes, Primes2, TraceModQF, IsPrimeInt,
  IsPrimePowerInt, NextPrimeInt, PrevPrimeInt, FactorsRho, FactorsInt,
  DivisorsSmall, DivisorsInt, Sigma, Tau, MoebiusMu, CoefficientsQadic,
  PowerModInt, LcmInt, Gcdex, Int, AbsInt, SignInt, ChineseRem, LogInt,
  RootInt, SmallestRootInt, IntegersAsAdditiveGroupOps );

AUTO( ReadLib( "lattperf" ),
  PerfectGroupsCatalogue );

AUTO( ReadLib( "list" ),
  List, Apply, Concatenation, Flat, Reversed, Sublist, Filtered, Number,
  Compacted, Collected, Equivalenceclasses, ForAll, ForAny, First,
  PositionProperty, PositionBound, Cartesian2, Cartesian, Sort, SortParallel,
  Sortex, Permuted, PositionSorted, Product, Sum, Iterated, Maximum, Minimum,
  R_N, R_X, RandomList, RandomSeed, PositionSet, SortingPerm, PermListList );

AUTO( ReadLib( "mapping" ),
  IsGeneralMapping, IsMapping, Image, Images, ImagesRepresentative, PreImage,
  PreImages, PreImagesRepresentative, CompositionMapping, PowerMapping,
  MappingOps, CompositionMappingOps, InverseMappingOps, MappingByFunction,
  MappingByFunctionOps, Embedding, Projection, Mappings, MappingsOps );

AUTO( ReadLib( "matgrp" ),
  IsMatGroup, MatGroupOps, MatPermPNumVec, MatPermPVecNum, MatPermPPermMatrix,
  MatPermPMatrixPerm, PermGroupOnVectorspace, RightCosetMatGroupOps, MatGroup,
  RandomMatGroup, MTXOps, EquivalenceTest, NEXTMATRIX, AddNextMatrixFunction,
  NextMatrix, SmallCorankMatrixRecord, RandomMatrixRecord, FingerprintEx,
  ClassicNextMatrix, ExtendedClassicNextMatrix, NextMatrix1, NextMatrix2,
  NextMatrix3 );

AUTO( ReadLib( "matring" ),
  IsMatrixRing, MatrixRingOps );

AUTO( ReadLib( "matrix" ),
  IsMatrix, MatricesOps, Matrices, DimensionsMat, IdentityMat, NullMat,
  RandomMat, RandomInvertableMat, RandomUnimodularMat, TransposedMat,
  PermutationMat, InvariantForm, OrderMatLimit, OrderMat, DiagonalOfMat,
  TraceMat, RankMat, DeterminantMat, TriangulizeMat, BaseMat, SemiEchelonMat,
  InducedActionSpaceMats, SumIntersectionMat, BaseFixedSpace, NullspaceMat,
  NullspaceModQ, SimultaneousEigenvalues, BaseNullspace, BestQuoInt,
  DiagonalizeIntMatNormDriven, DiagonalizeIntMat, DiagonalizeMat,
  DiagonalFormMat, SmithNormalizeMat, SmithNormalFormMat,
  ElementaryDivisorsMat, ElementaryDivisorsOfList, AbelianInvariantsOfList,
  SolutionMat, FieldMatricesOps, FieldMatrices, MinimalPolynomial,
  CharacteristicPolynomial, FiniteFieldMatricesOps, FiniteFieldMatrices );

AUTO( ReadLib( "module" ),
  IsModule, IsFactorModule, ModuleOps, Module, NaturalModule, IsNaturalModule,
  Submodule, AsModule, AsSubmodule, IsAbsolutelyIrreducible,
  StandardBasisModuleOps, SpinUpStandard, OperationModule,
  OperationHomomorphismModuleOps, FactorModuleOps, FactorModule,
  FreeModuleOps, IsFreeModule, FreeModule, ModuleCosetOps, ModuleCoset,
  IsModuleCoset, FixedSubmodule );

AUTO( ReadLib( "monomial" ),
  Alpha, Delta, BergerCondition, TestHomogeneous, TestQuasiPrimitive,
  IsQuasiPrimitive, TestInducedFromNormalSubgroup,
  IsInducedFromNormalSubgroup, TestSubnormallyMonomial, IsSubnormallyMonomial,
  TestMonomialQuick, TestMonomial, TestRelativelySM, IsRelativelySM,
  IsMinimalNonmonomial, MinimalNonmonomialGroup );

AUTO( ReadLib( "morpheus" ),
  MORPHEUSELMS, MorFroWords, PermAutImg, PermAutomorphismGroupOps,
  MorRatClasses, MorMaxFusClasses, MorClassLoop, MorFindGeneratingSystem,
  Morphium, IndependentGeneratorsAbelianPPermGroup,
  IndependentGeneratorsAbelianPermGroup, AutomorphismGroupAbelianGroup,
  IsomorphismAbelianGroups );

AUTO( ReadLib( "numfield" ),
  IsNumberField, IsCyclotomicField, NumberFieldOps, IsNFAutomorphism,
  NFAutomorphism, NFAutomorphismOps, OrderCyc, NormalBaseNumberField,
  ZumbroichBase, LenstraBase, NF, NumberField, CyclotomicFieldOps, CF,
  CyclotomicField, NumberRingOps, NumberRing, CyclotomicRing, CyclotomicsOps,
  Cyclotomics );

AUTO( ReadLib( "numtheor" ),
  PrimeResiduesSmall, PrimeResidues, Phi, Lambda, OrderMod,
  IsPrimitiveRootMod, PrimitiveRootMod, Jacobi, Legendre, RootModPrime,
  RootModPrimePower, RootMod, RootsModPrime, RootsModPrimePower, RootsMod,
  RootsUnityModPrime, RootsUnityModPrimePower, RootsUnityMod, LogMod,
  IsResidueClass, ResidueClass, ResidueClassOps, ResidueClasses,
  ResidueClassesOps, ResidueClassGroupOps, PrimeResidueClassGroup );

AUTO( ReadLib( "onecohom" ),
  OCAgGroupOps, OCPermGroupOps, OneCoboundariesOC, ConjugatingWordOC,
  EquationMatrixOC, SmallEquationMatrixOC, EquationVectorOC,
  SmallEquationVectorOC, OneCocyclesOC, OneCoboundaries, OneCocycles,
  PPrimeSetsOC );

AUTO( ReadLib( "operatio" ),
  OnRightCosets, OnLeftCosets, OnLines, Cycle, CycleLength, Cycles,
  CycleLengths, Permutation, IsFixpoint, IsFixpointFree, DegreeOperation,
  IsTransitive, Transitivity, IsRegular, IsSemiRegular, Orbit, OrbitLength,
  Orbits, OrbitLengths, Operation, OperationHomomorphism,
  OperationHomomorphismOps, Blocks, MaximalBlocks, IsPrimitive, Stabilizer,
  RepresentativeOperation, RepresentativesOperation, IsEquivalentOperation );

AUTO( ReadLib( "permag" ),
  MaximalBlocksPGroup, OrderFactorGroupElement, InsertStabChain,
  ClosureNormalizingElementPermGroup, ExtendElementaryAbelianSeriesPermGroup,
  BaseStrongSubnormalGeneratingSetPPermGroup, ExponentsPermSolvablePermGroup,
  PcPresentationPermGroup, CompositionSeriesSolvablePermGroup,
  SubnormalSeriesPPermGroup, CentralCompositionSeriesPPermGroup );

AUTO( ReadLib( "permcose" ),
  RightCosetPermGroupOps, PermRefinedChain, MainEntryCCEPermGroup );

AUTO( ReadLib( "permcser" ),
  CompositionSeriesPermGroup, NonPerfectCSPG, PerfectCSPG, CasesCSPG,
  FindNormalCSPG, FindRegularNormalCSPG, NinKernelCSPG, RegularNinKernelCSPG,
  NormalizerStabCSPG, TransStabCSPG, PullbackKernelCSPG, PullbackCSPG,
  CosetRepAsWord, ImageInWord, SiftAsWord, InverseAsWord, RandomElmAsWord,
  CentralizerNormalCSPG, CentralizerNormalTransCSPG, CentralizerTransSymmCSPG,
  IntersectionNormalClosurePermGroup, ActionAbelianCSPG, ImageOnAbelianCSPG );

AUTO( ReadLib( "permctbl" ),
  IdentificationPermGroup, RationalIdentificationPermGroup, FingerprintPerm );

AUTO( ReadLib( "permgrp" ),
  IsPermGroup, PermGroupOps, ConjugacyClassPermGroupOps );

AUTO( ReadLib( "permhomo" ),
  PermGroupHomomorphismByImagesOps, CoKernelGensPermHom,
  PermGroupHomomorphismByImagesPermGroupOps, TransConstHomomorphismOps,
  BlocksHomomorphismOps );

AUTO( ReadLib( "permnorm" ),
  SortedOrbitsButler, SortedOrbitsPermGroup, BaseForNormalizerPermGroup,
  SigmaSets );

AUTO( ReadLib( "permprod" ),
  DirectProductPermGroupOps, DirectProductPermGroupCentre,
  DirectProductPermGroupSylowSubgroup, DirectProductPermGroupCentralizer,
  EmbeddingDirectProductPermGroupOps, ProjectionDirectProductPermGroupOps,
  SubdirectProductPermGroupOps, ProjectionSubdirectProductPermGroupOps );

AUTO( ReadLib( "permstbc" ),
  StabChainOptions, StabChain, SCRMakeStabStrong, SCRStrongGenTest, SCRSift,
  SCRStrongGenTest2, SCRNotice, SCRExtend, SCRSchTree, SCRRandomPerm,
  SCRRandomString, SCRRandomSubproduct, SCRExtendRecord, SCRRestoreRecord,
  SC_level, BaseStabChain, MakeStabChain, MakeStabChainRandom,
  MakeStabChainStrongGenerators, ReduceStabChain, ExtendStabChain,
  ListStabChain );

AUTO( ReadLib( "permutat" ),
  PermutationsOps, Permutations, CycleStructurePerm, SmallestMovedPointPerm,
  ListPerm, MappingPermListList, RestrictedPerm );

AUTO( ReadLib( "polyfin" ),
  FiniteFieldPolynomialRingOps, FiniteFieldLaurentPolynomials,
  FiniteFieldLaurentPolynomialsOps, FiniteFieldPolynomials,
  FiniteFieldPolynomialsOps, FiniteFieldPolynomialOps, PrimePowersInt,
  ProductPP, LcmPP, OrderKnownDividendList, OKDInd );

AUTO( ReadLib( "polyfld" ),
  FieldPolynomialRingOps, FieldLaurentPolynomialRingOps );

AUTO( ReadLib( "polynom" ),
  Polynomial, IsPolynomial, CompanionMatrix, IsPrimitivePolynomial,
  PowerModEvalPol, CONWAYPOLYNOMIALS, ConwayPol, ConwayPolynomial,
  CYCLOTOMICPOLYNOMIALS, CyclotomicPol, CyclotomicPolynomial,
  EmbeddedPolynomial, RandomPol, Value, Indeterminate, X,
  InterpolatedPolynomial, LaurentPolynomialRing, IsLaurentPolynomialRing,
  LaurentPolynomialRingOps, Derivative, PolynomialRing, IsPolynomialRing,
  PolynomialRingOps, LaurentPolynomials, LaurentPolynomialsOps, Polynomials,
  PolynomialsOps, PolynomialOps, Degree, LeadingCoefficient,
  DisplayPolynomial, LaurentDegree );

AUTO( ReadLib( "polyrat" ),
  RationalsPolynomialOps, RationalsPolynomials, RationalsPolynomialsOps,
  BeauzamyBoundGcd, TryCombinations, FactorsOptions );

AUTO( ReadLib( "polystff" ),
  APolyProd, APolyMod, BPolyProd, ProductMod, RootRat, PseudoRemainder,
  RingElDegZeroPol, Resultant, Discriminant, MonicIntegerPolynomial,
  ApproxRational, ApproximateRoot, ApproxRootBound, RootBound, BombieriNorm,
  MinimizeBombieriNorm, BeauzamyBound, OneFactorBound, HenselBound, CoeffAbs,
  TrialQuotient, Characteristic, RandomNormedPol, PolynomialModP, ContentPol,
  ParityPol, EvalF, CheapFactorsInt, Berwick );

AUTO( ReadLib( "pq" ),
  PQpOps, PQp, SavePQp, InitPQp, AddGeneratorsPQp, DefineGeneratorsPQp,
  TailsPQp, EchelonizePQp, ConsistencyPQp, ElimTailsPQp, LiftHomomorphismPQp,
  CleanUpPQp, FirstClassPQp, PQuotient, pQuotient, PrimeQuotient,
  NextClassPQp, Weight );

AUTO( ReadLib( "ratclass" ),
  InjectionPrimeResidues, GroupsPrimeResidues, AgGroupPrimeResiduesOps,
  GroupPrimeResidues, ResidueAgWord, AgWordResidue,
  AgOrbitTransversalStabilizer, RationalClassPermGroupOps,
  RepresentativeRatConjElmsCoset, CompleteGaloisGroupPElement,
  OrbitsVectorSpace, ConstructList, SubspaceVectorSpaceAgGroup,
  PreImageProjectionSubspace, CentralStepConjugatingElement,
  CentralStepRationalClasses2Group, CentralStepIdentifyRationalClasses2Group,
  CentralStepRationalClassesPGroup, CentralStepIdentifyRationalClassesPGroup,
  SortRationalClasses, RationalClassesPAgGroup,
  RationalClassesElementaryAbelianSubgroup, FusionRationalClassesPSubgroup,
  RationalClassesAgGroup, RationalClassesPermGroup, RationalClassesPElements );

AUTO( ReadLib( "rational" ),
  RationalsOps, Rationals, RationalsAsRingOps );

AUTO( ReadLib( "ring" ),
  IsRing, RingOps, IsCommutativeRing, IsIntegralRing,
  IsUniqueFactorizationRing, IsEuclideanRing, Quotient, IsUnit, Units,
  IsAssociated, StandardAssociate, Associates, IsPrime, Factors,
  EuclideanDegree, EuclideanRemainder, EuclideanQuotient, QuotientRemainder,
  Mod, QuotientMod, PowerMod, Gcd, GcdRepresentation, Lcm, RingElementsOps,
  RingElements, Ring, DefaultRing );

AUTO( ReadLib( "rowmodul" ),
  RowModuleOps, RowModule, IsRowModule, Representation, ProperSubmodule );

AUTO( ReadLib( "rowspace" ),
  IntegerTable, RowSpace, IsRowSpace, Subspace, AsSubspace, AsSpace,
  RowSpaceOps, BaseTypeRowSpace, SiftedVector, NormedVectors, DualRowSpace,
  IsSemiEchelonBasis, BasisRowSpaceOps, SemiEchelonBasis,
  SemiEchelonBasisRowSpaceOps, CanonicalBasisRowSpaceOps, NumberVector,
  ElementRowSpace, QuotientRowSpaceOps, BasisQuotientRowSpaceOps,
  SemiEchelonBasisQuotientRowSpaceOps, CanonicalBasisQuotientRowSpaceOps,
  SpaceCoset, SpaceCosetRowSpaceOps, ModspaceOps );

AUTO( ReadLib( "saggroup" ),
  SagWeights, SagGroupOps, SpecialAgGroup );

AUTO( ReadLib( "sagsbgrp" ),
  ModuleDescrSagGroup, MatGroupSagGroup, DualModuleDescrSagGroup,
  DualMatGroupSagGroup, ConjugacyClassesMaximalSubgroups, PrefrattiniSubgroup,
  MaximalSubgroups, EulerianFunction, SystemNormalizer );

AUTO( ReadLib( "sq" ),
  SQOps, IntersectionMat, CollectedWordSQ, CollectorSQ, AddEquationsSQ,
  SolutionSQ, TwoCocyclesSQ, TwoCoboundariesSQ, ExtensionSQ, InitSQ,
  NextModuleSQ, NextPrimesSQ, SolvableQuotient );

AUTO( ReadLib( "sqstuff" ),
  mapSQ, convertSQ, QuotientSpaceSQ, pfactorsSQ, gaussSQ, minvSQ, ConjugateSQ,
  BlowupFieldSQ, distinguishSQ, IsEquivalentSQ, conjugateRepSQ, induceSQ,
  AlgConjugatesSQ, BlowupSQ, intertwineSQ, ModulesSQ, ModulesAgGroup,
  intmatdiagSQ, InitEpimorphismSQ, MakePreImagesSQ, LiftEpimorphismSQ );

AUTO( ReadLib( "string" ),
  StringInt, StringRat, StringCyc, StringFFE, StringPerm, StringAgWord,
  StringBool, StringList, StringRec, String, PrintArray, PrintRecIgnore,
  PrintRecIndent, RecordOps, PrintRec, DaysInYear, DaysInMonth, DMYDay,
  DayDMY, NameWeekDay, WeekDay, NameMonth, StringDate, HMSMSec, SecHMSM,
  StringTime, StringPP, Ordinal, WordAlp, MkString );

AUTO( ReadLib( "tom" ),
  IsTom, TableOfMarks, Marks, NrSubs, WeightsTom, TomMat, MatTom,
  DecomposedFixedPointVector, TestRow, TestTom, DisplayTom, NormalizerTom,
  IntersectionsTom, IsCyclicTom, PermCharsTom, FusionCharTableTom, MoebiusTom,
  CyclicExtensionsTom, IdempotentsTom, ClassTypesTom, ClassNamesTom,
  TomCyclic, TomDihedral, TomFrobenius );

AUTO( ReadLib( "vecspace" ),
  VectorSpace, IsVectorSpace, VectorSpaceOps, IsSubspace, AddBase,
  Information, Coefficients, LinearCombination, Enumeration, LineEnumeration,
  NormedVector, IsSpaceCoset, IsQuotientSpace, QuotientSpaceOps, STMappingOps,
  STMapping );

AUTO( ReadGrp( "basic" ),
  CyclicGroup, AbelianGroup, ElementaryAbelianGroup, DihedralGroup,
  PolyhedralGroup, AlternatingGroup, SymmetricGroup, GeneralLinearGroup,
  SpecialLinearGroup, SymplecticGroup, GeneralUnitaryGroup,
  SpecialUnitaryGroup, GL, SL, GU, SU, SP, ExtraspecialGroup );

AUTO( ReadGrp( "cryst" ),
  GLZOps, GLZ2, GLZ3, GLZ4, CR_TextStrings, CR_2, CR_3, CR_4,
  CrystGroupsCatalogue, CR_AgGroupQClass, CR_CharTableQClass,
  CR_DisplayQClass, CR_DisplaySpaceGroupGenerators, CR_DisplaySpaceGroupType,
  CR_DisplayZClass, CR_FpGroupQClass, CR_GeneratorsSpaceGroup,
  CR_GeneratorsZClass, CR_MatGroupZClass, CR_Name, CR_NormalizerZClass,
  CR_Parameters, CR_SpaceGroup, CR_ZClassRepsDadeGroup, AgGroupQClass,
  CharTableQClass, DisplayCrystalFamily, DisplayCrystalSystem, DadeGroup,
  DadeGroupNumbersZClass, DisplayQClass, DisplaySpaceGroupGenerators,
  DisplaySpaceGroupType, DisplayZClass, FpGroupQClass, MatGroupZClass,
  NormalizerZClass, NrCrystalFamilies, NrCrystalSystems, NrDadeGroups,
  NrQClassesCrystalSystem, NrSpaceGroupTypesZClass, NrZClassesQClass,
  SpaceGroup, TransposedSpaceGroup, ZClassRepsDadeGroup,
  CR_InitializeRelators );

AUTO( ReadGrp( "groupid" ),
  NrElementsOfOrder, EncodedStandardPresentation, GroupIdOps, GroupId );

AUTO( ReadGrp( "imf" ),
  BaseShortVectors, DisplayImfInvariants, DisplayImfReps, ImfInvariants,
  IMFLoad, ImfMatGroup, ImfMatrixToPermutation, ImfNumberQClasses,
  ImfNumberQQClasses, ImfNumberZClasses, ImfPermutationToMatrix,
  ImfPositionNumber, OrbitShortVectors, PermGroupImfGroup );

AUTO( ReadGrp( "imf0" ),
  IMFRec, IMFList );

AUTO( ReadGrp( "irredsol" ),
  IrredSolJSGens, IrredSolGroupList, IrreducibleSolvableGroup,
  IsLinearlyPrimitive, MinimalBlockDimension, AllIrreducibleSolvableGroups,
  OneIrreducibleSolvableGroup, PrimitivePermGroupIrreducibleMatGroup );

AUTO( ReadGrp( "matgrp" ),
  MatGroupLib, SpecialLinearMatGroup, GeneralLinearMatGroup,
  SymplecticMatGroup, GeneralUnitaryMatGroup, SpecialUnitaryMatGroup );

AUTO( ReadGrp( "perf" ),
  DisplayInformationPerfectGroups, NumberPerfectGroups,
  NumberPerfectLibraryGroups, PERFLoad, PerfectCentralProduct, PerfectGroup,
  PerfectSubdirectProduct, PermGroupPerfectGroup,
  PermGroupPerfectSubdirectProduct, SizeNumbersPerfectGroups,
  SizesPerfectGroups );

AUTO( ReadGrp( "perf0" ),
  PERFRec, PERFFun );

AUTO( ReadGrp( "permgrp" ),
  PermGroupLib, CyclicPermGroup, AbelianPermGroup, ElementaryAbelianPermGroup,
  DihedralPermGroup, PolyhedralPermGroup, AlternatingPermGroup,
  AlternatingPermGroupOps, SymmetricPermGroup, SymmetricPermGroupOps,
  GeneralLinearPermGroup, SpecialLinearPermGroup, SymplecticPermGroup,
  GeneralUnitaryPermGroup, SpecialUnitaryPermGroup );

AUTO( ReadGrp( "primitiv" ),
  PGGens, FAC, PGTable, PG, PrimitiveGroup, AllPrimitiveGroups,
  OnePrimitiveGroup );

AUTO( ReadGrp( "solvable" ),
  AGTable, AGGroup, SolvableGroup, IsNontrivialDirectProduct,
  AllSolvableGroups, OneSolvableGroup );

AUTO( ReadGrp( "sporadic" ),
  MathieuGroup );

AUTO( ReadGrp( "trans" ),
  TRANSGRP, TRANSPROPERTIES, TRANSLENGTHS, TRANSNONDISCRIM, TRANSSELECT,
  TRANSSIZES, TransGrpLoad, TRANSGrp, TRANSProperties, TransitiveGroup,
  SignPermGroup, AllBlocks, AllCycleStructures, CycleStructures, NumBol,
  SetOrbits, SeqOrbits, OnSetSets, OnSetTuples, OnTupleSets, OnTupleTuples,
  SetSetOrbits, OrbNEq, CntOp, TransitiveIdentification,
  SelectTransitiveGroups, AllTransitiveGroups, OneTransitiveGroup );

#AUTO( ReadTwo( "twogp" ),
#  TGParts, TGLoad, TGGroup, TwoGroup, pClass, AllTwoGroups, OneTwoGroup );
#
#AUTO( ReadThr( "thrgp" ),
#  ThGParts, ThGLoad, ThGGroup, ThreeGroup, AllThreeGroups, OneThreeGroup );
#
#AUTO( ReadTbl( "ctadmin" ),
#  TABLEFILENAME, LIBTABLE, SET_TABLEFILENAME, GALOIS, TENSOR, EvalChars, MBT,
#  MOT, LowercaseString, NotifyCharTableName, NotifyCharTable,
#  LibInfoCharTable, FirstNameCharTable, FileNameCharTable, ALN, ALF, ACM, ARC,
#  ConstructMixed, ConstructProj, ConstructDirectProduct, ConstructIsoclinic,
#  ConstructV4G, ConstructGS3, ConstructPermuted, ConstructSubdirect,
#  UnpackedCll, CllToClf, ConstructClifford, BrauerTree, DecMat,
#  BasicSetBrauerTree, AddDecMats, PartsBrauerTableName, BrauerTable,
#  LibraryTables, CharTableLibrary, OfThose, IsSporadicSimple, SchurCover,
#  AllCharTableNames, ShrinkClifford, TextString, BlanklessPrint, ShrinkChars,
#  ClfToCll, PrintFusion, PrintToLib, PrintClmsToLib );
#
#AUTO( ReadTbl( "ctprimar" ),
#  LIBLIST );
#
#AUTO( ReadTom( "tmprimar" ),
#  TOM, TOMLIST, TomLibrary );
#
#AUTO( ReadSml( "idgroup.sml" ),
#  AgGroupCode, CodeAgGroup, InitRandomIsomorphismChecking, RandomSpecialPres,
#  RandomIsomorphismChecking, IdGroupRandomTest, IdGroupSpecialFp, EvalFpCoc,
#  IdSmallGroup, IdP1Q1R1Group, IdP2Q1Group, IdP1Q2Group, IdP1Group, IdP2Group,
#  IdP3Group, IdP1Q1Group, IdGroup );
#
#AUTO( ReadSml( "smallgrp.sml" ),
#  Codes1000, PermGroupCode, AgGroupCode, GroupCode, LoadSmallGroups,
#  UnloadSmallGroups, SmallGroup, AllSmallGroups, NumberSmallGroups );

# load packages

ReadLib("double");
ReadLib("list");
ReadLib("field");
ReadLib("util");
ReadLib("delay");
ReadLib("namespaces");
ReadLib("newpackage");
ReadLib("save");
ReadLib("base");
ReadLib("hash");
ReadLib("float");
ReadLib("complex");
ReadLib("smartcomplete");
ReadLib("colors");

CantCopy(Rationals);
CantCopy(Doubles);
CantCopy(Complexes);
CantCopy(Cyclotomics);
CantCopy(Integers);

Concat:=ConcatenationString;
QUOTIFY := file_quotify_static;
RandomSeed(TimeInSecs());

# Info Lattice
# ------------
# set to Print for information
if not IsBound(InfoLatticeSpiral) then
  InfoLatticeSpiral := Ignore;
fi;

Read(Concat(Conf("spiral_dir"), Conf("path_sep"), "namespaces", Conf("path_sep"), "init.g"));

_fileProgress.done := true;
