
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

## 
##  Generate some simple GAP expressions and assignments.
##
failures := 0;
teststatus := false;
listTest := false;

##  Utility to print test message and success/failure status
PrintTestResult := function(testname, printValue, controlValue)
	listTest :=	TYPE(printValue) = "list";

	if listTest then
		
		printValue := printValue[1];
	fi;
	
	listTest :=	TYPE(controlValue) = "list";
	
	if listTest then
		controlValue := controlValue[1];
	fi;
	

	teststatus	:=	printValue = controlValue;
    Print("\nTest: ", testname, "....................\t");
    if teststatus then
        Print("Passed\n");
    else
        Print("Failed\n");
		PrintLine("Print Value ", printValue, " Control Value := " , controlValue);
		failures := failures + 1;
    fi;
    return;
end;

#PrString
stringTest := "Test";;
PrintTestResult("Print string ", PrintToString(stringTest) , "Test");

#PrInteger
smallInt := 32767;; #highest small int
PrintLine(TYPE(smallInt));
PrintTestResult("Print small integer ", PrintToString(smallInt) , "32767");
#Print("smallInt == ", smallInt, "\n");

oneAbove := smallInt + 1;; #largest med int go into large 
#Print("smallInt oneAbove == ", oneAbove, "\n");
PrintTestResult("Print  one above smallInt ", PrintToString(oneAbove) , "32768");

medInt := 2147483647;; #largest med int
#Print("medInt == ", medInt, "\n");
PrintTestResult("Print med integer ", PrintToString(medInt) , "2147483647");

oneAbove := medInt + 1;; #largest med int go into large 
#Print("oneAbove == ", oneAbove, "\n");
PrintTestResult("Print one above medInt ", PrintToString(oneAbove) , "2147483648");


largeInt := 2345678765423131435463512413453645312457685324135675813245768545323456787654231314354635124134536453124576853241356758132457685453;;
#Print("largeInt == ", largeInt, "\n");
PrintTestResult("Print large integer ", PrintToString(largeInt) , "2345678765423131435463512413453645312457685324135675813245768545323456787654231314354635124134536453124576853241356758132457685453");

oneAbove := largeInt + 1;; #largest med int go into large 
#Print("oneAbove == ", oneAbove, "\n");
PrintTestResult("Print one above largeInt ", PrintToString(oneAbove) , "2345678765423131435463512413453645312457685324135675813245768545323456787654231314354635124134536453124576853241356758132457685454");


#PrDbl
double := 123124124124103123.003123 ;; #double
PrintLine(TYPE(double));
PrintTestResult("Print double ", PrintToString(double) , "1.2312412412410312e+17.0");

#PrRat
rational := [2/3] + [4/2];; #rational
PrintLine(TYPE(rational));
PrintTestResult("Print rational ", PrintToString(rational) , "\[ 8/3 \]");

#PrSet
setTemp := Set([2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]);; #set
PrintLine(TYPE(setTemp));
PrintTestResult("Print set ", PrintToString(setTemp) , "Set(\[ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32 \])");

list := ["fe", "fi", "foo", "thumb"];; #list
PrintLine(TYPE(list));
PrintTestResult("Print list ", PrintToString(list) , "[ \"fe\", \"fi\", \"foo\", \"thumb\" ]");


vector := [ 1, 2, 3 ] + [ 1/2, 1/3, 1/4 ];; #vector  [3, 6, 2, 5/2]; 
PrintLine(TYPE(vector));
PrintTestResult("Print vector ", PrintToString(vector) , "[ 3/2, 7/3, 13/4 ]");

boolList := BlistList( [1..10], [2,3,5,7] );; #Boolean List
PrintLine(TYPE(boolList));
PrintTestResult("Boolean List", PrintToString(boolList) , "[ false, true, true, false, true, false, true, false, false, false ]");

Matrix := IdentityMat( 3, GF(2) );; #Matrix
PrintLine(TYPE(Matrix));
PrintTestResult("Print Matrix ", PrintToString(Matrix) , "[ [ Z(2)^0, 0*Z(2), 0*Z(2) ], [ 0*Z(2), Z(2)^0, 0*Z(2) ], [ 0*Z(2), 0*Z(2), Z(2)^0 ] ]");

#PrBool
PrintLine(TYPE(true));  
PrintTestResult("Print bool ", PrintToString(true) , "true");

#PrFunction
#internal function
#PrintLine(TYPE(Print));  
#PrintTestResult("Print internal function ", PrintToString(Print) , "(arg) -> Print(ApplyFunc(Print, arg), \"\\n\")");
#GS4 -Is falling back to printing the name of the function? 

#PrFunction
#function
PrintLine(TYPE(Load));
#listTest :=	TYPE(PrintToString(Load)) = "list";
#GS4 - this should be removed when we are done moving from develop to streamio.
#Comes in as a list of strings for each line. 
#if listTest then
#	PrintTestResult("Print function ", StringPrint(Load) , " function ( pkg ) " +
#"    local  path, usage, res;  " +
#"    usage := \"Load( package.subpackage1.subpackage2... )  " +
#"	\";\n    pkg := _pkg_resolve2(pkg, usage); " + 
#"    path := PATH_SEP :: PathNSSpec(NSId(pkg)); " +
#"    res := _Load(path, pkg); " +
#"    WarnUndefined(res, Eval(pkg)); " +
#"    return res; "+
#"	end");
	#PrintTestResult("Print function ", StringPrint(Load) , "function ( pkg )\nlocal  path, usage, res;\nusage := \"Load( package.subpackage1.subpackage2... )\\n\";\npkg := _pkg_resolve2(pkg, usage);\npath := PATH_SEP :: PathNSSpec(NSId(pkg));\nres := _Load(path, pkg);\nWarnUndefined(res, Eval(pkg));\nreturn res;\nend");
#else
listTest :=	TYPE(PrintToString(Load)) <> "list";
if listTest then
	PrintTestResult("Print function ", PrintToString(Load) , "function ( pkg )\nlocal  path, usage, res;\nusage := \"Load( package.subpackage1.subpackage2... )\\n\";\npkg := _pkg_resolve2(pkg, usage);\npath := PATH_SEP :: PathNSSpec(NSId(pkg));\nres := _Load(path, pkg);\nWarnUndefined(res, Eval(pkg));\nreturn res;\nend");
fi;
	

#"cyclotomic"
#PrCyc
cycVar := ER( 5 );;
PrintLine(TYPE(ER( 5 ))); 
PrintTestResult("Print cyclotomic ", PrintToString(ER( 5 )) , "E(5)-E(5)^2-E(5)^3+E(5)^4");


#range
#PrRange
rangeVar	 := [10..20];;
PrintLine(TYPE(rangeVar)); 
PrintTestResult("Print range ", PrintToString(rangeVar) , "[ 10 .. 20 ]");

#"permutation16"
permVar	 := (1,2,3) * (2,3,4);;
TYPE((1,2,3) * (2,3,4));
PrintTestResult("Print range ", PrintToString(permVar) , "(1,3)(2,4)");

#Print of a record

#rec(
#  year := 1992,
#  month := "Jan",
#  day := 13 )
# ResidueOps := rec( );;
# ResidueOps.Print := function ( r )
#   Print( "Residue( ",
#              r.representative mod r.modulus, ", ",
#               r.modulus, " )" );
# end;;
# Residue := function ( representative, modulus )
#  return rec(
#    representative := representative,
#     modulus        := modulus,
#     operations     := ResidueOps );
# end;;
# l := Residue( 33, 23 );

#PrintRec(date);
# m := [[1,2,3,4],[5,6,7,8],[9,10,11,12]];
#[ [ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ], [ 9, 10, 11, 12 ] ]
#PrintArray( m );
#    [ [   1,   2,   3,   4 ],
#      [   5,   6,   7,   8 ],
#      [   9,  10,  11,  12 ] ]


if failures <> 0 then
	TestFailExit();
	#Print("I Failed\n");
	else
	Print("I Succeeded\n");
fi;
