
##  Copyright (c) 2018-2023, Carnegie Mellon University
##  See LICENSE for details

#
failures := 0;
teststatus := false;
listTest := false;
PI :=  d_PI; #3.1415926535897931



##  Utility to print test message and success/failure status
PrintTestResult := function(testname, printValue, controlValue)

#this code is so that it can run on develop branch and on the new streamio branch.
#develop branch PrintToString prints a list with a string in it.
#Streamio has changed to just use a sting. 
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
		PrintLine("Test Value ", printValue, " Control Value := " , controlValue);
		failures := failures + 1;
    fi;
    return;
end;

#PI is contanst. Make sure always same.
double := 3.1415926535897931;;
#PrintLine(TYPE(double));
PrintTestResult("Print PI ", double , PI );
PrintTestResult("PrintToString PI ", PrintToString(double) , PrintToString(PI));

#Double
double := 123124124124103123.003123 ;; #double
#PrintLine(TYPE(double));
PrintTestResult("Print double ", PrintToString(double) , "1.2312412412410312e+17.0");

#PrintToString
double := 0.003123 ;; 
PrintTestResult("Print double PrintToString ", PrintToString(double) , "0.0031229999999999999");
#PrintTestResult("Print double StringDouble ", StringDouble("%f", double) , "0.003123");


#DblSum
double := 1.1 + 1.1 ;; 
PrintTestResult("Print sum ", PrintToString(double) , "2.2000000000000002");


#DblDiff
double := 5.25 - 5.00;;
PrintTestResult("Print diff ", PrintToString(double) , "0.25");


#DblProd
double := 5.25*5.25;;
PrintTestResult("Print prod ", PrintToString(double) , "27.5625");


#DblQuo
double := 50.50/5.05;;  
PrintTestResult("Print quotent ", PrintToString(double) , "10.0");

#test with mixing large positive and negative numbers

double := 2.5^19 + 2.5^19;;
PrintTestResult("Print equation 2.5^19 + 2.5^19", PrintToString(double) , "72759576.141834259");

double := 2.5^19 + (-2.5^19);;
PrintTestResult("Print equation 2.5^19 + (-2.5^19)", PrintToString(double) , "0.0");

double := -2.5^19 + (-2.5^19);;
PrintTestResult("Print -2.5^19 + (-2.5^19)", PrintToString(double) , "-72759576.141834259");

double := 2.5^19 - 2.5^19;;
PrintTestResult("Print equation 2.5^19 + 2.5^19", PrintToString(double) , "0.0");

double := 2.5^19 - (-2.5^19);;
PrintTestResult("Print equation 2.5^19 + (-2.5^19)", PrintToString(double) , "72759576.141834259");

double := -2.5^19 - (-2.5^19);;
PrintTestResult("Print equation -2.5^19 + (-2.5^19)", PrintToString(double) , "0.0");

double := -2.5^19 - 2.5^19;;
PrintTestResult("Print equation -2.5^19 + (-2.5^19)", PrintToString(double) , "-72759576.141834259");

double := 2.5^19 + 2.5^19;;
PrintTestResult("Print equation 2.5^19 + 2.5^19", PrintToString(double) , "72759576.141834259");

double := 2.5^19 + (-2.5^19);;
PrintTestResult("Print equation 2.5^19 + (-2.5^19)", PrintToString(double) , "0.0");
 

double := 2.5^19 * 2.5^19;;
PrintTestResult("Print equation 2.5^19 * 2.5^19", PrintToString(double) , "1323488980084844.2");

double := 2.5^19 * -2.5^19;;
PrintTestResult("Print equation 2.5^19 * -2.5^19", PrintToString(double) , "-1323488980084844.2");

double := -2.5^19 * -2.5^19;;
PrintTestResult("Print equation 2.5^19+(-2.5^19)", PrintToString(double) , "1323488980084844.2");

double := 2.5^19 / 2.5^19;;
PrintTestResult("Print equation 2.5^19 / 2.5^19", PrintToString(double) , "1.0");

double := 2.5^19 / -2.5^19;;
PrintTestResult("Print equation 2.5^19 / -2.5^19", PrintToString(double) , "-1.0");

double := -2.5^19 / -2.5^19;;
PrintTestResult("Print equation (-2.5^19 / -2.5^19)", PrintToString(double) , "1.0");

#get a different number then one
double := 2.5^25 / 4;; #8881784197 / 4
PrintTestResult("Print equation (2.5^25 / 4)", PrintToString(double) , "2220446049.2503133");

#highest numbers before infinity.
double := 2.34^834;;
PrintTestResult("Print equation 2.34^834", PrintToString(double) , "8.4338346088435799e+307.0");

double := 2.35^834;;
PrintTestResult("Print equation 2.35^834", PrintToString(double) , "inf");

double := -2.34^834;;
PrintTestResult("Print equation -2.34^834", PrintToString(double) , "-8.4338346088435799e+307.0");

double := -2.35^834;;
PrintTestResult("Print equation -2.35^834", PrintToString(double) , "-inf");

#DblAnySum
#DblAnyDiff
#DblAnyProd
#DblAnyQuo
#DblAnyPow

#raised to power y
double := 1.1^1.1;;
PrintTestResult("Print double 1.1 raised to power 1.1", PrintToString(double) , "1.1105342410545758");

#raised to power y function call
double :=  d_pow(1.1, 1.1);;
PrintTestResult("Print double 1.1 raised to power 1.1 ", PrintToString(double) , "1.1105342410545758");

#make sure function call and agorithim the same. 
double := 1.1^1.1;;
doublePowFunc :=  d_pow(1.1, 1.1);;
PrintTestResult("compare previous two power function vs algebric ", PrintToString(double) ,  PrintToString(doublePowFunc));

#natural logarithm of x
double := d_log(1.1);;
PrintTestResult("Print natural logarithm of x ", PrintToString(double) , "0.095310179804324935");

#base-10 logarithm of x
double := d_log10(1.1);;
PrintTestResult("Print base-10 logarithm of 1.1 ", PrintToString(double) , "0.041392685158225077");

#exponential of xdouble
double := d_exp(1);;
PrintTestResult("Print double ", PrintToString(double) , "2.7182818284590451");

# square root of x
double := d_sqrt(25);;
PrintTestResult("Print square root of 25 ", PrintToString(double) , "5.0");

#smallest integer not less than x
double := d_floor(1.1);;
PrintTestResult("Print smallest integer not less than 1.1 ", PrintToString(double) , "1.0");

#largest integer not greater than x 
double := d_ceil(1.1);;
PrintTestResult("Print largest integer not greater than 1.1  ", PrintToString(double) , "2.0");

#absolute value of x
double := d_fabs(-1.1);;
PrintTestResult("Print absolute value of -1.1 ", PrintToString(double) , "1.1000000000000001");

#sine of x
double := d_sin(1.1);;
PrintTestResult("Print sine of 1.1 ", PrintToString(double) , "0.89120736006143542");

#cosine of x
double := d_cos(1.1);;
PrintTestResult("Print cosine of 1.1 ", PrintToString(double) , "0.45359612142557731");

# tangent of x 
double := d_tan(1.1);;
PrintTestResult("Print tangent of 1.1 ", PrintToString(double) , "1.9647596572486523");

#if y non-zero, floating-point remainder of x/y, with same sign as x;
#if y zero, result is implementation-defined
#this doesn't work. 
double := d_fmod(100.132356, 871.10);; 
PrintTestResult("Print floating-point remainder of x/y  ", PrintToString(double) , "100.132356");

#arc-sine of x 
double := d_asin(0.2131230);;
PrintTestResult("Print arc-sine of x  ", PrintToString(double) , "0.21477028854003327");

# arc-cosine of x
double := d_acos(0.5631);;
PrintTestResult("Print arc-cosine of x ", PrintToString(double) , "0.97266403906690169");

#arc-tangent of x
double := d_atan(1.1);;
PrintTestResult("Print -> arc-tangent of x -> d_atan ", PrintToString(double) , "0.83298126667443173");

#arc-tangent of y/x
double :=  d_atan2(1.1, PI);;
PrintTestResult("Print arc-tangent of y/x ", PrintToString(double) , "0.33680031481098566");

# hyperbolic sine of x 
double := d_sinh(1.1);; 
PrintTestResult("Print hyperbolic sine of x ", PrintToString(double) , "1.3356474701241769");

#hyperbolic cosine of x
double := d_cosh(1.1);; 
PrintTestResult("Print hyperbolic cosine of x ", PrintToString(double) , "1.6685185538222564");

#hyperbolic tangent of x 
double := d_tanh(1.1);; 
#PrintLine(TYPE(double));
PrintTestResult("Print hyperbolic tangent of x  ", PrintToString(double) , "0.8004990217606297");


#possible additions. 
#for i in [1..800] do
#	PrintLine("3.14 to the power of i = ", i, " ==>", PI^i);
#od;

#StringDouble("2%f", 11.30); "211.300000"

#DoubleString("2.23333"); 2.23333

#DoubleString(PrintToString(2.343)); 2.343
#DoubleRep64(2.3);
#4612361558371493478

#RatDouble(3.3);
#3715469692580659/1125899906842624

#IsDouble
#IntDouble
#IntDouble(2.0);
#2


if failures <> 0 then
	TestFailExit();
	#Print("I Failed\n");
else
	Print("I Succeeded\n");
fi;
 