	failures := 0;
	teststatus := false;
	#epsilon := 1e-9; #.000000001
	epsilon := 1e-3; #.001
	#mostNegativeDouble := 1.7*10^-308; #64 bit
	#mostPositiveDouble := 1.7*10^308;


compareVar := function(expected, actual)
	local adiff;
	#PrintLine("expected: ", expected);
	#PrintLine("actual: ", actual);
    adiff := absdiff(expected, actual);
	
    if adiff <= epsilon then
         return true;
    else
		PrintLine("adiff: ", adiff);
        return false;
    fi;
 end;

  PrintTestResult := function(testname, doubleValue, controlValue)
	  teststatus := compareVar(doubleValue, controlValue);
	  #Print("\nTest: ", testname, "....................\t", "teststatus");
	  if teststatus then
		Print("");
        #Print("Passed\n", doubleValue, "", controlValue);
    else
		PrintLine("Failed ::  Test :: ",testname , "\n Value ", doubleValue, "\n Control Value := " , controlValue);
		failures := failures + 1;
    fi;
     return "";
  end;
 
 
 #PrintLine("");
 temp := 1.7;;
 i := 1;;
 currentAnswer := 0;;
 localstatus := false;; 
 
 repeat
	i := i + 10;
	
	currentAnswer := temp * 10 ^ i;
	
	localstatus := PrintToString(currentAnswer) = "inf";
	
	#if localstatus then
	#	PrintLine("Test Print double hit inf :: ", i);
	#else
	#	PrintLine("Test Print double till inf :: ", i );
	#fi;
 until localstatus;
 
#expand to go to right before infinite? 
double :=  d_PI;; 
 for i in [1..100] do
	double := double + double;
 od;
 
 for i in [1..100] do
	double := double - double;
 od;

	PrintTestResult("Addition and Minus should be zero ", double , 0);


#This is a multiplication then divide 
 temp := 1.5;;
 i := 1;; 
 currentAnswer := 0;;
 product := 1;;
 quotient := 1;;
 localstatus := false;; 
 
 repeat
	i := i + 10;
	
	currentAnswer := temp * 10 ^ i;
	product := currentAnswer * currentAnswer;
	quotient := product / currentAnswer; 
	 	
	localstatus := PrintToString(quotient) = "inf";
	
	if localstatus then
		PrintLine("Test Print double hit inf :: ", i);
	else
		PrintTestResult("Multiplication and Division X * X / X = X :: ", quotient , currentAnswer);
	#	PrintLine("Test Print double till inf :: ", i );
	fi;
 until localstatus;

 
  #sqaure root and inverse
  	PrintLine("Sqaure Root ");
    temp := 1.5;;
	i := 1;; 
	currentAnswer := 0;;
	localstatus := false;; 
 
    repeat
		i := i + 10;
		currentAnswer := temp * 10 ^ i * temp * 10 ^ i;
			
	
		tempAnswer := d_sqrt(currentAnswer)^2;
		#tempAnswer2 := d_sqrt(currentAnswer^2);
		
		localstatus := PrintToString(currentAnswer^2) = "inf";
		if localstatus then
			PrintLine("Test Print double hit inf :: ", i);
		else
			PrintTestResult("d_sqrt(x)^2 :: ",  tempAnswer, currentAnswer);
		#	PrintLine("Test Print double till inf :: ", i );
		fi;
	until localstatus;
 
	PrintLine("Log Exponent ");
    temp := 1.5;;
	i := 1;; 
	currentAnswer := 0;;
	localstatus := false;; 
 
    repeat
	i := i + 10;
	
	#currentAnswer := temp * 10 ^ i * temp * 10 ^ i;
	currentAnswer := temp * 10 ^ i;
	
	localstatus := PrintToString(currentAnswer) = "inf";
	
	logExp := d_log(d_exp(currentAnswer));
	expLog := d_exp(d_log(currentAnswer));
	
	localstatus := PrintToString(logExp) = "inf";
	
	if localstatus then
		PrintLine("Test Print double hit inf :: ", i);
	else
		PrintTestResult("d_log(d_exp(d_PI)), d_exp(d_log(d_PI)",logExp , expLog);
	#	PrintLine("Test Print double till inf :: ", i );
	fi;
	
 until localstatus;
 
 
 
 decimalDouble := 0.0;;
 toDecimal := 1e-2;;
 
for i in [1..360] do
	#PrintLine("\nTest: Sin(X)^2 + Cos(X)^2 = 1 ....................\t", "teststatus");
	#PrintLine("\nTest: ", d_sin(i)^2 + d_cos(i)^2);
	PrintTestResult("Sin(X)^2 + Cos(X)^2 = 1 :: ", d_sin(i)^2 + d_cos(i)^2 , 1);
  od;
  
  for i in [1..100] do
	decimalDouble := i * toDecimal;
	#PrintLine("i :", decimalDouble);
	PrintTestResult("ArcSin(Sin(X)) = x ", d_asin(d_sin(decimalDouble)) , decimalDouble);
  od;
  
  
  decimalDouble := 0.0;
  for i in [1..100] do
  
	decimalDouble := i * toDecimal;
	PrintTestResult("ArcCos(Cos(X)) = x ", d_acos(d_cos(decimalDouble)) , decimalDouble);
  od;
  
    decimalDouble := 0.0;
  for i in [1..100] do
  
	decimalDouble := i * toDecimal;
	PrintTestResult("ArcTan(Tan(X)) = x ", d_atan(d_tan(decimalDouble)) , decimalDouble);
  od;
  

 
 
 
 
 
 #odd and end pieces
 double := d_floor(1.5);;
 PrintTestResult("Floor of 1.5 :: ", double , 1);
 
 double := d_ceil(1.5);;
 PrintTestResult("Ceil of 1.5 :: ", double , 2);
 
 double := d_fabs(-1.5);;
 PrintTestResult("Absolute value of -1.5 :: ", double , 1.5);
 
 
 
#make sure function call and agorithim the same. 
#double := 1.1^1.1;;
#doublePowFunc :=  d_pow(1.1, 1.1);;


#if y non-zero, floating-point remainder of x/y, with same sign as x;
#if y zero, result is implementation-defined
#this doesn't work. 
#double := d_fmod

#https://math.stackexchange.com/questions/820094/what-is-the-best-way-to-calculate-log-without-a-calculator
#follow this 
#base-10 logarithm of x
#double := d_log10(1.1);;
 
 #arc-tangent of x
#double := d_atan(1.1);;

#arc-tangent of y/x
#double :=  d_atan2(1.1, PI);;

# hyperbolic sine of x 
#double := d_sinh(1.1);; 

#hyperbolic cosine of x
#double := d_cosh(1.1);; 

#hyperbolic tangent of x 

if failures <> 0 then
	TestFailExit();
	#Print("I Failed\n");
else
	Print("I Succeeded\n");
fi;
 