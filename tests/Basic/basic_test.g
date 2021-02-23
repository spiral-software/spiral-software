
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

## 
##  Generate some simple GAP expressions and assignments.
##

failures := 0;

##  Utility to print test message and success/failure status
PrintTestResult := function(testname, teststatus)
    Print("\nTest: ", testname, teststatus, "....................\t");
    if teststatus then
        Print("Passed\n");
    else
        Print("Failed\n");
		failures := failures + 1;
    fi;
    return;
end;

##  Assign 2 values, verify c = product...
a := 10;; b := 25;; 
c := a*b;;
PrintTestResult("Simple Assignment:  variable = 250 ? ", c = 250);

##  Create an array, verify Length()
z := Replicate(50, 0);;
PrintTestResult("Simple Array: Array 50 elements long? ", Length(z) = 50);

##  Loop over array...
tpass := true;;
for i in [1..50] do
    z[i] := i*a + 2;
    tpass := tpass and z[i] > 0;
od;
PrintTestResult("Looping, arithmetic... ", tpass);

##  Loop over array with calculations involving small & large integers...
tpass := true;;
for i in [1..50] do
    z[i] := 5^i;
od;
for i in [1..50] do
    j:=0; x:=z[i];
    while x > 1 do
        x:=x/5; j:=j+1;
    od;
    tpass := tpass and i = j;
    ##  Print ("i = ", i, ", x = ", x, ", z[i] = ", z[i], "; 5 divides this ", j, " times.\n"); 
    z[i]:=i;
od;
PrintTestResult("Looping, long integers, arithmetic... ", tpass);

##
##  Create a simple object and test get/set methods...
##
obj := rec(
    size := 8,
    name := "zz",

    getSize := meth(self) return self.size; end,
    getName := meth(self) return self.name; end,

    setSize := meth(self,size) self.size:=size; end,
    setName := meth(self,name) self.name:=name; end
);

obj.setSize(2);
obj.setName("test");

if obj.getSize() <> 2 then
    Print("\nsetSize/getSize failed\n");
    res := false;
else
    res := true;
fi;

if obj.getName() <> "test" then
    res := false;
    Error("setName/getName failed");
fi;

PrintTestResult("Create object test set/get methods... ", res);

# This is a very simple benchmark of S.Egner's FFT package interface
#
# It does a lot of memory allocations, and thus tests mostly the performance
# of GASMAN.
#
GASMAN("message");
a:=[1..100000];
Apply(a, ComplexAny);
b:=ComplexFFT(a);;
Length(a) = Length(b);

PrintTestResult("Egner's FFT package interface... ", last);

if failures <> 0 then
	TestFailExit();
fi;

