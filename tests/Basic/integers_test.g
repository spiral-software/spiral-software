
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

## 
##  Run some tests to exercise small and large integers and arithmetic with them
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

##
##  Calculate m powers of val and repeatedly divide to check results
##  m must be a positive integer!
##

MPowersOfN := function (val, m)
  local pp, x, z, result;
  pp := Replicate(m, 0);
  for i in [1..m] do
    pp[i]:=val^i;
  od;
  x:=0; z:=0; result:=true;
  for i in [1..m] do
    z:=pp[i];
    if z < 0 then z := z * -1; fi;
    x:=0;
    while z > 1 and x <= m do
      z:=z/val;
      if z < 0 then z := z * -1; fi;
      x:=x+1;
    od;
    if x <> i then
      PrintLine("i = ", i, " pp[i] = ", pp[i], " got ", x, " divisions: final dividend = ", z);
      result:=false;
      failures := failures + 1;
    fi;
  od;
  PrintLine("Iterative division of powers of ", val, " valid? -> ", result);
  return;
end;

a:= Replicate(100, 0);;
result := true;;
for i in [1..100] do
  a[i]:= 2^i;
  # PrintLine("i = ", i, ", a^i = ", a[i], " (0x", HexStringInt(a[i]), "), TYPE: ", TYPE(a[i]));
  if i > 1 then
    if a[i] / a[i-1] <> 2 then
      result:=false;
      failures := failures + 1;
    fi;
  fi;
od;
PrintLine("Powers of 2 valid? -> ", result);

result := true;;
for i in [1..100] do
  j := 0; gg := a[i];
  while gg > 1 do
    gg := gg/2; j:=j+1;
    if gg * 2^j <> a[i] then
      result := false;
      failures := failures + 1;
      PrintLine("Multiplication of factors fail: i, j, = ", i, ", ", j,
                ", gg = ", gg, ", gg * 2^j <> a[i]");
    fi;
  od;
od;
PrintLine("Factoring powers of 2 / mulitplication valid? -> ", result);

c:=Replicate(100, 0);;
result := true;;
for i in [1..100] do
  c[i]:=3^i;
  # PrintLine("i = ", i, ", c^i = ", c[i], " (0x", HexStringInt(c[i]), "), TYPE: ", TYPE(c[i]));
  if i > 1 then
    if c[i] / c[i-1] <> 3 then
      result:=false;
      failures := failures + 1;
    fi;
  fi;
od;
PrintLine("Powers of 3 valid> -> ", result);

result := true;;
for i in [1..100] do
  j := 0; gg := c[i];
  while gg > 1 do
    gg := gg/3; j:=j+1;
    if gg * 3^j <> c[i] then
      result := false;
      failures := failures + 1;
      PrintLine("Multiplication of factors fail: i, j, = ", i, ", ", j,
                ", gg = ", gg, ", gg * 3^j <> c[i]");
    fi;
  od;
od;
PrintLine("Factoring powers of 3 / mulitplication valid? -> ", result);

b:=Replicate(100, 0);;
result := true;;
for i in [1..100] do
  b[i]:=5^i;
  # PrintLine("i = ", i, ", b^i = ", b[i], " (0x", HexStringInt(b[i]), "), TYPE: ", TYPE(b[i]));
  if i > 1 then
    if b[i] / b[i-1] <> 5 then
      result:=false;
      failures := failures + 1;
    fi;
  fi;
od;
PrintLine("Powers of 5 valid? -> ", result);

result := true;;
for i in [1..100] do
  j := 0; gg := b[i];
  while gg > 1 do
    gg := gg/5; j:=j+1;
    if gg * 5^j <> b[i] then
      result := false;
      failures := failures + 1;
      PrintLine("Multiplication of factors fail: i, j, = ", i, ", ", j,
                ", gg = ", gg, ", gg * 5^j <> b[i]");
    fi;
  od;
od;
PrintLine("Factoring powers of 5 / mulitplication valid? -> ", result);

p2 := Replicate(100, 0);;
for i in [1..100] do 
  p2[i] := 2^i;
  # Print("i, 2^i = ", i, ", ", p2[i], ", TYPE = ", TYPE(p2[i]), "\n");
od;

zz:=0;; inc:=p2[56];;
for i in [1..128] do
  zz := zz + inc;
  if i = 2 then PrintLine("zz = 2^57 (p2[57]) -> ", zz = p2[57]); fi;
  if i = 4 then PrintLine("zz = 2^58 (p2[58]) -> ", zz = p2[58]); fi;
  if i = 8 then PrintLine("zz = 2^59 (p2[59]) -> ", zz = p2[59]); fi;
  if i = 16 then PrintLine("zz = 2^60 (p2[60]) -> ", zz = p2[60]); fi;
  if i = 32 then PrintLine("zz = 2^61 (p2[61]) -> ", zz = p2[61]); fi;
  if i = 64 then PrintLine("zz = 2^62 (p2[62]) -> ", zz = p2[62]); fi;
  if i = 128 then PrintLine("zz = 2^63 (p2[63]) -> ", zz = p2[63]); fi;
od;

## Test powers of positive integers
for i in [2..100] do
  MPowersOfN(i, 100);
od;

## Test powers of negative integers
for i in [-100..-2] do
  MPowersOfN(i, 100);
od;

if failures <> 0 then
	TestFailExit();
fi;
