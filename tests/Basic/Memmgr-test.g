##  Build large lists where elements are complex numbers, create new lists by
##  multiplying them; repeatedly multiply, unbind and re-create.  At the end
##  print the memory stats to see how many GC's performed and average time

##  20 lists, 500,000 entries / list

PrintResetRuntimeStats();

dimn := 500000;;

a01 := []; for i in [1..dimn] do a01[i] := Complex(01 * i, 0); od;
a02 := []; for i in [1..dimn] do a02[i] := Complex(02 * i, 0); od;
a03 := []; for i in [1..dimn] do a03[i] := Complex(03 * i, 0); od;
a04 := []; for i in [1..dimn] do a04[i] := Complex(04 * i, 0); od;
a05 := []; for i in [1..dimn] do a05[i] := Complex(05 * i, 0); od;
a06 := []; for i in [1..dimn] do a06[i] := Complex(06 * i, 0); od;
a07 := []; for i in [1..dimn] do a07[i] := Complex(07 * i, 0); od;
a08 := []; for i in [1..dimn] do a08[i] := Complex(08 * i, 0); od;
a09 := []; for i in [1..dimn] do a09[i] := Complex(09 * i, 0); od;
a10 := []; for i in [1..dimn] do a10[i] := Complex(10 * i, 0); od;
a11 := []; for i in [1..dimn] do a11[i] := Complex(11 * i, 0); od;
a12 := []; for i in [1..dimn] do a12[i] := Complex(12 * i, 0); od;
a13 := []; for i in [1..dimn] do a13[i] := Complex(13 * i, 0); od;
a14 := []; for i in [1..dimn] do a14[i] := Complex(14 * i, 0); od;
a15 := []; for i in [1..dimn] do a15[i] := Complex(15 * i, 0); od;
a16 := []; for i in [1..dimn] do a16[i] := Complex(16 * i, 0); od;
a17 := []; for i in [1..dimn] do a17[i] := Complex(17 * i, 0); od;
a18 := []; for i in [1..dimn] do a18[i] := Complex(18 * i, 0); od;
a19 := []; for i in [1..dimn] do a19[i] := Complex(19 * i, 0); od;
a20 := []; for i in [1..dimn] do a20[i] := Complex(20 * i, 0); od;
PrintLine("Array a01: ", TYPE(a01), " ", Length(a01), " ", SIZE(a01));
PrintLine("Array a02: ", TYPE(a02), " ", Length(a02), " ", SIZE(a02));
PrintLine("Array a03: ", TYPE(a03), " ", Length(a03), " ", SIZE(a03));
PrintLine("Array a04: ", TYPE(a04), " ", Length(a04), " ", SIZE(a04));
PrintLine("Array a05: ", TYPE(a05), " ", Length(a05), " ", SIZE(a05));
PrintLine("Array a06: ", TYPE(a06), " ", Length(a06), " ", SIZE(a06));
PrintLine("Array a07: ", TYPE(a07), " ", Length(a07), " ", SIZE(a07));
PrintLine("Array a08: ", TYPE(a08), " ", Length(a08), " ", SIZE(a08));
PrintLine("Array a09: ", TYPE(a09), " ", Length(a09), " ", SIZE(a09));
PrintLine("Array a10: ", TYPE(a10), " ", Length(a10), " ", SIZE(a10));
PrintLine("Array a11: ", TYPE(a11), " ", Length(a11), " ", SIZE(a11));
PrintLine("Array a12: ", TYPE(a12), " ", Length(a12), " ", SIZE(a12));
PrintLine("Array a13: ", TYPE(a13), " ", Length(a13), " ", SIZE(a13));
PrintLine("Array a14: ", TYPE(a14), " ", Length(a14), " ", SIZE(a14));
PrintLine("Array a15: ", TYPE(a15), " ", Length(a15), " ", SIZE(a15));
PrintLine("Array a16: ", TYPE(a16), " ", Length(a16), " ", SIZE(a16));
PrintLine("Array a17: ", TYPE(a17), " ", Length(a17), " ", SIZE(a17));
PrintLine("Array a18: ", TYPE(a18), " ", Length(a18), " ", SIZE(a18));
PrintLine("Array a19: ", TYPE(a19), " ", Length(a19), " ", SIZE(a19));
PrintLine("Array a20: ", TYPE(a20), " ", Length(a20), " ", SIZE(a20));

comment("compute elements for 20 arrays of length dimn each");

for j in [1..20] do
    b01 := [];  for i in [1..dimn] do b01[i] := a01[i] * a02[i]; od;
    b02 := [];  for i in [1..dimn] do b02[i] := a02[i] * a03[i]; od;
    b03 := [];  for i in [1..dimn] do b03[i] := a03[i] * a04[i]; od;
    b04 := [];  for i in [1..dimn] do b04[i] := a04[i] * a05[i]; od;
    b05 := [];  for i in [1..dimn] do b05[i] := a05[i] * a06[i]; od;
    b06 := [];  for i in [1..dimn] do b06[i] := a06[i] * a07[i]; od;
    b07 := [];  for i in [1..dimn] do b07[i] := a07[i] * a08[i]; od;
    b08 := [];  for i in [1..dimn] do b08[i] := a08[i] * a09[i]; od;
    b09 := [];  for i in [1..dimn] do b09[i] := a09[i] * a10[i]; od;
    b10 := [];  for i in [1..dimn] do b10[i] := a10[i] * a11[i]; od;

    b11 := [];  for i in [1..dimn] do b11[i] := a11[i] * a12[i]; od;
    b12 := [];  for i in [1..dimn] do b12[i] := a12[i] * a13[i]; od;
    b13 := [];  for i in [1..dimn] do b13[i] := a13[i] * a14[i]; od;
    b14 := [];  for i in [1..dimn] do b14[i] := a14[i] * a15[i]; od;
    b15 := [];  for i in [1..dimn] do b15[i] := a15[i] * a16[i]; od;
    b16 := [];  for i in [1..dimn] do b16[i] := a16[i] * a17[i]; od;
    b17 := [];  for i in [1..dimn] do b17[i] := a17[i] * a18[i]; od;
    b18 := [];  for i in [1..dimn] do b18[i] := a18[i] * a19[i]; od;
    b19 := [];  for i in [1..dimn] do b19[i] := a19[i] * a20[i]; od;
    b20 := [];  for i in [1..dimn] do b20[i] := a20[i] * a01[i]; od;
od;

comment("Finished: Print stats...");

PrintResetRuntimeStats();
