#!/usr/bin/perl

open IN, "<tests.txt";
open OUT, ">tests.gi";   #!

print OUT "HDLTest := function(which, type, bitwidth, twidwidth)\n   local opts;\n   opts := InitStreamHw();\n   spiral.paradigms.stream._setHDLDataType(type, bitwidth);\n
";

print OUT "   if ((type = \"fix\") and (twidwidth <> -1)) then
      spiral.paradigms.stream._setHDLTwidWidth(twidwidth);
   else
      spiral.backend.default_profiles.fpga_splhdl.makeopts.TWIDTYPE := \"\";
   fi;\n";


$ct = 0;

while (<IN>) {
  if ($_ =~ /#/) {
      ;
  }
  else {
      print OUT "if ((which=$ct) or (which=-1)) then\n";
      print OUT "   Print(\"--- test $ct: bitwidth \", bitwidth, \"$_";
      print OUT "fi;\n";
      $ct++;
  }

}
print OUT "end;\n";
