
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details


Import(formgen, search);

#F TestSPIRAL ( )
#F   tests the modules in SPIRAL
#F
TestSPIRAL := function ( )
  local passed;

  # test the formula generator
  passed := TestFormGen( 5 );

  # verify code generated
  passed := passed and TestGeneratedCode(10, 5);

  # verify search
  passed := passed and TestSearch();

  if passed then 
    Print("\nTesting SPIRAL: passed\n\n");
  else
    Print("\nTesting SPIRAL: Failed!\n\n");
  fi;
end;
