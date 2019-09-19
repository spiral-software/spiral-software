
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details

TestFailExit := function()
	Print("TEST FAILED\n");
	Exit(-1);
end;


TestSkipExit := function()
	Print("Skipping test\n");
	Exit(86);
end;
