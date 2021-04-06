
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(paradigms.vector, platforms.sse);

#	Tests SIMD functionality
TestSIMD := function()
	if SSE_4x32f.active then
		doSimdDft(5, SSE_4x32f);
		doSimdDft(5, SSE_4x32f, rec(oddSizes:=false, svct:=false));
		doSimdDft([2..20], SSE_4x32f);
	fi;
end;