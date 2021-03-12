
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

#
#  Enter incomplete or wrong information for GAP (i.e., garbage) to see if we can break it...
#

z := Replicate (100);

a := b * 100;
a := b/ 77
     
LiveBags := function()
    local B, E, dead, a;
    [B,E,dead] := BagBounds();
    return Difference(B + 8*[1..(E-B)/8], dead);
end;

BagStats := function(bags)
    local res, b, t
    res := Replicate(Length(TYPES));
    for b in bags do
	    t := 1+BagType(BagFromAddr(b));
	    res[t] := res[t]+1;
    od
    return res;
end;

