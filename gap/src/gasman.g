# -*- Mode: shell-script -*-

LiveBags := function()
    local B, E, dead, a;
    [B,E,dead] := BagBounds();
    return Difference(B + 8*[1..(E-B)/8], dead);
end;

BagStats := function(bags)
    local res, b, t;
    res := Replicate(Length(TYPES), 0);
    for b in bags do
	    t := 1+BagType(BagFromAddr(b));
	    res[t] := res[t]+1;
    od;
    return res;
end;

BagsByType := function(bags)
    local res, b, t;
    res := List([1..Length(TYPES)], x->[]);
    for b in bags do
	    t := 1+BagType(BagFromAddr(b));
	    Add(res[t], b);
    od;
    return res;
end;

PrintBagStats := function(bag_stats, range)
    local i, nprinted;
    nprinted := 1;
    for i in [1..Length(bag_stats)] do
        if bag_stats[i] in range then
	    Print(TYPES[i], " ", bag_stats[i], "\t");
	    if nprinted mod 4 = 0 then Print("\n"); fi;
	    nprinted := nprinted + 1;
	fi;
    od;
    Print("\n");
end;

clear_state := function()
    var.flush();
    transforms.HASH := HashTableDP();
    CodeSPL(100, F(2));
end;

GASMAN("message");
vdp4(10);

clear_state();
GASMAN("collect");GASMAN("collect");

a := LiveBags();; astats := BagStats(a);;
CodeRuleTree(100, TVec(TRC(DFT(16)), AVecReg(4)));;
clear_state();
GASMAN("collect");GASMAN("collect");
b := LiveBags();; bstats := BagStats(b);;

PrintBagStats(bstats-astats, [1..2^27]);
diff := Difference(b, a);; diff_stats := BagStats(diff);; tdiff := BagsByType(diff);;
PrintBagStats(diff_stats, [1..2^27]);

 bb := t ->      tdiff[1+t];
ebb := t -> List(tdiff[1+t], BagFromAddr);

# [ 137499044, 137499056, 137499076, 138064392, 138064396, 138064476,
#   138065088, 138199844, 138832556, 139121904, 139122092, 139122152,
#   140056760, 140056768, 140056776, 140060576, 140060584, 140060592,
#   140063488, 140063496, 140063504, 140114636, 140114644, 140114652,
#   140147824, 140148640, 140148648, 140148664, 140148676, 140148692,
#   140148704, 140148720, 140148724, 140148736, 140148848, 140148868, 140461152
#  ]
