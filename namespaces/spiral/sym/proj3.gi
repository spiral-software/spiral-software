
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#s:=SPLRuleTree(ExpandSPL(PolyBDFT3(16))[1]);

#s:=SPLRuleTree(ExpandSPL(BSkewDFT3(16))[1]);
# Even1 -- Odd0
s := K(16,4) * 
     DirectSum(
	 BSkewDFT3(4, 1/16), 
	 BSkewDFT3(4, 3/16), 
	 BSkewDFT3(4, 5/16), 
	 BSkewDFT3(4, 7/16)
     ) * 
     L(16,8) *
     Tensor(I(2), PolyBDFT3(8, 1/4));
     L(16,2);


left := K(8,2) * 
        DirectSum(
	  SkewDTT(DCT3(2), 1/8), 
	  SkewDTT(DCT3(2), 3/8), 
	  SkewDTT(DCT3(2), 5/8), 
	  SkewDTT(DCT3(2), 7/8)
        );

 right:=MatSPL(left)^-1 * MatSPL(DCT3(8));

spiral> ps(MatSPL(L(8,2))*right*MatSPL(L(8,4)));


left := K(8,4) * 
        DirectSum(
	  SkewDTT(DCT3(4), 1/4), 
	  SkewDTT(DCT3(4), 3/4)
        );

 right:=MatSPL(left)^-1 * MatSPL(DCT3(8));

spiral> ps(MatSPL(L(8,4))*right*MatSPL(L(8,2)));

# X * BRDFT * IJ = BRDFT,      X = BRDFT * IJ * BRDFT^-1
# Y * BRDFT * IJ = BRDFT * J,  Y = BRDFT*J*IJ*BRDFT^-1
# 
c3polyalgo := function(N,m,a) 
    local k, left, right, zz, mid;
    k := N/2/m;
    zz := RulesDFTSymmetry(L(N, m) * GathExtend(N,Odd0));
    left := K(N/2,m) * DirectSum(List([0..k-1], i -> SkewDTT(DCT3(m), fr(k,i,a).ev())));
    right := DirectSum(
	SkewDTT(DCT3(k), a), 
	List([1..(m-2)/2], i -> BRDFT3(2*k,a/2)*DirectSum(I(1), -J(2*k-1))), 
	PolyDTT(SkewDTT(DCT4(k),a))
    ) * 
    Compose(Drop(zz.children(), 1));

    mid := MatSPL(left)^-1 * MatSPL(SkewDTT(DCT3(N/2),a)) * MatSPL(right)^-1;
    #PrintMat(mid);
    VisualizeMat(mid, ", ");
    return mid;
end;


#N := 32; m := 4; k := N/2/m;

#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
mid := c3polyalgo(32,8);
p:=Prm(FList([0,2,3,6,7,10,11,14, 1,4,5,8,9,12,13, 15]).setRange(15));

ps(mid * MatSPL(p)^-1);