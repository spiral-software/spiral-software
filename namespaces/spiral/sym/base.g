
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

check := (me, them) -> Cond(
    InfinityNormMat(MatSPL(me) - MatSPL(them)) <> 0, Error("Fail"),
    PrintLine(them, "   OK"));

HRot := r -> let(c := CosPi(-r), s := SinPi(-r),
    Mat([[c+s, c-s],
         [c-s, -c-s]]));

rdft3_4 := Diag(1,1,1,-1) * Tensor(F(2), I(2)) *
           DirectSum(I(2), Rot(-1/4)) * L(4,2);

rDFT_4 := r -> Diag(1,1,1,-1) * Tensor(F(2), I(2)) *
               DirectSum(I(2), Rot(-r)) * L(4,2);

rdft4_4 := IJ(4,2) * Tensor(F(2), I(2)) *
           DirectSum(Rot(-1/8), Rot(-3/8)) * L(4,2);

rDFT2_4 := r -> IJ(4,2) * Tensor(F(2), I(2)) *
               DirectSum(Rot(-r/2), Rot(-3*r/2)) * L(4,2);

dht3_4 := IJ(4,2) * Tensor(F(2), I(2)) *
          DirectSum(F(2), HRot(-1/4)) * L(4,2);

dht4_4 := Diag(1,1,1,-1) * 
          Tensor(F(2), I(2)) *
          DirectSum(HRot(-1/8), HRot(-3/8)) * L(4,2);

rDHT_4 := r -> IJ(4,2) * Tensor(F(2), I(2)) *
               DirectSum(I(2), Rot(r)) * L(4,2);

rDHT2_4 := r -> Diag(1,1,1,-1) * 
                Tensor(F(2), I(2)) *
                DirectSum(Rot(r/2), Rot(3*r/2)) * L(4,2);
                
rDHT_4_new := r -> DirectSum(Diag(1,-1), J(2)) * rDFT_4(r) * Diag(1,1,-1,-1);

checkall := function() 
    local p;
    p := RC(K(4,2));

    check(rdft3_4, PRDFT3(4));
    check(rdft4_4, PRDFT4(4));
    check(dht3_4, PDHT3(4));
    check(dht4_4, PDHT4(4));
    
    check(p*DirectSum(rDHT_4 (1/8), rDHT_4 (3/8))*Tensor(PDHT3(4), I(2)), PDHT3(8));
    check(p*DirectSum(rDHT2_4(1/8), rDHT2_4(3/8))*Tensor(PDHT3(4), I(2)), PDHT4(8));

    check(p*DirectSum(rDFT_4 (1/8), rDFT_4 (3/8))*Tensor(PRDFT3(4), I(2)), PRDFT3(8));
    check(p*DirectSum(rDFT2_4(1/8), rDFT2_4(3/8))*Tensor(PRDFT3(4), I(2)), PRDFT4(8));
end;

#MatSPL(
#    RC(K(4,2)) * DirectSum(rdht4_4(1/8), rdht4_4(3/8)) * 
#    Tensor(PDHT3(4), I(2))
#) - MatSPL(PDHT4(8));
