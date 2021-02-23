
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


nz := x->When(x=0,0,1);
ps := x -> PrintMat(MapMat(x,nz));
crdiag := rr -> List([1..DimensionsMat(rr)[1]/2], 
    i->rr[2*i-1][2*i-1] + E(4)*rr[2*i][2*i-1]);

submat := (m, rbegin, colbegin, size) -> List(m{[rbegin..rbegin+size-1]}, x->x{[colbegin..colbegin+size-1]});

N := 8;
m := 4;
n := N/m;

BIGNT := PRDFT4; 
RIGHTNT := PRDFT4;
LEFTNT := BIGNT;

# DTT -> K^N_m (dirsum SkewDTT(m)) (DST3(n) tensor I(m)) B^N_m
ck := MatSPL(K(N,m));
cm := ck^-1;
rk := RCMatCyc(ck);
rm := RCMatCyc(cm);

big := MatSPL(BIGNT(2*N));
lhs := Tensor(I(n), LEFTNT(2*m));
rhs := Tensor(RIGHTNT(n), I(2*m));
quo := MatSPL(lhs)^-1 * rm * big;  #pquo := quo ^ MatSPL(L(2*N, n)); squo := MapMat(pquo, nz);
rb := quo * MatSPL(rhs)^-1;

# N=2, m=2, n=4
#prb := (MatSPL(L(16,2)*Tensor(M(8,4),I(2)))*rb*MatSPL(L(16,8)));

#TensorProductMat(IdentityMat(2*m), MatSPL(IPRDFT2(n,-1)*Diag(2,2,1,1)));
#PRDFT3(n))^-1);
#IPRDFT2(n,-1)*Diag(2,2,1,1)));

#pquo := quo ^ MatSPL(L(2*N, 2*n));
#blocks := List([1..m], x -> pquo{[1+2*n*(x-1) .. 2*n*x]}{[1+2*n*(x-1) .. 2*n*x]});

#brfinv := MatSPL(PRDFT3(2*n))^-1;
#rdiag := List(blocks, b -> b * brfinv);
#cdiag := List(rdiag, crdiag);
#diag := MatSPL(L(2*N, m)) * quo * MatSPL(L(2*N, 2*n)) * TensorProductMat(IdentityMat(m), brfinv); 
