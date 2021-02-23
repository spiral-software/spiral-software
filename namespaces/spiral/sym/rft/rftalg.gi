
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(spiral);

SkewRDFT := BSkewPRDFT;
BRDFT := PolyBDFT;

alphai := (n,i,a) -> Cond(i mod 2 = 0, (a + Int(i/2))/n, (1-a + Int(i/2))/n);

time_r := a -> [[1, -CosPi(2*a)/SinPi(2*a)], 
                [0, 1/SinPi(2*a)]];

r_time := a -> [[1, CosPi(2*a)], 
                [0, SinPi(2*a)]];

B1_time_r := n -> Cond(IsOddInt(n), 
    DirectSum(I(1), List([1..(n-1)/2], i-> Mat(time_r(i/n)))),
    DirectSum(I(1), List([1..n/2-1], i-> Mat(time_r(i/n))), I(1)));

RDFT1 := SymFunc("RDFT1", n -> Cond(
	n=2,F(2), 
	IsEvenInt(n), VStack(Gath(H(n+2,1,0,1)), Gath(H(n+2,n-2,2,1)), Gath(H(n+2,1,n,1))) * PRDFT1(n),
	IsOddInt(n), VStack(Gath(H(n+1,1,0,1)), Gath(H(n+1,n-1,2,1))) * PRDFT1(n)));

RDFT3 := SymFunc("RDFT3", n -> Cond(
	n=2,I(2), 
	IsEvenInt(n), PRDFT3(n),
	IsOddInt(n), VStack(Gath(H(n+1,n-1,0,1)), Gath(H(n+1,1,n-1,1))) * PRDFT3(n)));

BRDFT1 := SymFunc("BRDFT1", n -> B1_time_r(n) * RDFT1(n));
 
rdft1 := (k,m) -> 
   Perm(PermFunc(rP(2*k*m, 2*k), 2*k*m), 2*k*m).transpose() * 
   DirectSum(RDFT1(m), List([1..k-1], i->Mat(MatrDFT(2*m,i/(2*k)))), RDFT3(m)) * 
   Tensor(RDFT1(2*k), I(m)); 

rdft1odd := (k,m) -> 
   Perm(PermFunc(rPhat((2*k-1)*m, 2*k-1), (2*k-1)*m), (2*k-1)*m).transpose() * 
   DirectSum(RDFT1(m), List([1..k-1], i->Mat(MatrDFT(2*m,i/(2*k-1))))) * 
   Tensor(RDFT1(2*k-1), I(m)); 

test_rdft1 := (k,m) -> PrintLine(inf_norm(MatSPL(rdft1(k,m)) - MatSPL(RDFT1(2*k*m))));
test_rdft1odd := (k,m) -> PrintLine(inf_norm(MatSPL(rdft1odd(k,m)) - MatSPL(RDFT1((2*k-1)*m))));
do_test_rdft1 := function()
   test_rdft1(4,4);
   test_rdft1(4,8);
   test_rdft1(6,8);
   test_rdft1(6,2);
   test_rdft1(6,5);
   test_rdft1(11,5);
end;
do_test_rdft1odd := function()
   test_rdft1odd(4,4);
   test_rdft1odd(4,8);
   test_rdft1odd(6,8);
   test_rdft1odd(6,2);
   test_rdft1odd(6,5);
   test_rdft1odd(5,5);
   test_rdft1odd(11,5);
end;

rdft3 := (k,m) -> 
   RC(K(k*m, m)) *
   DirectSum(List([0..k-1], i->Mat(MatrDFT(2*m,(i+1/2)/(2*k))))) * #alphai(k,i,1/4))))) *
   Tensor(RDFT3(2*k),I(m));

rdft3odd := (k,m) -> 
   Perm(PermFunc(rQhat((2*k-1)*m, 2*k-1), (2*k-1)*m), (2*k-1)*m).transpose() * 
   DirectSum(List([0..k-2], i->Mat(MatrDFT(2*m,(i+1/2)/(2*k-1)))), RDFT3(m)) * #alphai(k,i,1/4))))) *
   Tensor(RDFT3(2*k-1),I(m));

test_rdft3 := (k,m) -> PrintLine(inf_norm(MatSPL(rdft3(k,m)) - MatSPL(RDFT3(2*k*m))));
test_rdft3odd := (k,m) -> PrintLine(inf_norm(MatSPL(rdft3odd(k,m)) - MatSPL(RDFT3((2*k-1)*m))));
do_test_rdft3 := function()
   test_rdft3(4,4);
   test_rdft3(4,8);
   test_rdft3(6,8);
   test_rdft3(6,2);
   test_rdft3(6,5);
   test_rdft3(11,5);
end;
do_test_rdft3odd := function()
   test_rdft3odd(4,4);
   test_rdft3odd(4,8);
   test_rdft3odd(6,8);
   test_rdft3odd(6,2);
   test_rdft3odd(6,5);
   test_rdft3odd(5,5);
   test_rdft3odd(11,5);
end;

rrdft := (k,m,a) -> 
   RC(K(k*m, m)) *
   DirectSum(List([0..k-1], i->Mat(MatrDFT(2*m,(i+a)/(2*k))))) *
   Tensor(Mat(MatrDFT(2*k,a)),I(m));

brdft := (k,m,a) -> 
   RC(K(k*m, m)) *
   DirectSum(List([0..k-1], i->BRDFT(2*m,alphai(k,i,a)))) * 
   Tensor(BRDFT(2*k, a), I(m));

bruun_rdft3 := (k,m) -> 
   RC(K(k*m, m)) *
   DirectSum(List([0..k-1], i->SkewRDFT(2*m,alphai(k,i,a)))) * 
   Tensor(BRDFT(2*k,1/4),I(m));

bruun_rdft1 := (k,m) -> 
   DirectSum(RDFT1(m), List([1..k-1], i->SkewRDFT(2*m,i/(2*k))), RDFT3(m)) * 
   Tensor(BRDFT1(2*k), I(m)); 

rcred := mat -> List(mat{[0..Length(mat)/2-1]*2+1}, row -> row{[0..Length(row)/2-1]*2+1});
