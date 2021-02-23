
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


RulesFor(DFT, rec(
    DFT_SR_Reg := rec(
	isApplicable := P -> P[1] mod 2 = 0 and P[1] > 4,

	allChildren := P -> let(k:=P[2], 
	    Map2(Filtered(DivisorPairs(P[1]), dd->dd[2]>2 and (dd[2] mod 2 = 0)),
	    (m,n) -> [DFT(m,k), DFT(2*m,k), UDFT(n,k)])), 

	rule := (P,C) -> let( # we know that n >= 4
	    N := P[1], m:=Cols(C[1]), n:=Cols(C[3]), k:=P[2], hf:=(n-2)/2,
	    i := Ind(hf), j := Ind(hf), 

	    SUM(        _hhi(N, 2*m, 0,      n/2, C[2] * L(2*m,2)),
		ISum(i, _hhi(N,   m, i+1,    n,   C[1] * Diag(Twid(N,m,k,0,0,i+1)))),
		ISum(j, _hhi(N,   m, j+2+hf, n,   C[1] * Diag(Twid(N,m,k,0,0,j+2+hf))))
	    ), 
	    Tensor(I(m), C[3]) * 
	    L(N,m)
	)
    ),

    DFT_SR_Reg2 := rec(
	isApplicable := P -> P[1] mod 4 = 0 and P[1] > 4,

	allChildren := P -> let(k:=P[2], 
	    Map2(Filtered(DivisorPairs(P[1]), dd->dd[1]>2 and dd[2]>2 and (dd[1] mod 2 = 0) and (dd[2] mod 2 = 0)),
	    (m,n) -> [UDFT(m,k).transpose(), UUDFT(2*m,k).transpose(), 
		      UDFT(n,k),             UUDFT(2*n, k)])), 

	rule := (P,C) -> let( # we know that n >= 4
	    N := P[1], m:=Cols(C[1]), n:=Cols(C[3]), k:=P[2], nhf:=(n-2)/2, mhf:=(m-2)/2,
	    i := Ind(nhf), j := Ind(nhf), ii := Ind(mhf), jj := Ind(mhf), 
	    q := Ind(2*n),
	    dd := Diag(Lambda(q, cond(eq(q,3*n/2), (-E(4))^k, 1))), 

	    SUM(        _hhi(N, 2*m, 0,       n/2, C[2] * L(2*m,2)),
		ISum(i, _hhi(N,   m, i+1,     n,   C[1] * Diag(UTwid(N,m,k,0,0,i+1)))),
		ISum(j, _hhi(N,   m, j+2+nhf, n,   C[1] * Diag(UTwid(N,m,k,0,0,j+2+nhf))))
	    ) * 
	    SUM(Scat(fStack(H(N,n,0,1), H(N,n,N/2,1))) * dd * C[4] * Gath(H(N, 2*n, 0, m/2)), 
		ISum(ii, _hho(N,   n, n+ii*n,         1,  ii+1,     m,   C[3])),
		ISum(jj, _hho(N,   n, 2*n+mhf*n+jj*n, 1,  jj+mhf+2, m,   C[3]))
	    )
	)
    ),

    DFT_SR123_Corr := rec(
	isApplicable := P -> P[1] mod 4 = 0,
	allChildren := P -> let(k:=P[2], 
	    Map2(Filtered(DivisorPairs(P[1]), dd -> (dd[1] mod 2=0) and (dd[2] mod 2 = 0)),
	    (m,n) -> [DFT(m,k), DFT3(m,k), DFT(n,k), DFT2(n, k)])), 

	rule := function(P,C)
	    local N,m,n,k,nhf,mhf, i,j,ii,jj,q,dd,isum, tw1a, tw1b, tw2a, tw2b;

	    N := P[1]; m:=Cols(C[1]); n:=Cols(C[3]); k:=P[2]; nhf:=(n-2)/2; mhf:=(m-2)/2;
	    i := Ind(nhf); j := Ind(nhf); ii := Ind(mhf); jj := Ind(mhf); 
	    q := Ind(n);
	    dd := Diag(Lambda(q, cond(eq(q,n/2), (-E(4))^k, 1)));
	    isum := (var, expr) -> When(var.range=0, [], ISum(var, expr));
	
	    return
	    SUM(         _hhi(N,   m, 0,       n,   C[1]),
		isum(i,  _hhi(N,   m, i+1,     n,   C[1] * Diag(fPrecompute(UTwid(N,m,k,0,0,i+1))))),
		         _hhi(N,   m, nhf+1,   n,   C[2]),
		isum(j,  _hhi(N,   m, j+nhf+2, n,   C[1] * Diag(fPrecompute(UTwid(N,m,k,0,0,j+2+nhf)))))
	    ) * 
	    SUM(         _hho(N,   n, 0,              1,  0,        m,    C[3]),
		isum(ii, _hho(N,   n, n+ii*n,         1,  ii+1,     m,    C[3])),
		         _hho(N,   n, n+mhf*n,        1,  mhf+1,    m, dd*C[4]), 
		isum(jj, _hho(N,   n, 2*n+mhf*n+jj*n, 1,  jj+mhf+2, m,    C[3]))
	    );
	end
    ),

    DFT_SR123 := rec(
	isApplicable := P -> P[1] mod 4 = 0,
	maxRadix := 4,
	allChildren := (self, P) >> let(k:=P[2], 
	    Map2(Filtered(DivisorPairs(P[1]), dd -> (dd[1] mod 2=0 and dd[1]<=self.maxRadix) and (dd[2] mod 2 = 0)),
	    (m,n) -> [DFT(m,k), DFT3(m,k), DFT(n,k), DFT2(n, k)])), 

	inplace := Inplace, 
	rule := meth(self,P,C)
	    local N,m,n,k,nhf,mhf, i,j,ii,jj,q,dd,isum, tw1a, tw1b, tw2a, tw2b, d1;

	    N := P[1]; m:=Cols(C[1]); n:=Cols(C[3]); k:=P[2]; nhf:=(n-2)/2; mhf:=(m-2)/2;
	    i := Ind(nhf); j := Ind(nhf); ii := Ind(mhf); jj := Ind(mhf); 
	    q := Ind(n);
	    dd := Diag(Lambda(q, cond(eq(q,n/2), (-E(4))^k, 1))); 
	    isum := (var, expr) -> When(var.range=0, [], ISum(var, expr));
	
	    tw1a := fPrecompute(fCompose(dOmega(N,k), dLin(mhf, i+1, i+1,     TInt)));
	    tw1b := fPrecompute(fCompose(dOmega(N,k), dLin(mhf, i+1, (i+1)*(mhf+2), TInt)));

	    tw2a := fPrecompute(fCompose(dOmega(N,k), dLin(mhf, j+2+nhf, j+2+nhf,     TInt)));
	    tw2b := fPrecompute(fCompose(dOmega(N,k), dLin(mhf, j+2+nhf, (j+2+nhf)*(mhf+2), TInt)));
	    
	    d1 := fConst(1, 1.0);

	    return
	    self.inplace(
	    SUM(         _hhi(N,   m, 0,       n,   C[1]),
		isum(i,  _hhi(N,   m, i+1,     n,   C[1] * When(tw1a.domain()=0, I(2), Diag(diagDirsum(d1, tw1a, d1, tw1b))))),
		         _hhi(N,   m, nhf+1,   n,   C[2]),
		isum(j,  _hhi(N,   m, j+nhf+2, n,   C[1] * When(tw2a.domain()=0, I(2), Diag(diagDirsum(d1, tw2a, d1, tw2b)))))
	    )) * 
	    SUM(         _hho(N,   n, 0,              1,  0,        m,    C[3]),
		isum(ii, _hho(N,   n, n+ii*n,         1,  ii+1,     m,    C[3])),
		         _hho(N,   n, n+mhf*n,        1,  mhf+1,    m,    C[4]), #dd * C[4]),
		isum(jj, _hho(N,   n, 2*n+mhf*n+jj*n, 1,  jj+mhf+2, m,    C[3]))
	    );
	end
    )
));

 RulesFor(DFT3, rec(
    DFT3_CT_Twids := Inherit(DFT_CT, rec(
 	switch := false,
	maxRadix := 4,
 	allChildren  := (self, P) >> Map2(Filtered(DivisorPairs(P[1]), dd->dd[1] <= self.maxRadix), 
 	    (m,n) -> [ DFT(m, P[2]), DFT3(n, P[2]) ]),
       
	inplace := Inplace,
 	rule := meth(self, P,C) 
	    local N, m, n, k, i, tw;
	    N := P[1]; 
	    m := Cols(C[1]); n := Cols(C[2]); k := P[2]; 
	    i := Ind(n);
	    tw := fPrecompute(fCompose(dOmega(2*N,k), dLin(m-1, 2*i+1, 2*i+1,  TInt)));

	    return self.inplace(ISum(i, _hhi(N,  m,  i,  n,  C[1] * Diag(diagDirsum(fConst(1,1.0), tw))))) *
 	           Tensor(I(m), C[2]) * 
		   L(N, m);
 	end
     )),

    DFT3_SR34_Corr := rec(
	isApplicable := P -> P[1] mod 4 = 0,
	allChildren := P -> let(k:=P[2], 
	    Map2(Filtered(DivisorPairs(P[1]), dd -> (dd[1] mod 2=0) and (dd[2] mod 2 = 0)),
	    (m,n) -> [DFT(m,k), DFT3(n,k), DFT4(n, k)])), 

	rule := function(P,C)
	    local N,m,n,k,nhf,mhf, i,j,ii,jj,q,dd,isum, tw1a, tw1b, tw2a, tw2b;

	    N := P[1]; m:=Cols(C[1]); n:=Cols(C[3]); k:=P[2]; nhf:=(n-2)/2; mhf:=(m-2)/2;
	    i := Ind(n); ii := Ind(mhf); jj := Ind(mhf); 
	    isum := (var, expr) -> When(var.range=0, [], ISum(var, expr));
	
	    return
            isum(i,  _hhi(N,   m, i,     n,   C[1] * Diag(fPrecompute(UTwid1(N,m,k,1/2,0,i))))) * 
	    SUM(         _hho(N,   n, 0,              1,  0,        m,    C[2]),
		isum(ii, _hho(N,   n, n+ii*n,         1,  ii+1,     m,    C[2])),
		         _hho(N,   n, n+mhf*n,        1,  mhf+1,    m,    C[3]), 
		isum(jj, _hho(N,   n, 2*n+mhf*n+jj*n, 1,  jj+mhf+2, m,    C[2]))
	    );
	end
    ),

 ));

RulesFor(UUDFT, rec(
   UUDFT1_Base4 := BaseRule(UUDFT1, [4, @])
));

RulesFor(UDFT1, rec(
   UDFT1_Base2 := rec(
       isApplicable := P -> P[1] = 2,
       rule := (P, C) -> I(2)
   ),

   UDFT1_Base4 := rec(
       isApplicable := P -> P[1] = 4,
       rule := (P, C) -> 
           DirectSum(I(2), F(2))^L(4,2) *
	   Diag(Tw1(4, 2, 1)) * 
	   Tensor(I(2), F(2)) * 
	   L(4, 2)
   ),

   UDFT1_toDFT := rec(
       isApplicable := P -> P[1] > 4,
       allChildren := P -> [[ DFT(P[1], P[2]) ]],
       rule := (P, C) -> 
           DirectSum((1/2)*F(2), I(P[1]-2))^L(P[1],P[1]/2) * C[1]
   ),

   UDFT1_SR_Reg := rec(
	isApplicable := P -> Is2Power(P[1]) and P[1] > 4,

	allChildren := P -> let(k:=P[2], Map2(Filtered(DivisorPairs(P[1]), dd->dd[2]>2),
	    (m,n) -> [DFT(m,k), UDFT(2*m,k), UDFT(n,k)])),

	forTransposition := false,

	rule := (P,C) -> let( # we know that n >= 4
	    N := P[1], m:=Cols(C[1]), n:=Cols(C[3]), k:=P[2], j := Ind((n-2)/2), jj := Ind((n-2)/2), 
	    SUM(
		Scat(H(N,2*m,0,n/2)) * C[2] * L(2*m,2) * Gath(H(N,2*m,0,n/2)),
		ISum(j, 
		     Scat(H(N,m,j+1,n)) *
		     C[1] * Diag(Twid(N,m,k,0,0,j+1)) *
		     Gath(H(N, m,j+1,n))),
		ISum(jj, 
		     Scat(H(N,m,jj+2+(n-2)/2,n)) *
		     C[1] * Diag(Twid(N,m,k,0,0,jj+2+(n-2)/2)) *
		     Gath(H(N, m,jj+2+(n-2)/2,n)))

	    ) * 
	    Tensor(I(m), C[3]) * L(N,m))
    )
));

#Unbind(UDFT.transpose);
#SwitchRules(DFT, [1,3,18]);
#DFT_CT.isApplicable := P -> P[1]=4;
