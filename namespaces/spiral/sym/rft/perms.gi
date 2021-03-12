
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# mn=16, m=4, n=4, h=3
ImportAll(spiral);

# even k, P permutation on pairs
 P := (km,k) -> (i -> let(m := km/k, h := Int((m+2)/2), mc:=Int((m+1)/2),
	 Cond(
	     i < h, k*i, 
	     i >= km/2-mc+1, k/2 + k*(i-(km/2-mc+1)), 
	     ((i-h) mod m) mod 2 = 0, k*((i-h) mod m)/2 + 1 + Int((i-h)/m),
		              k*((i-h) mod m)/2 + 1 + Int(k/2-1-(i-h)/m)
	 )));

# even k, P permutation on single el-s
 rP := (km,k) -> (i -> let(m := km/k, 
	 Cond(
	     i = 0, 0,
	     1 <= i and i < m, -1 + 2*k*Int((i+1)/2) + ((i-1) mod 2),
	     i >= km-m, k + 2*k*Int((i-(km-m))/2) - 1 + ((i-m) mod 2), 
	     (Int((i-m)/2) mod m) mod 2 = 0, k*(Int((i-m)/2) mod m) + 1 + 2*Int((i-m)/(2*m)) + (i-m) mod 2,
	                                     k*(Int((i-m)/2) mod m) + 1 + 2*Int((km-m-i)/(2*m)) + (i-m) mod 2
	 )));

# odd k, \hat{P} permutation on pairs
 Phat := (km,k) -> (i -> let(m := km/k, h := Int((m+2)/2),
	 Cond(
	     i < h, k*i, 
	     ((i-h) mod m) mod 2 = 0, Int(k*((i-h) mod m)/2) + 1 + Int((i-h)/m),
		                      Int(k*((i-h) mod m)/2) +  Int((k+1)/2-(i-h)/m)
	 )));

# odd k
 rPhat := (km,k) -> (i -> let(m := km/k,
	 Cond(
	     i = 0, 0,
	     1 <= i and i < m, -1 + 2*k*Int((i+1)/2) + ((i-1) mod 2),
	     (Int((i-m)/2) mod m) mod 2 = 0, k*(Int((i-m)/2) mod m) + 1 + 2*Int((i-m)/(2*m)) + (i-m) mod 2,
		                             k*(Int((i-m)/2) mod m) +     2*Int((km-i)/(2*m)) + (i-m) mod 2
	 )));

# odd k
 rQhat := (km,k) -> (i -> let(m := km/k,
	 Cond(
	     i >= km-m, k-1 + 2*k*Int((i-(km-m))/2) + (i-(km-m)) mod 2, 
	     (Int(i/2) mod m) mod 2 = 0, k*(Int(i/2) mod m) +     2*Int(i/(2*m)) + i mod 2,
	                                 k*(Int(i/2) mod m) + 1 + 2*Int((km-m-i)/(2*m)) + i mod 2
	 )));

# Pshow := (km,k)->let(N := Int(km/2)+1, pm(Perm(PermFunc(P(km,k),N),N)));
# rPshow := (km,k)->let(N := km, pm(Perm(PermFunc(rP(km,k),N),N)));
# Phatshow := (km,k)->let(N := Int((km+2)/2), pm(Perm(PermFunc(Phat(km,k),N),N)));

PermP := (m, k) -> Checked(IsEvenInt(k), PermFunc(rP(k*m,k),k*m)^-1);
PermPhat := (m, k) -> Checked(IsOddInt(k), PermFunc(rPhat(k*m,k),k*m)^-1);
PermQhat := (m, k) -> Checked(IsOddInt(k), PermFunc(rQhat(k*m,k),k*m)^-1);

p_4_4 := PermP(4,4);
p_4_8 := PermP(4,8);
p_8_4 := PermP(8,4);
p_8_8 := PermP(8,8);

p_3_4 := PermP(3,4);
p_5_4 := PermP(5,4);
p_5_8 := PermP(5,8);

phat_4_3 := PermPhat(4,3);
phat_4_5 := PermPhat(4,5);
phat_8_5 := PermPhat(8,5);
phat_8_7 := PermPhat(8,7);
phat_7_7 := PermPhat(7,7);

qhat_4_3 := PermQhat(4,3);
qhat_4_5 := PermQhat(4,5);
qhat_8_5 := PermQhat(8,5);
qhat_8_7 := PermQhat(8,7);
qhat_7_7 := PermQhat(7,7);
