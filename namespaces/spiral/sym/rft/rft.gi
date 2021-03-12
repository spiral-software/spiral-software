
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


ImportAll(spiral);

alphai := (n,i,a) ->  Cond(
    i mod 2 = 0, (  a + Int(i/2)) / n, 
    i mod 2 = 1, (1-a + Int(i/2)) / n);

#alphai := (n,i,alpha) -> fr(n,i,2*alpha).ev()/2;
cat := Concatenation;
cos := x->CosPi(2*x);
sin := x->SinPi(2*x);

ls := (n,f) -> List([0..n-1], f);

MatRDFT1 := n -> cat(
     [ List([0..n-1], x->1), List([0..n-1], x->(-1)^x) ],
     cat(List([1..n/2-1], i->
                    [ ls(n, j -> cos(j*i/n)), 
                      ls(n, j -> sin(j*i/n)) ])));

MatRDFT3 := n -> 
     cat(List([0..n/2-1], i->
                    [ ls(n, j -> cos(j*(i+1/2)/n)), 
                      ls(n, j -> sin(j*(i+1/2)/n)) ]));

MatRDFT := (N,a) -> let(n:=N/2,
     cat(ls(n, i-> let(aa := alphai(n,i,a), # 2 since we use 2pi*a
                    [ ls(2*n, j -> cos(j*aa)), 
                      ls(2*n, j -> sin(j*aa)) ]))));

MatrDFT := (N,a) -> let(n:=N/2,
     cat(ls(n, i-> let(aa := alphai(n,i,a),
                    [ cat(ls(n, j->cos(j*aa)), (-1)^i*ls(n, j -> -sin(j*aa))),
                      cat(ls(n, j->sin(j*aa)), (-1)^i*ls(n, j ->  cos(j*aa))) ]))));

MatBRDFT := (N,a) -> MatSPL(BlockMat(let(n:=N/2, 
     ls(n, i->let(phi := [[0,-1],[1,2*cos(alphai(n,i,a))]],
	 ls(n, j->Mat(phi^(2*j))))))));


