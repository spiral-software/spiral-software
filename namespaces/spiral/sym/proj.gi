
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


apply_nt := (NT, input) -> 
    List(TransposedMat(MatSPL(NT) * TransposedMat([input]))[1], EvalScalar);

dft_a := [0, 0, 1/2, 1/2];
dft_b := [0, 1/2, 0, 1/2];

dft_algo := [ [DFT1, DFT1], # f1 -> f1, f1
              [DFT2, DFT1], # f2 -> f2, f1
	      [DFT1, DFT3], # f3 -> f1, f3
	      [DFT2, DFT3]  # f4 -> f2, f3
	    ];
twid := (type,N,m,i) -> List(
    Twid(N,m,1,dft_a[type], dft_b[type],i).lambda().tolist(), EvalScalar);

# F_N -> Tensor(F_m, I_n) * T(mn, n) * Tensor(I_m, F_n) * L(mn, m)
#
[n, inputs, dfts, inputs2, twids, tinputs2, outputs, out, tdfts] := [0,0,0,0,0,0,0,0,0];

dft_projection := function(type, N, m, input)
    n := N/m;
    inputs := List([0..m-1], x->part(input,m,x)); 
    dfts    := List(inputs, x->apply_nt(dft_algo[type][2](Length(x)), x));
    inputs2 := TransposedMat(dfts);
    twids := List([0..n-1], i -> twid(type,N,m,i));
    tinputs2 := List([1..n], i -> listmul(inputs2[i], twids[i]));
    outputs := List(tinputs2, x->apply_nt(dft_algo[type][1](Length(x)), x));
    return List([inputs, dfts, inputs2, tinputs2, outputs], x->List(x,csym));
end;

tdft_projection := function(type, N, m, input)
    n := N/m;
    inputs := List([0..n-1], x->part(input,n,x)); 
    dfts    := List(inputs, x->apply_nt(dft_algo[type][2](Length(x)), x));
    twids := List([0..n-1], i->twid(Cond(type=2,3,type=3,2,type),N,m,i)); 
    tdfts := List([1..n], i -> listmul(dfts[i], twids[i]));
    tinputs2 := TransposedMat(tdfts);
    outputs := List(tinputs2, x->apply_nt(dft_algo[type][1](Length(x)), x));
    return List([inputs, dfts, tdfts, tinputs2, outputs], x->List(x,csym));
end;

Declare(inp, p1, p2, imid, dft1, dft2);
dft1_radproj := function(N, input)
    local mid; #,imid, p1, p2, inp, out;
    mid := DFT_Rader.raderMid(N,1,PrimitiveRootMod(N));
    input := apply_nt(RR(N), input);
    p1 := [input[1]]; inp := Drop(input,1);
    dft1 := apply_nt(DFT(N-1,-1), inp);
    imid := apply_nt(mid, Concat(p1, dft1));
    p2 := [imid[1]]; imid := Drop(imid, 1); 
    dft2 := apply_nt(DFT(N-1,-1), imid);
    out := apply_nt(RR(N).transpose(), Concat(p2, dft2));
    return List([inp, dft1, imid, dft2, out], csym);
end;
# left transform in Rader rule: 
# jDFT1 -> IRDFT1 dirsum jIRDFT2
# jDFT1 -> (jDFT1   dirsum jDFT3  )^L (R^2 dirsum C^2 ... I^2)
# jDFT3 -> (jDFT1 D dirsum jDFT1 D)^L (R^2 dirsum C^2 ... R^2)
#
#
rcexp := l -> ConcatList(l, 
    e -> let(ee := Complex(e), [ReComplex(ee), ImComplex(ee)]));


# DFT_mn = CRT(m,n,1,1)^T (OS(m,n) F_m x I_n) (I_m x OS(n,m) F_n) CRT(m,n,1,1)
pf_projection := function(N, m, input)
    local i, j;
    n := N/m;
    inputs := List([0..m-1], j -> List([0..n-1], i->input[1+ (j*n + i*m) mod N]));
    dfts    := List(inputs, x->apply_nt(OS(n,m)*DFT(n), x));
    inputs2 := TransposedMat(dfts);
    outputs := List(inputs2, x->apply_nt(OS(m,n)*DFT(m), x));
    out := [1..N];
    for j in [0..n-1] do 
       for i in [0..m-1] do 
           out[1+ (j*m + i*n) mod N] := outputs[1+j][1+i];
       od;
    od;
    return List([inputs, dfts, inputs2, outputs], x->List(x,csym));
end;

