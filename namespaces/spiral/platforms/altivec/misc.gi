
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# NOTE: The first section is Intel SSE/2/3 specific, and has to be ported
# over to SPU

unpacklo := (l1,l2,n,k) -> Flat(List([1..n/(2*k)], i-> [
        List([1..k], j->l1[(i-1)*k+j]),
        List([1..k], j->l2[(i-1)*k+j])
]));

unpackhi := (l1,l2,n,k) -> Flat(List([1..n/(2*k)], i -> [
        List([1..k], j->l1[n/2+(i-1)*k+j]),
        List([1..k], j->l2[n/2+(i-1)*k+j])
]));

sparams := (l,n) -> List([1..l], i->[1..n]);

shuffle := (in1, in2, p, n, k) -> Flat([
    List([1..n/(2*k)],     i->List([1..k], j->in1[(p[i]-1)*k+j])),
    List([n/(2*k)+1..n/k], i->List([1..k], j->in2[(p[i]-1)*k+j]))
]);

iperm4 := self >> Filtered(Cartesian(self.params()), i->i[1]<>i[2] and i[3] <> i[4]);
