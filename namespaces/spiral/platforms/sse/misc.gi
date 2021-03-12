
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


unpacklo := (l1,l2,n,k) -> Flat(List([1..n/(2*k)], i-> [
        List([1..k], j->l1[(i-1)*k+j]),
        List([1..k], j->l2[(i-1)*k+j])
]));

unpackhi := (l1,l2,n,k) -> Flat(List([1..n/(2*k)], i -> [
        List([1..k], j->l1[n/2+(i-1)*k+j]),
        List([1..k], j->l2[n/2+(i-1)*k+j])
]));

# saturation in pack_semantic is not taken into account
pack_semantic := (l1, l2, n) -> Flat(List( [1..n], i -> [l1[i], l2[i]]));

sparams := (l,n) -> List([1..l], i->[1..n]);

shuffle := (in1, in2, p, n, k) -> Flat([
    List([1..n/(2*k)],     i->List([1..k], j->in1[(p[i]-1)*k+j])),
    List([n/(2*k)+1..n/k], i->List([1..k], j->in2[(p[i]-1)*k+j]))
]);

inverse_ushuffle := (inp, p, n) -> List([1..n], i->inp[p[i]]);

shufflehi := (in1, p, n, k) -> Concat(
    Sublist(in1, [1..n/2]),
    let(l := Sublist(in1, [n/2+1..n]), shuffle(l, l, p, n/2, k)));

shufflelo := (in1, p, n, k) -> Concat(
    let(l := Sublist(in1, [1..n/2]), shuffle(l, l, p, n/2, k)),
    Sublist(in1, [n/2+1..n]));

iclshuffle := p -> Print("_MM_SHUFFLE", When(Length(p)=2, "2", ""), "(", PrintCS(Reversed(p-1)), ")");

iclprintop := self >> Print(self.icl, "(", _vcprintcs(self.args), ")");

iperm4 := self >> Filtered(Cartesian(self.params()), i->i[1]<>i[2] and i[3] <> i[4]);

vtakehi := (v) -> Checked(IsValue(v) and IsVecT(v.t) and v.t.size>1,  TakeLast(v.v, Length(v.v)/2));
vtakelo := (v) -> Checked(IsValue(v) and IsVecT(v.t) and v.t.size>1,  Take(v.v, Length(v.v)/2));

