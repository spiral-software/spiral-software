
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


vtransposelo := (l1,l2,n) -> Flat(List([1..n/2], i-> [l1[2*i-1],l2[2*i-1]]
));

vtransposehi := (l1,l2,n) -> Flat(List([1..n/2], i-> [l1[2*i],l2[2*i]]
));

vrev64 := (l1,n) -> Flat(List([1..n/2], i-> [l1[2*i],l1[2*i-1]]
));
