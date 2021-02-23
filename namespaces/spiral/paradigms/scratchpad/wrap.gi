
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(ScratchWrap, rec(
    __call__ := (self, size, nsgmts, linesize) >> Checked(
        Is2Power(size), Is2Power(nsgmts), Is2Power(linesize),
       
        WithBases(self, rec(
            operations := PrintOps, 
            size := size, 
            nsgmts := nsgmts, 
            linesize := linesize
        ))
    ),

    wrap := (self, r, t, opts) >> let(
        tags := [ALStore(self.size, 1, self.linesize)] 
            :: When(IsBound(r.node) and IsBound(r.node.getTags), r.node.getTags(), []),
        tt := GT(r.node, GTPar, GTPar, [1]).withTags(tags),
        rr := AllApplicableRules(tt, opts.breakdownRules),
                
        Checked(rr <> [], 
            rr[1](tt, r))
    ),

    twrap := (self, t, opts) >> let(
        tags := [ALStore(self.size, 1, self.linesize)] 
            :: When(IsBound(t._children) and IsBound(t.child(1).getTags), t.child(1).getTags(), []),
        tt := TTensorI(t, self.nsgmts, APar, APar).withTags(tags),

        tt
    ),
    
    print := (self) >> Print(self.name, "(", self.size, ", ", self.nsgmts, ", ", self.linesize, ")"),
));

