
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(IdFunc, DiagFunc, rec(
    range := self >> TInt
));

#F fConst(<N>, <c>) - constant diagonal function, <N> values of <c>
#F
Class(idConst, IdFunc, rec(
#    def := (N, c) -> Checked(IsPosInt0(N), rec(size:=N)),
    def := (N, c) -> rec(size:=N),
    inline := true,
    lambda := self >> let(i:=Ind(self.size), Lambda(i, self.params[2])),
    isReal := self >> IsRealNumber(self.params[2]),
    domain := self >> self.params[1],
));

Class(idId, IdFunc, rec(
    def := size -> rec(size := size),
    lambda := self >> let(i := Ind(self.size), Lambda(i,i)),
    #inverse := i->i,
    transpose := self >> self,
    domain := self >> self.params[1],
));

#NOTE Derive the idTensor from diagTensor
Class(idTensor, fTensorBase, rec(
    print := FuncClassOper.print,
#    range := self >> UnifyTypes(List(self.children(), x->x.range())),
    combine_op := (self, jv, split, f, g) >> f.relaxed_at(jv, split[1]) * g.relaxed_at(jv, split[2]),

    updateRange := self >> UnifyTypes(List(self.children(), x->x.range())),
    range       := self >> self.updateRange(),
#    combine_op := (self, split, F, G) >> F.at(split[1]) * G.at(split[2])
    # idTensor is something else again ???
#    combine_op := (self, split, F, G) >> self.child(2).domain() * F.at(split[1]) + G.at(split[2])
));


#Class(idfTensor, fTensorBase, rec(
#    print := FuncClassOper.print,
#    updateRange := self >> UnifyTypes(List(self.children(), x->x.range())),
#    combine_op := (self, split, F, G) >> self.child(2).domain() * F.at(split[1]) + G.at(split[2])
#));
#


Class(idCompose, fCompose);


Class(Id, Diag, rec(
  toAMat := self >> IdentityMatAMat(self.element.domain()),
  domain := self >> self.element.domain()
));
