
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Declare(VScat, VScat_sv, RCVScat_sv, VScat_pc, VScat_red, IxVScat_pc, IxRCVScat_pc, VStretchScat, RCVStretchScat, VScat_zero, VGath_u, VScat_u, vRCStretchGath, vRCStretchScat);

_rounddown := (n,v) -> v*idiv(n,v);
_roundup   := (n,v) -> v*idiv(n+v-1,v);
_res       := (n,v) -> imod(n, v); # == n - _rounddown(n, v) == n - v * idiv(n, v);


NeedInterleavedLeft := e->IsBound(e.needInterleavedLeft) and e.needInterleavedLeft();
NeedInterleavedRight := e->IsBound(e.needInterleavedRight) and e.needInterleavedRight();

#   NOTE: Is that right??
Compose.unroll := self >> Compose(List(self.children(), i-> When(IsBound(i.unroll), i.unroll(), i)));

Compose.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
Compose.needInterleavedRight := self >> Last(self.children()).needInterleavedRight();
Compose.cannotChangeDataFormat := self >> ForAny(self.children(), f->IsBound(f.cannotChangeDataFormat) and f.cannotChangeDataFormat());
Compose.totallyCannotChangeDataFormat := self >> ForAll(self.children(), f->IsBound(f.totallyCannotChangeDataFormat) and f.totallyCannotChangeDataFormat());

ISum.needInterleavedLeft := self >> self.child(1).needInterleavedLeft();
ISum.needInterleavedRight := self >> self.child(1).needInterleavedRight();
ISum.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
ISum.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();

SUM.needInterleavedLeft := self >> ForAny(self.children(), i->i.needInterleavedLeft());
SUM.needInterleavedRight := self >> ForAny(self.children(), i->i.needInterleavedRight());
SUM.cannotChangeDataFormat := self >> ForAny(self.children(), f->IsBound(f.cannotChangeDataFormat) and f.cannotChangeDataFormat());
SUM.totallyCannotChangeDataFormat := self >> ForAll(self.children(), f->IsBound(f.totallyCannotChangeDataFormat) and f.totallyCannotChangeDataFormat());

Prm.needInterleavedLeft := True;
Prm.needInterleavedRight := True;


Buf.needInterleavedLeft := self >> NeedInterleavedLeft(self.child(1));
Buf.needInterleavedRight := self >> NeedInterleavedRight(self.child(1));
Buf.cannotChangeDataFormat := self >> self.child(1).cannotChangeDataFormat();
Buf.totallyCannotChangeDataFormat := self >> self.child(1).totallyCannotChangeDataFormat();



Declare(VRCLR);
Declare(VReplicate);
Declare(VBlk);
Declare(VScat);
Declare(VGath);

Declare(VBlk);
Include(vtensor);
Include(vdirsum);
Include(perm);
Include(diag);
Include(gather);
Include(scatter);
Include(vrc);
Include(vblk);
Include(rader);
Include(vbase);
Include(vcontainer);
Include(vs);
Include(voj);
Include(ol);
Include(codegen);
Include(vtcast);
