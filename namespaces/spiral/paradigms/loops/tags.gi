
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(ABuf, AGenericTag, rec(
    __call__ := (self, bs) >> WithBases(self, rec(bs:=bs)),
    print := (self) >> When(IsBound(self.bs), Print(self.name, "(", self.bs, ")"), Print(self.name)),
#D    isBuffer := true,
    operations := Inherit(PrintOps, rec(\= := (self, other) >> ObjId(other) = ObjId(self) and (Same(other, self) or IsBound(self.bs) and IsBound(other.bs) and self.bs = other.bs)))
));

Class(AVecMem, AVec, rec(
    __call__ := (self, v) >> WithBases(self, rec(v:=v)),
    print := (self) >> When(IsBound(self.v), Print(self.name, "(", self.v, ")"), Print(self.name)),
#D    isMem := true,
    operations := Inherit(PrintOps, rec(\= := (self, other) >> ObjId(other) = ObjId(self) and (Same(other, self) or IsBound(self.v) and IsBound(other.v) and self.v = other.v)))
));

Class(AVecMemL, AVec, rec(
    __call__ := (self, v) >> WithBases(self, rec(v:=v)),
    print := (self) >> When(IsBound(self.v), Print(self.name, "(", self.v, ")"), Print(self.name)),
#D    isMemL := true,
    operations := Inherit(PrintOps, rec(\= := (self, other) >> ObjId(other) = ObjId(self) and (Same(other, self) or IsBound(self.v) and IsBound(other.v) and self.v = other.v)))
));

Class(AVecMemR, AVec, rec(
    __call__ := (self, v) >> WithBases(self, rec(v:=v)),
    print := (self) >> When(IsBound(self.v), Print(self.name, "(", self.v, ")"), Print(self.name)),
#D    isMemR := true,
    operations := Inherit(PrintOps, rec(\= := (self, other) >> ObjId(other) = ObjId(self) and (Same(other, self) or IsBound(self.v) and IsBound(other.v) and self.v = other.v)))
));
