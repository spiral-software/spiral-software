
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(AStream, rec(
#D Class(AStream, ATag, rec(
#D    isStream := true,
    __call__ := (self, bs) >> WithBases(self, rec(bs:=bs)),
    print := (self) >> Print(self.name, "(", self.bs, ")"),
    operations := Inherit(PrintOps, rec(\= := (self, other) >> ObjId(other) = ObjId(self) and self.bs = other.bs)),
    #?? legal_kernel := (self,p) >> self.bs <= 2^p

    # If we have a block size of 1, allow a radix 2 kernel so we can fold further.
    legal_kernel := (self, p) >> Cond(self.bs=1, p=2, p <= self.bs),

    # BWD: Somehow this becomes a tag, so we need .kind
    kind := self >> ObjId(self)
));
