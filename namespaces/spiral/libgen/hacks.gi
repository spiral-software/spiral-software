
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# b/s parameters
LibgenParametrizeStrides := function()
    H.codeletParNums := [3, 4];
    H.signature := self >> [IntVar("b"), IntVar("s")];
    H.mkCodelet := self >> let(ss := self.signature(), H(self.params[1], self.params[2], ss[1], ss[2]));
    H.codeletShape := self >> ObjId(self);

    BH.codeletParNums  := [2, 4, 5];
    BH.signature    := self >> List(["R", "b", "s"], IntVar);
    BH.codeletShape    := self >> ObjId(self);
end;

# b parameter
LibgenHardcodeStrides := function()
    H.codeletShape := self >> [H, self.params[4]];
    H.codeletParNums := [3];
    H.signature := self >> [IntVar("b")];
    H.mkCodelet := self >> let(ss := self.signature(), H(self.params[1], self.params[2], ss[1], self.params[4]));

    BH.codeletParNums := [2, 4];
    BH.signature := self >> List(["R", "b"], IntVar);
    BH.codeletShape := self >> [BH, self.params[5]];
end;

# LibgenHardcodeBase := function()
#     H.codeletShape := self >> [H, self.params[4]];
#     H.codeletParNums := [4];
#     H.signature := self >> [IntVar("s")];
#     H.mkCodelet := self >> let(ss := self.signature(), H(self.params[1], self.params[2], 0, ss[1]));

#     BH.codeletParNums := [2, 4, 5];
#     BH.signature := self >> List(["R", "b", "s"], IntVar);
#     BH.codeletShape := self >> ObjId(self);
# end;

# LibgenHardcodeBaseStrides := function()
#     H.codeletShape := self >> [H, self.params[4]];
#     H.codeletParNums := [];
#     H.signature := self >> [];
#     H.mkCodelet := self >> let(ss := self.signature(), H(self.params[1], self.params[2], 0, self.params[4]));

#     BH.codeletParNums := [2, 4];
#     BH.signature := self >> List(["R", "b"], IntVar);
#     BH.codeletShape := self >> [BH, self.params[5]];
# end;

LibgenHardcodeStrides();

#F save_bf(<bf>, <filename>) - saves hash table <bf> to <filename>
#F   the variable in the file will be called 'bf', the function takes care
#F   to correctly unparse generated code, so that it can be reloaded later.
save_bf := function(bf, fname)
    var.print := var.printFull;
    HashSave(bf, fname);
    AppendTo(fname, "bf:=savedHashTable;\n");
    var.print := var.printShort;
end;

stringNT := nt -> Concatenation(nt.name, String(Cols(nt)));

#clets  := x -> List(CollectRecursSteps(x), rs -> CodeletShape(rs.child(1)));
#csigns := x -> List(CollectRecursSteps(x), rs -> CodeletSignature(rs.child(1)));
#cpara  := x -> List(CollectRecursSteps(x), rs -> CodeletParams(rs.child(1)));
