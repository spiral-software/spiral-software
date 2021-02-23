
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# This file contains code relevant to arbitrary bit matrices.
# At some point, Marek and I should make one file of this sort that we can both use. (PM)


##    BitMatrixToInts(P) returns bit matrix converted into integers (0, 1).
##
BitMatrixToInts := meth(x)
   return List([1 .. Length(x)], i-> IntVecFFE(x[i]));
end;


##   PrintBitMatrix(P) prints bit matrix P in a readable form.
##
PrintBitMatrix := meth(x)
   PrintMat(BitMatrixToInts(x));
end;


##   NumberBitRep(x, b) returns a b-bit number, formed as a vector over GF(2),
##       corresponding to decimal number x.
##
NumberBitRep := meth(x, bits)
   return Reversed(List([1..bits], i -> [ Mod(QuoInt(x, 2^(i-1)), 2) * GF(2).one ] ));
end;


##   BitsToNumber(x) returns the decimal number associated with bit vector x.
##
BitsToNumber := meth(x)
   return Sum(List([1..Length(x)], i -> 2^(i-1) * IntVecFFE(Reversed(x)[i])[1])); 
end;


##    BasisVecToNum(b) returns a number corresponding to given basis
##       vector b.
##
BasisVecToNum := meth(b)
   return Sum(List([1..Length(b)], i -> Cond(IsList(b[i]), b[i][1], b[i]) * (i-1)));
end;


##   BitsToPermMatrix(P) returns the SPL permutation matrix corresponding
##      to bit matrix P.
##
BitsToPermMatrix := meth(P)
   local x, l;
   x := [];
   l := Dimensions(P)[1];

   x := List([1..2^l], i -> BasisVec(2^l, BitsToNumber(P * NumberBitRep(i-1, l))));

   return Transposed(x);
end;

Declare(TransposeVector);

##   PermMatrixToBits(P) returns the bit matrix associated with
##      permutation matrix P.
##
PermMatrixToBits := function(perm)
   local length, res, i, logLength, pm;

   length := Length(perm);
   logLength := Log2Int(length);

   res := [];
   for i in [0..logLength-1] do
      res[logLength-i] := 
         TransposeVector(
	    NumberBitRep(
	       BasisVecToNum(
	          perm * TransposeVector(BasisVec(length, 2^i))
	       ), logLength
	    )
	 );
   od;

   res := TransposedMat(res);

   pm := BitsToPermMatrix(res);

   if (Length(pm) <> Length(perm) or Maximum(Maximum(pm - perm)) <> 0) then
#       Print("ERROR: Given matrix is not a permutation or is not linear on the bits.\n");
       return -1;
   fi;   

   return res;
 end;


# ===============================================
# A permutation represented as a linear mapping on the bits
# LinearBits(matrix over GF(2), <SPL Permutation object>)
# ===============================================
Class(LinearBits, BaseMat, SumsBase, rec(
	new := (self, m, p) >> 
	   SPL(WithBases(self, 
		   rec(
		       dimensions := [2^Length(m), 2^Length(m)],
		       _children  := [m, p]
		   )
	)),

	rSetChild := meth(self, n, what) 
	   self._children[n] := what; 
	end,

	rChildren  := self >> self._children,
	child      := (self, n) >> self._children[n],
	children   := self >> self._children,

	createCode := self >> self,
        print      := (self,i,is) >> Print(self.name, "(", self._children[1], ")"),

	toAMat     := self >> AMatMat(BitsToPermMatrix(self.child(1))),
	dims       := self >> [2^(Length(self._children[1])), 2^(Length(self._children[1]))],

	sums       := self >> self
));


TL.permBits := 
    meth(self)
        local stride, logStride, size, logSize, res, row, i, j, a, b;
	stride    := self.params[2];
	logStride := Log2Int(stride);
	size      := self.params[1];
	logSize   := Log2Int(size);
	a         := Log2Int(self.params[3]);
	b         := Log2Int(self.params[4]);
	res       := [];

	for i in [1 .. a] do
	   row := Concatenation([ List([1 .. i-1],     i -> 0*GF(2).one), 
		                  [ GF(2).one ], 
				  List([i+1 .. a],     i -> 0*GF(2).one),
				  List([1 .. logSize], i -> 0*GF(2).one),
				  List([1 .. b],       i -> 0*GF(2).one) ]);

	   Append(res, [row]);
	od;

	for i in [logSize-logStride+1 .. logSize] do
	   row := Concatenation([ List([1 .. a],         i -> 0*GF(2).one),
		                  List([1 .. i-1],       i -> 0*GF(2).one),
				  [ GF(2).one ],
				  List([i+1 .. logSize], i -> 0*GF(2).one),
				  List([1 .. b],         i -> 0*GF(2).one)]);
	   
	   Append(res, [row]);
	od;
	
	for i in [1 .. logSize-logStride] do
	   row := Concatenation([ List([1 .. a],   i -> 0*GF(2).one),
		                  List([1 .. i-1], i -> 0*GF(2).one),
				  [ GF(2).one ],
				  List([i+1 .. logSize], i -> 0*GF(2).one),
				  List([1 .. b],         i -> 0*GF(2).one)]);

	   Append(res, [row]);
	od;

	for i in [1 .. b] do
	   row := Concatenation([ List([1 .. a],       i -> 0*GF(2).one),
		                  List([1 .. logSize], i -> 0*GF(2).one),
				  List([1 .. i-1],     i -> 0*GF(2).one),
				  [ GF(2).one ],
				  List([i+1 .. b],     i -> 0*GF(2).one)]);

	   Append(res, [row]);
	od;

	return res;
end;

DR.permBits := meth(self)
       local k,  n, res;
       k := LogInt(self.params[2], 2);
       n := LogInt(self.domain(), 2);
       return MatSPL(Tensor(J(n/k), I(k))) * GF(2).one;
end;