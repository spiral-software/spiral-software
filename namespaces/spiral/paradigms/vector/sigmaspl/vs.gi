
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(VectorizedBaseMat, BaseMat, SumsBase, rec(
    transpose := self >> Inherit(self, rec(transposed := not self.transposed)),
    #-----------------------------------------------------------------------
    print := (self,i,is) >> Print(self.name, "(", self.n, ", ", self.v,")", 
	self.printA(), When(self.transposed, ".transpose()", "")),
));
 
#F VS(<n>, <v>)  --  vectorized version of doubly-diagonal matrix
#F 
#F Represents the matrix that has 1s on the main diagonal and upper shifted by 1 diagonal
#F
#F   <n> - matrix size
#F   <v> - vector length, does not affect matrix shape, only needed for Codegen 
#F
#F Example:
#F
#F spiral> PrintMat(MatSPL(VS(8,4)));
#F [ [ 1, 1,  ,  ,  ,  ,  ,   ], 
#F   [  , 1, 1,  ,  ,  ,  ,   ], 
#F   [  ,  , 1, 1,  ,  ,  ,   ], 
#F   [  ,  ,  , 1, 1,  ,  ,   ], 
#F   [  ,  ,  ,  , 1, 1,  ,   ], 
#F   [  ,  ,  ,  ,  , 1, 1,   ], 
#F   [  ,  ,  ,  ,  ,  , 1, 1 ], 
#F   [  ,  ,  ,  ,  ,  ,  , 1 ] ]
#F 
Class(VS, VectorizedBaseMat, rec(
    new := (self, n, v) >> SPL(WithBases(self, 
        rec(n := n, v := v, transposed := false))).setDims(),
    #-----------------------------------------------------------------------
    dims := self >> [self.n, self.n], 
    #-----------------------------------------------------------------------
    rChildren := self >> [self.n, self.v],
    rSetChild := rSetChildFields("n", "v"),
    #-----------------------------------------------------------------------
    area := self >> self.n * self.n,
    #-----------------------------------------------------------------------
    toAMat := self >> let(s:=S(self.n), AMatMat(MatSPL(When(self.transposed, s.transpose(), s))))
));

Declare(VLD);

#F VUD(<n>, <v>)  --  vectorized upper diagonal matrix
#F
#F VUD(n,v) is equivalent to UD(n,1)
#F
#F   <n> -- matrix size
#F   <v> -- vector length, does not affect matrix shape, only needed for Codegen 
#F
#F spiral> PrintMat(MatSPL(VUD(8,4)));
#F [ [  , 1,  ,  ,  ,  ,  ,   ], 
#F   [  ,  , 1,  ,  ,  ,  ,   ], 
#F   [  ,  ,  , 1,  ,  ,  ,   ], 
#F   [  ,  ,  ,  , 1,  ,  ,   ], 
#F   [  ,  ,  ,  ,  , 1,  ,   ], 
#F   [  ,  ,  ,  ,  ,  , 1,   ], 
#F   [  ,  ,  ,  ,  ,  ,  , 1 ], 
#F   [  ,  ,  ,  ,  ,  ,  ,   ] ]
#F
Class(VUD, VectorizedBaseMat, rec(
    new := (self, n, v) >> SPL(WithBases(self, rec(n := n, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := (self) >> VLD(self.n, self.v),
    #-----------------------------------------------------------------------
    dims := self >> [self.n, self.n], 
    #-----------------------------------------------------------------------
    rChildren := self >> [self.n, self.v],
    rSetChild := rSetChildFields("n", "v"),
    #-----------------------------------------------------------------------
    area := self >> self.n,
    #-----------------------------------------------------------------------
    toAMat := self >> UD(self.n).toAMat()
));

#F VLD(<n>, <v>)  --  vectorized lower diagonal matrix
#F
#F VLD(n,v) is equivalent to LD(n,1)
#F
#F   <n> -- matrix size
#F   <v> -- vector length, does not affect matrix shape, only needed for Codegen 
#F
#F spiral> PrintMat(MatSPL(VLD(8,4)));
#F [ [  ,  ,  ,  ,  ,  ,  ,   ], 
#F   [ 1,  ,  ,  ,  ,  ,  ,   ], 
#F   [  , 1,  ,  ,  ,  ,  ,   ], 
#F   [  ,  , 1,  ,  ,  ,  ,   ], 
#F   [  ,  ,  , 1,  ,  ,  ,   ], 
#F   [  ,  ,  ,  , 1,  ,  ,   ], 
#F   [  ,  ,  ,  ,  , 1,  ,   ], 
#F   [  ,  ,  ,  ,  ,  , 1,   ] ]
#F
Class(VLD, VectorizedBaseMat, rec(
    new := (self, n, v) >> SPL(WithBases(self, rec(n := n, v := v))).setDims(),
    #-----------------------------------------------------------------------
    transpose := (self) >> VUD(self.n, self.v),
    #-----------------------------------------------------------------------
    dims := self >> [self.n, self.n], 
    #-----------------------------------------------------------------------
    rChildren := self >> [self.n, self.v],
    rSetChild := rSetChildFields("n", "v"),
    #-----------------------------------------------------------------------
    area := self >> self.n,
    #-----------------------------------------------------------------------
    toAMat := self >> LD(self.n, 1).toAMat()
));
