
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(DTTBase, NonTerminal, rec(
    abbrevs := [ N -> Checked(IsInt(N), N >= 1, [N]) ],
    dims := self >> [self.params[1], self.params[1]],
    isReal := True,

    normalizedArithCost := self >> let(n:=Double(self.params[1]),
        2.5*n*d_log(n)/d_log(2)),

    hashAs := self >> self
));

Declare(DCT3, DCT7);

#F DCT1(<n>) - Discrete Cosine Transform, Type I, non-terminal
#F Definition: (n x n)-matrix [ cos(k*l*pi/(n-1)) | k,l = 0...n-1 ]
#F Note:       The natural size for a DCT1 is 2^k + 1
#F Example:    DCT1(9)
#F 
Class(DCT1, DTTBase, rec(
    terminate := self >> Mat(DCT_Iunscaled(self.params[1])),
    transpose := self >> Copy(self), 
    SmallRandom := () -> Random([3,4,5,7,9,13,17,25,33])
));

#F DCT2(<n>) - Discrete Cosine Transform, Type II, non-terminal
#F Definition: (n x n)-matrix [ cos(k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F Note:       DCT2 is the transpose of DCT3
#F Example:    DCT2(8)
Class(DCT2, DTTBase, rec(
    terminate := self >> Mat(DCT_IIunscaled(self.params[1])),
    transpose := self >> DCT3(self.params[1]),
    SmallRandom := () -> Random([2,3,4,5,6,8,9,10,12,15,16,18,24,27,30,32])
));

#F DCT3(<n>) - Discrete Cosine Transform, Type III, non-terminal (unscaled)
#F Definition: (n x n)-matrix [ cos((k+1/2)*l*pi/n) | k,l = 0...n-1 ]
#F  [scaled]   (n x n)-matrix [ a_l*cos((k+1/2)*l*pi/n) | k,l = 0...n-1 ]
#F                              a_j = 1/sqrt(2) for j = 0 and = 1 else
#F Note:       DCT3 is the transpose of DCT2, scaled NOT supported yet
#F Example:    DCT3(8)
Class(DCT3, DTTBase, rec(
    terminate := self >> Mat(DCT_IIIunscaled(self.params[1])),
    transpose := self >> DCT2(self.params[1]),
    SmallRandom := () -> Random([2,3,4,5,6,8,9,10,12,15,16,18,24,27,30,32])
));

#F DCT4(<size>) - Discrete Cosine Transform, Type IV, non-terminal
#F Definition: (n x n)-matrix  [ cos((k+1/2)*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F Note:       DCT4 is symmetric
#F Example:    DCT4(8)
Class(DCT4, DTTBase, rec(
    terminate := self >> Mat(DCT_IVunscaled(self.params[1])),
    transpose := self >> Copy(self),
    SmallRandom := () -> Random([2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32])
));

#F DCT5(<size>) - Discrete Cosine Transform, Type V, non-terminal
#F Definition: (n x n)-matrix   [ cos(k*l*pi/(n-1/2)) | k,l = 0...n-1 ]
#F Note:       DCT5 is symmetric
#F Example:    DCT5(8)
Class(DCT5, DTTBase, rec(
    terminate := self >> Mat(DCT_Vunscaled(self.params[1])),
    transpose := self >> Copy(self),
));

#F DCT6(<size>) - Discrete Cosine Transform, Type VI, non-terminal
#F Definition: (n x n)-matrix   [ cos(k*(l+1/2)*pi/(n-1/2)) | k,l = 0...n-1 ]
#F Note:       The transpose of DCT6 is DCT7
#F Example:    DCT6(8)
Class(DCT6, DTTBase, rec(
    terminate := self >> Mat(DCT_VIunscaled(self.params[1])),
    transpose := self >> DCT7(self.params[1]),
));

#F DCT7(<size>) - Discrete Cosine Transform, Type VII, non-terminal
#F Definition: (n x n)-matrix   [ cos((k+1/2)*l*pi/(n-1/2)) | k,l = 0...n-1 ]
#F Note:       The transpose of DCT7 is DCT6
#F Example:    DCT7(8)
Class(DCT7, DTTBase, rec(
    terminate := self >> Mat(DCT_VIIunscaled(self.params[1])),
    transpose := self >> DCT6(self.params[1]),
));

#F DCT8(<size>) - Discrete Cosine Transform, Type VIII, non-terminal
#F Definition: (n x n)-matrix [ cos((k+1/2)*(l+1/2)*pi/(n+1/2)) | k,l = 0...n-1 ]
#F Note:       DCT8 is symmetric
#F Example:    DCT8(8)
Class(DCT8, DTTBase, rec(
    terminate := self >> Mat(DCT_VIIIunscaled(self.params[1])),
    transpose := self >> Copy(self)
));
