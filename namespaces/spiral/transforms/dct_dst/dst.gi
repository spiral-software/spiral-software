
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(DST3, DST7);

#F DST1(<n>) - Discrete Sine Transform, Type I, non-terminal
#F Definition: (n x n)-matrix [ sin((k+1)*(l+1)*pi/(n+1)) | k,l = 0...n-1 ]
#F Note:       The natural size for a DST1 is 2^k - 1
#F Note:       DST4 is symmetric
#F Example:    DST1(7)
#F 
Class(DST1, DTTBase, rec(
    terminate := self >> When(self.params[1] > 1, Mat(DST_Iunscaled(self.params[1])), I(1)),
    transpose := self >> Copy(self), 
    SmallRandom := () -> Random([2,3,5,7,11,15,23,31])
));

#F DST2(<n>) - Discrete Sine Transform, Type II, non-terminal
#F Definition: (n x n)-matrix [ sin(k*(l+1/2)*pi/n) | k,l = 0...n-1 ]
#F Note:       DST2 is the transpose of DST3
#F Example:    DST2(8)
Class(DST2, DTTBase, rec(
    terminate := self >> Mat(DST_IIunscaled(self.params[1])),
    transpose := self >> DST3(self.params[1]),
    SmallRandom := () ->  Random([2,3,4,6,8,9,12,16,18,24,27,32])
));

#F DST3(<n>) - Discrete Sine Transform, Type III, non-terminal
#F Definition: (n x n)-matrix [ sin((k-1/2)*l*pi/n) | k,l = 1...n ]
#F Note:       DST3 is the transpose of DST2
#F Example:    DST3(8)
#F Scaled variant (not supported) is:
#F                [ a_l*sin(k*(l-1/2)*pi/n) | k,l = 1...n ]
#F            with  a_j = 1/sqrt(2) for j = n and = 1 else.
Class(DST3, DTTBase, rec(
    terminate := self >> Mat(DST_IIIunscaled(self.params[1])),
    transpose := self >> DST2(self.params[1]),
    SmallRandom := () ->  Random([2,3,4,6,8,9,12,16,18,24,27,32])
));

#F DST4(<n>) - Discrete Sine Transform, Type IV, non-terminal
#F Definition: (n x n)-matrix [ sin((k-1/2)*(l-1/2)*pi/n) | k,l = 1...n ]
#F Note:       DST4 is symmetric
#F Example:    DST4(8)
Class(DST4, DTTBase, rec(
    terminate := self >> Mat(DST_IVunscaled(self.params[1])),
    transpose := self >> Copy(self), 
    SmallRandom := () ->  Random([2,3,4,6,8,9,12,16,18,24,27,32])
));

#F DST5(<n>) - Discrete Sine Transform, Type V, non-terminal
#F Definition: (n x n)-matrix [ sin((k+1)*(l+1)*pi/(n+1/2)) | k,l = 0...n-1 ]
#F Note:       DST5 is symmetric
#F Example:    DST5(8)
Class(DST5, DTTBase, rec(
    terminate := self >> Mat(DST_Vunscaled(self.params[1])),
    transpose := self >> Copy(self), 
));


#F DST6(<n>) - Discrete Sine Transform, Type VI, non-terminal
#F Definition: (n x n)-matrix [ sin((k+1)*(l+1/2)*pi/(n+1/2)) | k,l = 0...n-1 ]
#F Note:       The transpose of DST6 is DST7
#F Example:    DST6(8)
Class(DST6, DTTBase, rec(
    terminate := self >> Mat(DST_VIunscaled(self.params[1])),
    transpose := self >> DST7(self.params[1])
));

#F DST7(<n>) - Discrete Sine Transform, Type VII, non-terminal
#F Definition: (n x n)-matrix [ sin((k+1/2)*(l+1)*pi/(n+1/2)) | k,l = 0...n-1 ]
#F Note:       The transpose of DST7 is DST6
#F Example:    DST7(8)
Class(DST7, DTTBase, rec(
    terminate := self >> Mat(DST_VIIunscaled(self.params[1])),
    transpose := self >> DST6(self.params[1]) 
));

#F DST8(<n>) - Discrete Sine Transform, Type VIII, non-terminal
#F Definition: (n x n)-matrix [ sin((k+1/2)*(l+1/2)*pi/(n-1/2)) | k,l = 0...n-1 ]
#F Note:       DST8 is symmetric
#F Example:    DST8(8)
Class(DST8, DTTBase, rec(
    terminate := self >> Mat(DST_VIIIunscaled(self.params[1])),
    transpose := self >> Copy(self), 
));
