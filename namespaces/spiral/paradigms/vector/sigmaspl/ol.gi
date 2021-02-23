
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


## Vector Multiplication operator
## VOLMultiplication(<number inputs>, <length inputs>, <length vector>) 
Class(VOLMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [self.rChildren()[2] * self.rChildren()[3], StripList(Replicate(self.rChildren()[1], self.rChildren()[2] * self.rChildren()[3]))],
));

#F RCVOLMultiplication(<number inputs>, <length inputs>, <length vector>)
#F   is a vRC(VOLMultiplication(<number inputs>, <length inputs>, <length vector>/2))
#F   inputs/output is in inteleaved complex format: [re1, im1, re2, im2, ...]

Class(RCVOLMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [self.rChildren()[2] * self.rChildren()[3], StripList(Replicate(self.rChildren()[1], self.rChildren()[2] * self.rChildren()[3]))],
));

#F VRCOLMultiplication(<number inputs>, <length inputs>, <length vector>)
#F   inputs/output is in vector inteleaved complex format: 
#F     [[re1, re2, ..., re_v], [im1, im2, ..., im_v], [re_(v+1), ...], ...]

Class(VRCOLMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [2 * self.rChildren()[2] * self.rChildren()[3], StripList(Replicate(self.rChildren()[1], 2 * self.rChildren()[2] * self.rChildren()[3]))],
));

## VOLConjMultiplication(<number inputs>, <length inputs>, <length vector>) 
Class(VOLConjMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [self.rChildren()[2] * self.rChildren()[3], StripList(Replicate(self.rChildren()[1], self.rChildren()[2] * self.rChildren()[3]))],
));

#F RCVOLConjMultiplication(<complex input length>, <length vector>)
#F   is a vRC(VTensor(OLConjMultiplication(<complex input length>), <length vector>/2))
#F   inputs/output is in inteleaved complex format: [re1, im1, re2, im2, ...]

Class(RCVOLConjMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [self.rChildren()[2] * self.rChildren()[3], StripList(Replicate(self.rChildren()[1], self.rChildren()[2] * self.rChildren()[3]))],
));

#F VRCOLConjMultiplication(<complex input length>, <length vector>)
#F   inputs/output is in vector inteleaved complex format: 
#F     [[re1, re2, ..., re_v], [im1, im2, ..., im_v], [re_(v+1), ...], ...]

Class(VRCOLConjMultiplication, RewritableObject, BaseMat, OLBase, rec(
    dims := self >> [2 * self.rChildren()[2] * self.rChildren()[3], StripList(Replicate(self.rChildren()[1], 2 * self.rChildren()[2] * self.rChildren()[3]))],
));

