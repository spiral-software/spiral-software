
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(VStack);

# ==========================================================================
# _HStack(<spl1>, ...)
#    equivalent of HStack, expressed using Tensor/Mat/DirectSum
#    use for unrolled code, to avoid problems with array scalarization
#
_HStack := arg -> Tensor(Mat([Replicate(Length(arg), 1)]), I(Rows(arg[1]))) * DirectSum(arg);

# ==========================================================================
# _SUM(<spl1>, ...)
#    overlapping dense spl1+spl2+..., expressed using Tensor/Mat/VStack
#    use for unrolled code, to avoid problems with array scalarization
#
_SUM    := arg -> Tensor(Mat([Replicate(Length(arg), 1)]), I(Rows(arg[1]))) * VStack(arg);


# ==========================================================================
# HStack(<spls>) - horizontal stacking operator (row of blocks)
#
# Example: MatSPL( HStack(F(2), I(2))) );
#     [ [ 1, 1, 1, 0 ],
#       [ 1, -1, 0, 1 ] ]
# ==========================================================================
Class(HStack, BaseOperation, rec(
    abbrevs := [ arg -> [Flat(arg)] ],
    new := (self, spls) >> Checked(
		IsList(spls), Length(spls) >= 1, 
		SPL(WithBases(self, rec(_children := spls)).setDims())
    ),

    dims := self >> [Maximum(List(self.children(), c->c.dims()[1])), Sum(self._children, x->x.dims()[2])], 

    isPermutation := self >> false,
    transpose := self >> CopyFields(self, VStack(List(self._children, x->x.transpose()))),
    conjTranspose := self >> CopyFields(self, VStack(List(self._children, x->x.conjTranspose()))),

    toAMat := self >> ApplyFunc(_HStack, self.children()).toAMat()
));

# this is a HStack that is guaranteed not to perform any ops, and doesnt need ScatAcc in SumsGen
Class(HStack1, HStack);


# ==========================================================================
# VStack(<spls>) - vertical stacking operator (column of blocks)
#
# Example: MatSPL( VStack(F(2), I(2))) );
#    [ [ 1, 1 ],
#      [ 1, -1 ],
#      [ 1, 0 ],
#      [ 0, 1 ] ]
# ==========================================================================
Class(VStack, BaseOperation, rec(
    abbrevs := [ arg -> [Flat(arg)] ],

    new := (self, spls) >> Checked(
		IsList(spls), Length(spls) >= 1, 
		SPL(WithBases(self, rec(_children := spls)).setDims())
    ),

    dims := self >> [Sum(self._children, x->x.dims()[1]), Maximum(List(self.children(), c->c.dims()[2]))], 

    isPermutation := self >> false,
    transpose := self >> CopyFields(self, HStack(List(self._children, x->x.transpose()))),
    conjTranspose := self >> CopyFields(self, HStack(List(self._children, x->x.conjTranspose()))),

    toAMat := self >> ApplyFunc(_HStack, List(self.children(), e -> e.transpose())).transpose().toAMat()
));

# ==========================================================================
# BlockMat(<matrix of spls>) - puts several spls in a bigger block matrix
#
# Example: MatSPL( BlockMat([[I(2), J(2)], [O(2,2), -1 * I(2)]]) )
#   [ [ 1,  0,  0,  1 ],
#     [ 0,  1,  1,  0 ],
#     [ 0,  0, -1,  0 ],
#     [ 0,  0,  0, -1 ] ]
# ==========================================================================
BlockMat := arg -> let(mat := When(Length(arg)=1, arg[1], arg),
    VStack(List(mat, x->HStack(x))));

