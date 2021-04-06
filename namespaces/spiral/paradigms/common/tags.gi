
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# tag to force GT_NthLoop expansions to be smaller, results in less
# ruletrees being generated, which is sometimes desirable.
#
# NOTE: it is used in the GT_NthLoop rule at currently needs no
#       parameters

Class(ALimitNthLoop, AGenericTag);

# Input/Output tag which describes the input/outputs of the given block. 
# Designed with OL in mind, although written for transforms.
#
# You initialize the tag with a list of pairs which specify which input
# and outputs are connected. 
#
# In the case of transforms, this means AIO([1,1]) is the only acceptable 
# input. And it means the block is done inplace. Normal operation is 
# out-of-place.
#
# for OL, this can be AIO([1,3],[2,1]), which means input1 -> output3 and
# input2 -> output1
#
Class(AIO, AGenericTag);


