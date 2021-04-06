
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#######################################################################################
#   AltiVec 4-way 32-bit float instructions

# shuffle operations
Class(vbinop_4x32f_av, vbinop_av, rec(
    v := self >> 4,
    #ctype := self >> "float",
)); #Class added for AltiVec

Class(vunpacklo_4x32f_av, vbinop_4x32f_av, rec(
    altivec := "vec_mergeh",
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 4, 1)
)); #Class added for AltiVec

Class(vunpackhi_4x32f_av, vbinop_4x32f_av, rec(
    altivec := "vec_mergel",
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 4, 1)
)); #Class added for AltiVec


#   binary instructions
vunpacklo_4x32f.altivec := "vec_mergel";
vunpackhi_4x32f.altivec := "vec_mergeh";

Class(vperm_4x32f, vbinop_4x32f_av, rec(
    altivec := "vec_perm",
    semantic := (in1, in2, p) -> vpermop(in1, in2, p, 4),
    params := self >> sparams(4,8),
    permparams := aperm
));


#   unary instructions
Class(vuperm_4x32f, vunbinop_av, rec(
    binop := vperm_4x32f
));


#######################################################################################
#   AltiVec 8-way 16-bit integer instructions

#   binary instructions
vunpacklo_8x16i.altivec := "vec_mergel";
vunpackhi_8x16i.altivec := "vec_mergeh";

Class(vperm_8x16i, VecExp_8.binary(), rec(
    altivec := "vec_perm",
    semantic := (in1, in2, p) -> vpermop(in1, in2, p, 8),
    params := self >> sparams(8,16),
    permparams := aperm
));


#   unary instructions
Class(vuperm_8x16i, VecExp_8.unaryFromBinop(vperm_8x16i)); 


#######################################################################################
#   AltiVec 16-way 8-bit integer instructions

#   binary instructions
vunpacklo_16x8i.altivec := "vec_mergel";
vunpackhi_16x8i.altivec := "vec_mergeh";

Class(vperm_16x8i, VecExp_8.binary(), rec(
    altivec := "vec_perm",
    semantic := (in1, in2, p) -> vpermop(in1, in2, p, 16),
    params := self >> sparams(16,32),
    permparams := aperm
));


#   unary instructions
Class(vuperm_16x8i, VecExp_8.unaryFromBinop(vperm_16x8i)); 
