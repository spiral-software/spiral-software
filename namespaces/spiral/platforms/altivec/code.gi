
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#---------------------------------------------------------------------------------------
#   altivec vector instructions
#---------------------------------------------------------------------------------------

#Class(vbinop_4x32f_av, vbinop_av, rec( v := self >> 4));

#Class(exp_4x32f, rec(v := 4, computeType := self >> TVect(TDouble, 4))); 

#Class(cmd_4x32f, rec(v := 4, computeType := self >> TVect(TDouble, 4))); 

# Load -----------------------------

# Store ----------------------------

# Zero -----------------------------
# use data_type.zero() instead of special class
# Class(vzero_4x32f, vop_av, exp_4x32f, rec(numargs := 0));

# Subvec ---------------------------

# Binary ---------------------------
# Class(bin_or, vbinop_new, exp_4x32f);

# Rotates ---------------------------

# Binary shuffle -------------------

#Class(vperm_4x32f_spu, vbinop_4x32f_spu, rec(
#    semantic := (in1, in2, p) -> vpermop(in1, in2, p, 4),
#    params := self >> sparams(4, 8),
#    permparams := aperm
#));

# Unary shuffle --------------------
#Class(vuperm_4x32f_spu, vunbinop_spu, rec(binop := vperm_4x32f_spu));

