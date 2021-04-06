
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#---------------------------------------------------------------------------------------
#   SPU vector instructions
#---------------------------------------------------------------------------------------

Class(vbinop_8x16i_spu, vbinop_spu, rec( v := self >> 8));
Class(vbinop_4x32f_spu, vbinop_spu, rec( v := self >> 4));
Class(vbinop_2x64f_spu, vbinop_spu, rec( v := self >> 2));

Class(exp_8x16i, rec(v := 8, computeType := self >> TVect(TInt,    8))); 
Class(exp_4x32f, rec(v := 4, computeType := self >> TVect(TDouble, 4))); 
Class(exp_2x64f, rec(v := 2, computeType := self >> TVect(TDouble, 2))); 

Class(cmd_8x16i, rec(v := 8, computeType := self >> TVect(TInt,    8))); 
Class(cmd_4x32f, rec(v := 4, computeType := self >> TVect(TDouble, 4))); 
Class(cmd_2x64f, rec(v := 2, computeType := self >> TVect(TDouble, 2))); 

# Load -----------------------------
Class(vloadu8_spu8x16i, vloadop_new, exp_8x16i, rec(numargs := 1));
Class(vloadu4_spu4x32f, vloadop_new, exp_4x32f, rec(numargs := 1));

# Store ----------------------------

# Zero -----------------------------
Class(vzero_8x16i, vop_new, exp_8x16i, rec(numargs := 0));
Class(vzero_4x32f, vop_new, exp_4x32f, rec(numargs := 0));
Class(vzero_2x64f, vop_new, exp_2x64f, rec(numargs := 0));

# Subvec ---------------------------
Class(promote_spu8x16i, vbinop_new, exp_8x16i);
Class(promote_spu4x32f, vbinop_new, exp_4x32f);
Class(promote_spu2x64f, vbinop_new, exp_2x64f);

Class(extract_spu8x16i, vbinop_new, exp_8x16i, rec(computeType := self >> TInt));
Class(extract_spu4x32f, vbinop_new, exp_4x32f, rec(computeType := self >> TReal));
Class(extract_spu2x64f, vbinop_new, exp_2x64f, rec(computeType := self >> TReal));

Class(insert_spu8x16i, vop_new, exp_8x16i, rec(numargs := 3));
Class(insert_spu4x32f, vop_new, exp_4x32f, rec(numargs := 3));
Class(insert_spu2x64f, vop_new, exp_2x64f, rec(numargs := 3));

Class(vsplat_8x16i, vloadop_new, exp_8x16i, rec(numargs := 1));
Class(vsplat_4x32f, vloadop_new, exp_4x32f, rec(numargs := 1));
Class(vsplat_2x64f, vloadop_new, exp_2x64f, rec(numargs := 1));

# Binary ---------------------------
# VA: This breaks bin_or, looks like something unfinished so commented out.
# Class(bin_or, vbinop_new, exp_8x16i);
# Class(bin_or, vbinop_new, exp_4x32f);
# Class(bin_or, vbinop_new, exp_2x64f);

# Rotates ---------------------------
Class(slqwbyte_spu4x32f, vop_new, exp_4x32f, rec(numargs := 1));

Class(rlmaskqwbyte_spu4x32f, vop_new, exp_4x32f, rec(numargs := 1));

# Binary shuffle -------------------
#NOTE: What are sparams, params, semantic, and permparams?
#NOTE: Can we combine all these perms into the same thing somehow?
Class(vperm_8x16i_spu, vbinop_8x16i_spu, rec(
    semantic := (in1, in2, p) -> vpermop(in1, in2, p, 8),
    params := self >> sparams(8, 16),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2], rch[3].p]),
    permparams := aperm
));

Class(vperm_4x32f_spu, vbinop_4x32f_spu, rec(
    semantic := (in1, in2, p) -> vpermop(in1, in2, p, 4),
    params := self >> sparams(4, 8),

    #HACK: small hack: it'd be nice to not define from_rChildren explicitly
    #here. We have to do it though, because the last param is a perm
    #(vparam_spu) type, but the object must be created with a List, and not a
    #vparam_spu.

    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2], rch[3].p]),
    permparams := aperm
));

Class(vperm_2x64f_spu, vbinop_2x64f_spu, rec(
    semantic := (in1, in2, p) -> vpermop(in1, in2, p, 2),
    params := self >> sparams(2, 4),
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2], rch[3].p]),
    permparams := aperm
));

# Unary shuffle --------------------
Class(vuperm_8x16i_spu, vunbinop_spu, 
    rec(binop := vperm_8x16i_spu,
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2].p])
));

Class(vuperm_4x32f_spu, vunbinop_spu, 
    rec(binop := vperm_8x16i_spu,
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2].p])
));

Class(vuperm_2x64f_spu, vunbinop_spu, 
    rec(binop := vperm_8x16i_spu,
    from_rChildren := (self, rch) >> ApplyFunc(ObjId(self), [rch[1], rch[2].p])
));
