
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_neon_binop_ev := rec(
    ev := self >> self.t.value(self.semantic(self._vval(1), self._vval(2), [])).ev(),
);

############### HALF NEON #############
Class(vunpacklo_half, _neon_binop_ev, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 2, 1)));

Class(vunpackhi_half, _neon_binop_ev, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 2, 1)));

Class(vrev_half, VecExp_2.unary(), rec(
    semantic := (in1, p) -> vrev64(in1,2),
));

Class(vtrnq_half, _neon_binop_ev, VecExp_2.binary(), rec(
    semantic := (in1, in2, p) -> [vtransposelo(in1, in2, 2), vtransposehi(in1, in2, 2)],
    computeType := self >> TVect(TVect(T_Real(32), 2), 2),
));

Class(vextract_half, VecExp_2.unary(), rec(
    computeType := self >> TVect(T_Real(32), 2),
    semantic := (in1, p) -> Checked(Length(in1)=2, in1[p[1]+1]),
    ev := self >> self.t.value(self.semantic(self._vval(1), self.args[2].p)).ev(),
));

Class(vmulcx_half, VecExp_2.binary());
Class(vswapcx_half, VecExp_2.unary());

Class(vload1_half,  VecExp_2.binary());
Class(vstore1_half, VecExpCommand);


# vdup_lane_half(<float32x2>, <int lane_num>) ==> float32x2
Class(vdup_lane_half, VecExp_2.binary(), rec(
    ev := self >> let(lane := self.args[2].ev(),
	Checked(IsInt(lane), lane >=0, lane <= 1, 
	        self.t.value( Replicate(2, self.args[1].ev()[1+lane]) )))
));

# vext_half(<float32x2>, <float32x2>, 1) ==> float32x2
# vext_half([a, b], [c, d], 1) = [b c]
Class(vext_half, VecExp_2.ternary(), rec(
    ev := self >> let(num := self.args[3].ev(),
	Checked(IsInt(num), num=1,
	        self.t.value( self.args[1].ev()[2], self.args[2].ev()[1])))
));

#############################################################
# Composite instructions. Serve as a bridge between NEON and
# TL ISA DB. Later rewritten into combination of vuzpq_32f, 
# vzipq_32f and vtrnq_32f with vextract_neon_4x32f.
#

Class(vpacklo_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [1, 3, 1, 3], 4, 1)));

Class(vpackhi_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [2, 4, 2, 4], 4, 1)));

Class(vunpacklo_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 4, 1)));

Class(vunpackhi_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 4, 1)));

Class(vtransposelo_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> vtransposelo(in1, in2, 4)));

Class(vtransposehi_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> vtransposehi(in1, in2, 4)));

Class(vunpacklolo2_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpacklo(in1, in2, 4, 2)));

Class(vunpacklohi2_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [1,2,3,4], 4, 1)));

Class(vunpackhilo2_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> shuffle(in1, in2, [3,4,1,2], 4, 1)));

Class(vunpackhihi2_neon, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> unpackhi(in1, in2, 4, 2)));

Class(vrev_neon, VecExp_4.unary(), rec(
		semantic := (in1, p) -> vrev64(in1,4),
));
#############################################################

Class(vuzpq_32f, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> [shuffle(in1, in2, [1, 3, 1, 3], 4, 1), shuffle(in1, in2, [2, 4, 2, 4], 4, 1)],
    computeType := self >> TVect(TVect(T_Real(32), 4), 2),
));

Class(vzipq_32f, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> [unpacklo(in1, in2, 4, 1), unpackhi(in1, in2, 4, 1)],
    computeType := self >> TVect(TVect(T_Real(32), 4), 2),
));

Class(vtrnq_32f, _neon_binop_ev, VecExp_4.binary(), rec(
    semantic := (in1, in2, p) -> [vtransposelo(in1, in2, 4), vtransposehi(in1, in2, 4)],
    computeType := self >> TVect(TVect(T_Real(32), 4), 2),
));

Class(vextract_neon_4x32f, VecExp_4.unary(), rec(
    computeType := self >> TVect(T_Real(32), 4),
    semantic := (in1, p) -> Checked(Length(in1)=2, in1[p[1]+1]),
    ev := self >> self.t.value(self.semantic(self._vval(1), self.args[2].p)).ev(),
));


Class(vmulcx_neon, VecExp_4.binary());
Class(vswapcx_neon, VecExp_4.unary());


# vload1_neon(<float*>, <float32x4>, <int lane_num>) => float32x4
Class(vload1_neon,  VecExp_4.binary());

# vload_half_neon(<float*>) => float32x2
Class(vload_half_neon, VecExp_2.unary());

# vstore1_neon(<float*>, <float32x4>, <int lane_num>) 
Class(vstore1_neon, VecStoreCommand.binary());

# vstore2lo_neon(<float*>, <float32x4>)
Class(vstore2lo_neon, VecStoreCommand.binary());

# vstore2hi_neon(<float*>, <float32x4>)
Class(vstore2hi_neon, VecStoreCommand.binary());

# vcombine_neon(<float32x2>, <float32x>) => float32x4
Class(vcombine_neon, VecExp_4.binary());

# vdup_lane_neon(<float32x4>, <int lane_num>) ==> float32x4
Class(vdup_lane_neon, VecExp_4.binary(), rec(
    ev := self >> let(lane := self.args[2].ev(),
	Checked(IsInt(lane), lane >=0, lane <= 3, 
	        self.t.value( Replicate(4, self.args[1].ev()[1+lane]) )))
));

# vext_neon(<float32x4>, <float32x4>, 1) ==> float32x4
# vext_neon([a, b, c, d], [e, f, g, h], 1) = [d e f g]
# vext_neon([a, b, c, d], [e, f, g, h], 2) = [c d e f]
Class(vext_neon, VecExp_4.ternary(), rec(
    ev := self >> let(num := self.args[3].ev(),
	Checked(IsInt(num), num>=0, num<=3,
	        self.t.value( self.args[1].ev(){[5-num .. 4]} :: self.args[2].ev(){[1..4-num]})))
));
