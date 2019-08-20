
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details

HDLTest := function(which, type, bitwidth, twidwidth)
   local opts;
   opts := InitStreamHw();
   spiral.paradigms.stream._setHDLDataType(type, bitwidth);

   if ((type = "fix") and (twidwidth <> -1)) then
      spiral.paradigms.stream._setHDLTwidWidth(twidwidth);
   else
      spiral.profiler.default_profiles.fpga_splhdl.makeopts.TWIDTYPE := "";
   fi;
if ((which=0) or (which=-1)) then
   Print("--- test 000: bitwidth ", bitwidth, " Pease DFT16 r=2, w=2: ", HDLVerify(streamDFTPease(16,2,2,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=1) or (which=-1)) then
   Print("--- test 001: bitwidth ", bitwidth, " Pease DFT16 r=2, w=4: ", HDLVerify(streamDFTPease(16,2,4,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=2) or (which=-1)) then
   Print("--- test 002: bitwidth ", bitwidth, " Pease DFT16 r=2, w=8: ", HDLVerify(streamDFTPease(16,2,8,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=3) or (which=-1)) then
   Print("--- test 003: bitwidth ", bitwidth, " Pease DFT16 r=2, w=16: ", HDLVerify(streamDFTPease(16,2,16,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=4) or (which=-1)) then
   Print("--- test 004: bitwidth ", bitwidth, " Pease DFT16 r=2, w=32: ", HDLVerify(streamDFTPease(16,2,32,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=5) or (which=-1)) then
   Print("--- test 005: bitwidth ", bitwidth, " Pease DFT16 r=4, w=8: ", HDLVerify(streamDFTPease(16,4,8,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=6) or (which=-1)) then
   Print("--- test 006: bitwidth ", bitwidth, " Pease DFT16 r=4, w=16: ", HDLVerify(streamDFTPease(16,4,16,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=7) or (which=-1)) then
   Print("--- test 007: bitwidth ", bitwidth, " Pease DFT16 r=4, w=32: ", HDLVerify(streamDFTPease(16,4,32,1), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=8) or (which=-1)) then
   Print("--- test 008: bitwidth ", bitwidth, " Fully unfolded DFT16 ", HDLVerify(streamDFTNoFold(16,1), RC(DFT(16)), opts), "\n");
fi;
if ((which=9) or (which=-1)) then
   Print("--- test 009: bitwidth ", bitwidth, " Pease DFTDR16 r=2, w=4: ", HDLVerify(streamDFTDRPease(16,2,4), RC(DFTDR(16,1,2)), opts), "\n"); 
fi;
if ((which=10) or (which=-1)) then
   Print("--- test 010: bitwidth ", bitwidth, " Pease DFTDR16 r=2, w=8: ", HDLVerify(streamDFTDRPease(16,2,8), RC(DFTDR(16, 1,2)), opts), "\n"); 
fi;
if ((which=11) or (which=-1)) then
   Print("--- test 011: bitwidth ", bitwidth, " Pease DFTDR16 r=2, w=16: ", HDLVerify(streamDFTDRPease(16,2,16), RC(DFTDR(16, 1,2)), opts), "\n"); 
fi;
if ((which=12) or (which=-1)) then
   Print("--- test 012: bitwidth ", bitwidth, " Pease DFTDR16 r=2, w=32: ", HDLVerify(streamDFTDRPease(16,2,32), RC(DFTDR(16, 1,2)), opts), "\n");
fi;
if ((which=13) or (which=-1)) then
   Print("--- test 013: bitwidth ", bitwidth, " Pease DFTDR16 r=4, w=8: ", HDLVerify(streamDFTDRPease(16,4,8), RC(DFTDR(16, 1,4)), opts), "\n"); 
fi;
if ((which=14) or (which=-1)) then
   Print("--- test 014: bitwidth ", bitwidth, " Pease DFTDR16 r=4, w=16: ", HDLVerify(streamDFTDRPease(16,4,16), RC(DFTDR(16, 1,4)), opts), "\n"); 
fi;
if ((which=15) or (which=-1)) then
   Print("--- test 015: bitwidth ", bitwidth, " Pease DFTDR16 r=4, w=32: ", HDLVerify(streamDFTDRPease(16,4,32), RC(DFTDR(16, 1,4)), opts), "\n");
fi;
if ((which=16) or (which=-1)) then
   Print("--- test 016: bitwidth ", bitwidth, " Pease DRDFT16 r=2, w=4: ", HDLVerify(streamDRDFTPease(16,2,4), RC(DRDFT(16, 1, 2)), opts), "\n"); 
fi;
if ((which=17) or (which=-1)) then
   Print("--- test 017: bitwidth ", bitwidth, " Pease DRDFT16 r=2, w=8: ", HDLVerify(streamDRDFTPease(16,2,8), RC(DRDFT(16, 1, 2)), opts), "\n"); 
fi;
if ((which=18) or (which=-1)) then
   Print("--- test 018: bitwidth ", bitwidth, " Pease DRDFT16 r=2, w=16: ", HDLVerify(streamDRDFTPease(16,2,16), RC(DRDFT(16, 1, 2)), opts), "\n"); 
fi;
if ((which=19) or (which=-1)) then
   Print("--- test 019: bitwidth ", bitwidth, " Pease DRDFT16 r=2, w=32: ", HDLVerify(streamDRDFTPease(16,2,32), RC(DRDFT(16, 1, 2)), opts), "\n"); 
fi;
if ((which=20) or (which=-1)) then
   Print("--- test 020: bitwidth ", bitwidth, " Pease DRDFT16 r=4, w=8: ", HDLVerify(streamDRDFTPease(16,4,8), RC(DRDFT(16, 1, 4)), opts), "\n"); 
fi;
if ((which=21) or (which=-1)) then
   Print("--- test 021: bitwidth ", bitwidth, " Pease DRDFT16 r=4, w=16: ", HDLVerify(streamDRDFTPease(16,4,16), RC(DRDFT(16, 1, 4)), opts), "\n"); 
fi;
if ((which=22) or (which=-1)) then
   Print("--- test 022: bitwidth ", bitwidth, " Pease DRDFT16 r=4, w=32: ", HDLVerify(streamDRDFTPease(16,4,32), RC(DRDFT(16, 1, 4)), opts), "\n"); 
fi;
if ((which=23) or (which=-1)) then
   Print("--- test 023: bitwidth ", bitwidth, " Pease DFT64 r=2, w=2: ", HDLVerify(streamDFTPease(64,2,2,1), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=24) or (which=-1)) then
   Print("--- test 024: bitwidth ", bitwidth, " Pease DFT64 r=2, w=4: ", HDLVerify(streamDFTPease(64,2,4,1), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=25) or (which=-1)) then
   Print("--- test 025: bitwidth ", bitwidth, " Pease DFT64 r=2, w=16: ", HDLVerify(streamDFTPease(64,2,16,1), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=26) or (which=-1)) then
   Print("--- test 026: bitwidth ", bitwidth, " Pease DFT64 r=8, w=16: ", HDLVerify(streamDFTPease(64,8,16,1), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=27) or (which=-1)) then
   Print("--- test 027: bitwidth ", bitwidth, " Pease DFTDR64 r=2, w=4: ", HDLVerify(streamDFTDRPease(64,2,4), RC(DFTDR(64, 1,2)), opts), "\n"); 
fi;
if ((which=28) or (which=-1)) then
   Print("--- test 028: bitwidth ", bitwidth, " Pease DFTDR64 r=2, w=16: ", HDLVerify(streamDFTDRPease(64,2,16), RC(DFTDR(64, 1,2)), opts), "\n"); 
fi;
if ((which=29) or (which=-1)) then
   Print("--- test 029: bitwidth ", bitwidth, " Pease DFTDR64 r=8, w=16: ", HDLVerify(streamDFTDRPease(64,8,16), RC(DFTDR(64, 1,8)), opts), "\n"); 
fi;
if ((which=30) or (which=-1)) then
   Print("--- test 030: bitwidth ", bitwidth, " Pease DRDFT64 r=2, w=4: ", HDLVerify(streamDRDFTPease(64,2,4), RC(DRDFT(64, 1, 2)), opts), "\n"); 
fi;
if ((which=31) or (which=-1)) then
   Print("--- test 031: bitwidth ", bitwidth, " Pease DRDFT64 r=2, w=16: ", HDLVerify(streamDRDFTPease(64,2,16), RC(DRDFT(64, 1, 2)), opts), "\n"); 
fi;
if ((which=32) or (which=-1)) then
   Print("--- test 032: bitwidth ", bitwidth, " Pease DRDFT64 r=8, w=16: ", HDLVerify(streamDRDFTPease(64,8,16), RC(DRDFT(64, 1, 8)), opts), "\n"); 
fi;
if ((which=33) or (which=-1)) then
   Print("--- test 033: bitwidth ", bitwidth, " Pease DFT16 r=2, w=4, unrolling=2: ", HDLVerify(streamDFTPease(16,2,4,2), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=34) or (which=-1)) then
   Print("--- test 034: bitwidth ", bitwidth, " Pease DFT16 r=2, w=8, unrolling=2: ", HDLVerify(streamDFTPease(16,2,8,2), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=35) or (which=-1)) then
   Print("--- test 035: bitwidth ", bitwidth, " Pease DFT64 r=2, w=4, unrolling=2: ", HDLVerify(streamDFTPease(64,2,4,2), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=36) or (which=-1)) then
   Print("--- test 036: bitwidth ", bitwidth, " Pease DFT64 r=2, w=4, unrolling=3: ", HDLVerify(streamDFTPease(64,2,4,3), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=37) or (which=-1)) then
   Print("--- test 037: bitwidth ", bitwidth, " Pease DFT64 r=8, w=16, unrolling=2: ", HDLVerify(streamDFTPease(64,8,16,2), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=38) or (which=-1)) then
   Print("--- test 038: bitwidth ", bitwidth, " Streaming DFT16 r=2, w=4: ", HDLVerify(streamDFTUnroll(16,2,4), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=39) or (which=-1)) then
   Print("--- test 039: bitwidth ", bitwidth, " Streaming DFT16 r=2, w=8: ", HDLVerify(streamDFTUnroll(16,2,8), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=40) or (which=-1)) then
   Print("--- test 040: bitwidth ", bitwidth, " Streaming DFT16 r=2, w=16: ", HDLVerify(streamDFTUnroll(16,2,16), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=41) or (which=-1)) then
   Print("--- test 041: bitwidth ", bitwidth, " Streaming DFT16 r=2, w=32: ", HDLVerify(streamDFTUnroll(16,2,32), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=42) or (which=-1)) then
   Print("--- test 042: bitwidth ", bitwidth, " Streaming DFT16 r=4, w=8: ", HDLVerify(streamDFTUnroll(16,4,8), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=43) or (which=-1)) then
   Print("--- test 043: bitwidth ", bitwidth, " Streaming DFT16 r=4, w=16: ", HDLVerify(streamDFTUnroll(16,4,16), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=44) or (which=-1)) then
   Print("--- test 044: bitwidth ", bitwidth, " Streaming DFT16 r=4, w=32: ", HDLVerify(streamDFTUnroll(16,4,32), RC(DFT(16)), opts), "\n"); 
fi;
if ((which=45) or (which=-1)) then
   Print("--- test 045: bitwidth ", bitwidth, " Streaming DFTDR16 r=2, w=4: ", HDLVerify(streamDFTDRUnroll(16,2,4), RC(DFTDR(16, 1,2)), opts), "\n"); 
fi;
if ((which=46) or (which=-1)) then
   Print("--- test 046: bitwidth ", bitwidth, " Streaming DFTDR16 r=2, w=8: ", HDLVerify(streamDFTDRUnroll(16,2,8), RC(DFTDR(16, 1,2)), opts), "\n"); 
fi;
if ((which=47) or (which=-1)) then
   Print("--- test 047: bitwidth ", bitwidth, " Streaming DFTDR16 r=2, w=16: ", HDLVerify(streamDFTDRUnroll(16,2,16), RC(DFTDR(16, 1,2)), opts), "\n"); 
fi;
if ((which=48) or (which=-1)) then
   Print("--- test 048: bitwidth ", bitwidth, " Streaming DFTDR16 r=2, w=32: ", HDLVerify(streamDFTDRUnroll(16,2,32), RC(DFTDR(16, 1,2)), opts), "\n");  
fi;
if ((which=49) or (which=-1)) then
   Print("--- test 049: bitwidth ", bitwidth, " Streaming DFTDR16 r=4, w=8: ", HDLVerify(streamDFTDRUnroll(16,4,8), RC(DFTDR(16, 1,4)), opts), "\n"); 
fi;
if ((which=50) or (which=-1)) then
   Print("--- test 050: bitwidth ", bitwidth, " Streaming DFTDR16 r=4, w=16: ", HDLVerify(streamDFTDRUnroll(16,4,16), RC(DFTDR(16, 1,4)), opts), "\n"); 
fi;
if ((which=51) or (which=-1)) then
   Print("--- test 051: bitwidth ", bitwidth, " Streaming DRDFT16 r=2, w=4: ", HDLVerify(streamDRDFTUnroll(16,2,4), RC(DRDFT(16, 1, 2)), opts), "\n"); 
fi;
if ((which=52) or (which=-1)) then
   Print("--- test 052: bitwidth ", bitwidth, " Streaming DRDFT16 r=2, w=8: ", HDLVerify(streamDRDFTUnroll(16,2,8), RC(DRDFT(16, 1, 2)), opts), "\n"); 
fi;
if ((which=53) or (which=-1)) then
   Print("--- test 053: bitwidth ", bitwidth, " Streaming DRDFT16 r=2, w=16: ", HDLVerify(streamDRDFTUnroll(16,2,16), RC(DRDFT(16, 1, 2)), opts), "\n");  
fi;
if ((which=54) or (which=-1)) then
   Print("--- test 054: bitwidth ", bitwidth, " Streaming DRDFT16 r=2, w=32: ", HDLVerify(streamDRDFTUnroll(16,2,32), RC(DRDFT(16, 1, 2)), opts), "\n");  
fi;
if ((which=55) or (which=-1)) then
   Print("--- test 055: bitwidth ", bitwidth, " Streaming DRDFT16 r=4, w=8: ", HDLVerify(streamDRDFTUnroll(16,4,8), RC(DRDFT(16, 1, 4)), opts), "\n"); 
fi;
if ((which=56) or (which=-1)) then
   Print("--- test 056: bitwidth ", bitwidth, " Streaming DRDFT16 r=4, w=16: ", HDLVerify(streamDRDFTUnroll(16,4,16), RC(DRDFT(16, 1, 4)), opts), "\n"); 
fi;
if ((which=57) or (which=-1)) then
   Print("--- test 057: bitwidth ", bitwidth, " Streaming DFT64 r=2, w=4: ", HDLVerify(streamDFTUnroll(64,2,4), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=58) or (which=-1)) then
   Print("--- test 058: bitwidth ", bitwidth, " Streaming DFT64 r=2, w=16: ", HDLVerify(streamDFTUnroll(64,2,16), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=59) or (which=-1)) then
   Print("--- test 059: bitwidth ", bitwidth, " Streaming DFT64 r=8, w=16: ", HDLVerify(streamDFTUnroll(64,8,16), RC(DFT(64)), opts), "\n"); 
fi;
if ((which=60) or (which=-1)) then
   Print("--- test 060: bitwidth ", bitwidth, " Streaming DFTDR64 r=2, w=4: ", HDLVerify(streamDFTDRUnroll(64,2,4), RC(DFTDR(64, 1,2)), opts), "\n"); 
fi;
if ((which=61) or (which=-1)) then
   Print("--- test 061: bitwidth ", bitwidth, " Streaming DFTDR64 r=2, w=16: ", HDLVerify(streamDFTDRUnroll(64,2,16), RC(DFTDR(64, 1,2)), opts), "\n"); 
fi;
if ((which=62) or (which=-1)) then
   Print("--- test 062: bitwidth ", bitwidth, " Streaming DFTDR64 r=8, w=16: ", HDLVerify(streamDFTDRUnroll(64,8,16), RC(DFTDR(64, 1,8)), opts), "\n"); 
fi;
if ((which=63) or (which=-1)) then
   Print("--- test 063: bitwidth ", bitwidth, " Streaming DRDFT64 r=2, w=4: ", HDLVerify(streamDRDFTUnroll(64,2,4), RC(DRDFT(64, 1, 2)), opts), "\n"); 
fi;
if ((which=64) or (which=-1)) then
   Print("--- test 064: bitwidth ", bitwidth, " Streaming DRDFT64 r=2, w=16: ", HDLVerify(streamDRDFTUnroll(64,2,16), RC(DRDFT(64, 1, 2)), opts), "\n"); 
fi;
if ((which=65) or (which=-1)) then
   Print("--- test 065: bitwidth ", bitwidth, " Streaming DRDFT64 r=8, w=16: ", HDLVerify(streamDRDFTUnroll(64,8,16), RC(DRDFT(64, 1, 8)), opts), "\n"); 
fi;
if ((which=66) or (which=-1)) then
   Print("--- test 066: bitwidth ", bitwidth, " Pease IDFT16 r=2, w=4: ", HDLVerify(streamIDFTPease(16,2,4,1), RC(DFT(16,-1)), opts), "\n"); 
fi;
if ((which=67) or (which=-1)) then
   Print("--- test 067: bitwidth ", bitwidth, " Pease IDFT16 r=4, w=16: ", HDLVerify(streamIDFTPease(16,4,16,1), RC(DFT(16,-1)), opts), "\n"); 
fi;
if ((which=68) or (which=-1)) then
   Print("--- test 068: bitwidth ", bitwidth, " Pease IDFTDR16 r=2, w=8: ", HDLVerify(streamIDFTDRPease(16,2,8), RC(DFTDR(16, -1,2)), opts), "\n");  
fi;
if ((which=69) or (which=-1)) then
   Print("--- test 069: bitwidth ", bitwidth, " Pease DRIDFT16 r=4, w=32: ", HDLVerify(streamDRIDFTPease(16,4,32), RC(DRDFT(16, -1, 4)), opts), "\n"); 
fi;
if ((which=70) or (which=-1)) then
   Print("--- test 070: bitwidth ", bitwidth, " Pease IDFT64 r=8, w=16: ", HDLVerify(streamIDFTPease(64,8,16,1), RC(DFT(64,-1)), opts), "\n"); 
fi;
if ((which=71) or (which=-1)) then
   Print("--- test 071: bitwidth ", bitwidth, " Streaming IDFT16 r=2, w=4: ", HDLVerify(streamIDFTUnroll(16,2,4), RC(DFT(16,-1)), opts), "\n"); 
fi;
if ((which=72) or (which=-1)) then
   Print("--- test 072: bitwidth ", bitwidth, " Streaming IDFT16 r=4, w=16: ", HDLVerify(streamIDFTUnroll(16,4,16), RC(DFT(16,-1)), opts), "\n"); 
fi;
if ((which=73) or (which=-1)) then
   Print("--- test 073: bitwidth ", bitwidth, " Streaming IDFTDR16 r=2, w=8: ", HDLVerify(streamIDFTDRUnroll(16,2,8), RC(DFTDR(16, -1,2)), opts), "\n"); 
fi;
if ((which=74) or (which=-1)) then
   Print("--- test 074: bitwidth ", bitwidth, " Streaming DRIDFT16 r=4, w=16: ", HDLVerify(streamDRIDFTUnroll(16,4,16), RC(DRDFT(16, -1, 4)), opts), "\n"); 
fi;
if ((which=75) or (which=-1)) then
   Print("--- test 075: bitwidth ", bitwidth, " Streaming IDFT64 r=8, w=16: ", HDLVerify(streamIDFTUnroll(64,8,16), RC(DFT(64,-1)), opts), "\n"); 
fi;
if ((which=76) or (which=-1)) then
   Print("--- test 076: bitwidth ", bitwidth, " 2D DFT(8,2,4,1,1): ", HDLVerify(stream2DDFT(8,2,4,1,1), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=77) or (which=-1)) then
   Print("--- test 077: bitwidth ", bitwidth, " 2D DFT(8,2,8,1,1): ", HDLVerify(stream2DDFT(8,2,8,1,1), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=78) or (which=-1)) then
   Print("--- test 078: bitwidth ", bitwidth, " 2D DFT(8,2,4,1,3): ", HDLVerify(stream2DDFT(8,2,4,1,3), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=79) or (which=-1)) then
   Print("--- test 079: bitwidth ", bitwidth, " 2D DFT(8,2,8,1,3): ", HDLVerify(stream2DDFT(8,2,8,1,3), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=80) or (which=-1)) then
   Print("--- test 080: bitwidth ", bitwidth, " 2D DFT(8,2,4,2,1): ", HDLVerify(stream2DDFT(8,2,4,2,1), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=81) or (which=-1)) then
   Print("--- test 081: bitwidth ", bitwidth, " 2D DFT(8,2,8,2,1): ", HDLVerify(stream2DDFT(8,2,8,2,1), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=82) or (which=-1)) then
   Print("--- test 082: bitwidth ", bitwidth, " 2D DFT(8,2,4,2,3): ", HDLVerify(stream2DDFT(8,2,4,2,3), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=83) or (which=-1)) then
   Print("--- test 083: bitwidth ", bitwidth, " 2D DFT(8,2,8,2,3): ", HDLVerify(stream2DDFT(8,2,8,2,3), RC(MDDFT([8,8])), opts), "\n");
fi;
if ((which=84) or (which=-1)) then
   Print("--- test 084: bitwidth ", bitwidth, " Streaming DFT32 r=4, w=8: ", HDLVerify(streamDFTUnroll(32,4,8), RC(DFT(32)), opts), "\n");
fi;
if ((which=85) or (which=-1)) then
   Print("--- test 085: bitwidth ", bitwidth, " Streaming DFT32 r=4, w=16: ", HDLVerify(streamDFTUnroll(32,4,16), RC(DFT(32)), opts), "\n");
fi;
if ((which=86) or (which=-1)) then
   Print("--- test 086: bitwidth ", bitwidth, " Streaming DFT32 r=4, w=32: ", HDLVerify(streamDFTUnroll(32,4,32), RC(DFT(32)), opts), "\n");
fi;
if ((which=87) or (which=-1)) then
   Print("--- test 087: bitwidth ", bitwidth, " Streaming DFT32 r=8, w=16: ", HDLVerify(streamDFTUnroll(32,8,16), RC(DFT(32)), opts), "\n");
fi;
if ((which=88) or (which=-1)) then
   Print("--- test 088: bitwidth ", bitwidth, " Streaming DFT32 r=8, w=32: ", HDLVerify(streamDFTUnroll(32,8,32), RC(DFT(32)), opts), "\n");
fi;
if ((which=89) or (which=-1)) then
   Print("--- test 089: bitwidth ", bitwidth, " Streaming DFT256 r=8, w=16: ", HDLVerify(streamDFTUnroll(256,8,16), RC(DFT(256)), opts), "\n");
fi;
if ((which=90) or (which=-1)) then
   Print("--- test 090: bitwidth ", bitwidth, " Streaming DFT256 r=8, w=32: ",HDLVerify(streamDFTUnroll(256,8,32), RC(DFT(256)), opts), "\n");
fi;
if ((which=91) or (which=-1)) then
   Print("--- test 091: bitwidth ", bitwidth, " Pease RDFT(32, 4, 4, 1): ", HDLVerify(streamRDFT(32,4,4,1,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=92) or (which=-1)) then
   Print("--- test 092: bitwidth ", bitwidth, " Pease RDFT(32, 4, 4, 2): ", HDLVerify(streamRDFT(32,4,4,2,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=93) or (which=-1)) then
   Print("--- test 093: bitwidth ", bitwidth, " Strm. RDFT(32, 4, 4, 4): ", HDLVerify(streamRDFT(32,4,4,4,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=94) or (which=-1)) then
   Print("--- test 094: bitwidth ", bitwidth, " Pease RDFT(32, 4, 8, 1): ", HDLVerify(streamRDFT(32,4,8,1,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=95) or (which=-1)) then
   Print("--- test 095: bitwidth ", bitwidth, " Pease RDFT(32, 4, 8, 2): ", HDLVerify(streamRDFT(32,4,8,2,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=96) or (which=-1)) then
   Print("--- test 096: bitwidth ", bitwidth, " Strm. RDFT(32, 4, 8, 4): ", HDLVerify(streamRDFT(32,4,8,4,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=97) or (which=-1)) then
   Print("--- test 097: bitwidth ", bitwidth, " Pease RDFT(32, 4, 16, 1): ", HDLVerify(streamRDFT(32,4,16,1,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=98) or (which=-1)) then
   Print("--- test 098: bitwidth ", bitwidth, " Pease RDFT(32, 4, 16, 2): ", HDLVerify(streamRDFT(32,4,16,2,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=99) or (which=-1)) then
   Print("--- test 099: bitwidth ", bitwidth, " Strm. RDFT(32, 4, 16, 4): ", HDLVerify(streamRDFT(32,4,16,4,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=100) or (which=-1)) then
   Print("--- test 100: bitwidth ", bitwidth, " Pease RDFT(32, 8, 8, 1): ", HDLVerify(streamRDFT(32,8,8,1,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=101) or (which=-1)) then
   Print("--- test 101: bitwidth ", bitwidth, " Strm. RDFT(32, 8, 8, 2): ", HDLVerify(streamRDFT(32,8,8,2,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=102) or (which=-1)) then
   Print("--- test 102: bitwidth ", bitwidth, " Pease RDFT(32, 8, 16, 1): ", HDLVerify(streamRDFT(32,8,16,1,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=103) or (which=-1)) then
   Print("--- test 103: bitwidth ", bitwidth, " Strm. RDFT(32, 8, 16, 2): ", HDLVerify(streamRDFT(32,8,16,2,1), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=104) or (which=-1)) then
   Print("--- test 104: bitwidth ", bitwidth, " Pease RDFT(128, 4, 4, 1): ", HDLVerify(streamRDFT(128, 4, 4, 1, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=105) or (which=-1)) then
   Print("--- test 105: bitwidth ", bitwidth, " Pease RDFT(128, 4, 4, 3): ", HDLVerify(streamRDFT(128, 4, 4, 3, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=106) or (which=-1)) then
   Print("--- test 106: bitwidth ", bitwidth, " Strm. RDFT(128, 4, 4, 6): ", HDLVerify(streamRDFT(128, 4, 4, 6, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=107) or (which=-1)) then
   Print("--- test 107: bitwidth ", bitwidth, " Pease RDFT(128, 4, 8, 1): ", HDLVerify(streamRDFT(128, 4, 8, 1, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=108) or (which=-1)) then
   Print("--- test 108: bitwidth ", bitwidth, " Pease RDFT(128, 4, 8, 3): ", HDLVerify(streamRDFT(128, 4, 8, 3, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=109) or (which=-1)) then
   Print("--- test 109: bitwidth ", bitwidth, " Strm. RDFT(128, 4, 8, 6): ", HDLVerify(streamRDFT(128, 4, 8, 6, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=110) or (which=-1)) then
   Print("--- test 110: bitwidth ", bitwidth, " Pease RDFT(128, 4, 16, 1): ", HDLVerify(streamRDFT(128, 4, 16, 1, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=111) or (which=-1)) then
   Print("--- test 111: bitwidth ", bitwidth, " Pease RDFT(128, 4, 16, 3): ", HDLVerify(streamRDFT(128, 4, 16, 3, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=112) or (which=-1)) then
   Print("--- test 112: bitwidth ", bitwidth, " Strm. RDFT(128, 4, 16, 6): ", HDLVerify(streamRDFT(128, 4, 16, 6, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=113) or (which=-1)) then
   Print("--- test 113: bitwidth ", bitwidth, " Pease RDFT(128, 8, 8, 1): ", HDLVerify(streamRDFT(128, 8, 8, 1, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=114) or (which=-1)) then
   Print("--- test 114: bitwidth ", bitwidth, " Strm. RDFT(128, 8, 8, 3): ", HDLVerify(streamRDFT(128, 8, 8, 3, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=115) or (which=-1)) then
   Print("--- test 115: bitwidth ", bitwidth, " Pease RDFT(128, 8, 16, 1): ", HDLVerify(streamRDFT(128, 8, 16, 1, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=116) or (which=-1)) then
   Print("--- test 116: bitwidth ", bitwidth, " Strm. RDFT(128, 8, 16, 3): ", HDLVerify(streamRDFT(128, 8, 16, 3, 1), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=117) or (which=-1)) then
   Print("--- test 117: bitwidth ", bitwidth, " Strm. RDFTSkew(32, 1/4, 4, 4): ", HDLVerify(streamRDFTSkewUnroll(32,1/4,4,4), rDFTSkew(32,1/4), opts), "\n");
fi;
if ((which=118) or (which=-1)) then
   Print("--- test 118: bitwidth ", bitwidth, " Strm. RDFTSkew(32, 1/8, 4, 8): ", HDLVerify(streamRDFTSkewUnroll(32,1/8,4,8), rDFTSkew(32,1/8), opts), "\n");
fi;
if ((which=119) or (which=-1)) then
   Print("--- test 119: bitwidth ", bitwidth, " Strm. RDFTSkew(64,0,8,8) (r8/r4): ", HDLVerify(streamRDFTSkewUnroll(64,0,8,8), rDFTSkew(64,0), opts), "\n");
fi;
if ((which=120) or (which=-1)) then
   Print("--- test 120: bitwidth ", bitwidth, " Strm. RDFTSkew(64,0,8,16) (r8/r4): ", HDLVerify(streamRDFTSkewUnroll(64,0,8,16), rDFTSkew(64,0), opts), "\n");
fi;
if ((which=121) or (which=-1)) then
   Print("--- test 121: bitwidth ", bitwidth, " Strm. RDFTSkew(256,0,16,16) (r16/r4): ", HDLVerify(streamRDFTSkewUnroll(256,0,16,16), rDFTSkew(256,0), opts), "\n");
fi;
if ((which=122) or (which=-1)) then
   Print("--- test 122: bitwidth ", bitwidth, " Strm. RDFTSkew(512,0,16,16) (r16/r8): ", HDLVerify(streamRDFTSkewUnroll(512,0,16,16), rDFTSkew(512,0), opts), "\n");
fi;
if ((which=123) or (which=-1)) then
   Print("--- test 123: bitwidth ", bitwidth, " Pease WHT(16), w=2: ", HDLVerify(streamGen(WHT(4).withTags([AStream(2)]), opts), WHT(4), opts), "\n"); 
fi;
if ((which=124) or (which=-1)) then
   Print("--- test 124: bitwidth ", bitwidth, " Pease WHT(16), w=4: ", HDLVerify(streamGen(WHT(4).withTags([AStream(4)]), opts), WHT(4), opts), "\n"); 
fi;
if ((which=125) or (which=-1)) then
   Print("--- test 125: bitwidth ", bitwidth, " Pease WHT(16), w=8: ", HDLVerify(streamGen(WHT(4).withTags([AStream(8)]), opts), WHT(4), opts), "\n"); 
fi;
if ((which=126) or (which=-1)) then
   Print("--- test 126: bitwidth ", bitwidth, " Pease WHT(16), w=16: ", HDLVerify(streamGen(WHT(4).withTags([AStream(16)]), opts), WHT(4), opts), "\n"); 
fi;
if ((which=127) or (which=-1)) then
   Print("--- test 127: bitwidth ", bitwidth, " Pease WHT(64), w=2: ", HDLVerify(streamGen(WHT(6).withTags([AStream(2)]), opts), WHT(6), opts), "\n"); 
fi;
if ((which=128) or (which=-1)) then
   Print("--- test 128: bitwidth ", bitwidth, " Streaming L(16,2), w=1: ", HDLVerify(streamGen(TL(16,2).withTags([AStream(1)]), opts), TL(16,2), opts), "\n"); 
fi;
if ((which=129) or (which=-1)) then
   Print("--- test 129: bitwidth ", bitwidth, " Streaming L(16,2), w=2: ", HDLVerify(streamGen(TL(16,2).withTags([AStream(2)]), opts), TL(16,2), opts), "\n"); 
fi;
if ((which=130) or (which=-1)) then
   Print("--- test 130: bitwidth ", bitwidth, " Streaming L(16,2), w=4: ", HDLVerify(streamGen(TL(16,2).withTags([AStream(4)]), opts), TL(16,2), opts), "\n"); 
fi;
if ((which=131) or (which=-1)) then
   Print("--- test 131: bitwidth ", bitwidth, " Streaming L(16,2), w=8: ", HDLVerify(streamGen(TL(16,2).withTags([AStream(8)]), opts), TL(16,2), opts), "\n"); 
fi;
if ((which=132) or (which=-1)) then
   Print("--- test 132: bitwidth ", bitwidth, " Streaming L(16,2), w=16: ", HDLVerify(streamGen(TL(16,2).withTags([AStream(16)]), opts), TL(16,2), opts), "\n"); 
fi;
if ((which=133) or (which=-1)) then
   Print("--- test 133: bitwidth ", bitwidth, " Streaming RC(L(16,2)), w=4: ", HDLVerify(streamGen(TRC(TL(16,2)).withTags([AStream(4)]), opts), TRC(TL(16,2)), opts), "\n"); 
fi;
if ((which=134) or (which=-1)) then
   Print("--- test 134: bitwidth ", bitwidth, " Streaming RC(L(16,2)), w=32: ", HDLVerify(streamGen(TRC(TL(16,2)).withTags([AStream(32)]), opts), TRC(TL(16,2)), opts), "\n");
fi;
if ((which=135) or (which=-1)) then
   Print("--- test 135: bitwidth ", bitwidth, " Streaming R(16,2), w=2: ", HDLVerify(streamGen(TDR(16,2).withTags([AStream(2)]), opts), TDR(16,2), opts), "\n"); 
fi;
if ((which=136) or (which=-1)) then
   Print("--- test 136: bitwidth ", bitwidth, " Streaming R(16,2), w=8: ", HDLVerify(streamGen(TDR(16,2).withTags([AStream(8)]), opts), TDR(16,2), opts), "\n"); 
fi;
if ((which=137) or (which=-1)) then
   Print("--- test 137: bitwidth ", bitwidth, " Streaming R(16,2), w=16: ", HDLVerify(streamGen(TDR(16,2).withTags([AStream(16)]), opts), TDR(16,2), opts), "\n");
fi;
if ((which=138) or (which=-1)) then
   Print("--- test 138: bitwidth ", bitwidth, " Streaming RC(R(16,2)), w=32: ", HDLVerify(streamGen(TRC(TDR(16,2)).withTags([AStream(32)]), opts), TRC(TDR(16,2)), opts), "\n");
fi;
if ((which=139) or (which=-1)) then
   Print("--- test 139: bitwidth ", bitwidth, " Streaming random perm(16), w=1: ", HDLVerify2(StreamRandomPerm(16,1), opts), "\n"); 
fi;
if ((which=140) or (which=-1)) then
   Print("--- test 140: bitwidth ", bitwidth, " Streaming random perm(16), w=2: ", HDLVerify2(StreamRandomPerm(16,2), opts), "\n"); 
fi;
if ((which=141) or (which=-1)) then
   Print("--- test 141: bitwidth ", bitwidth, " Streaming random perm(16), w=4: ", HDLVerify2(StreamRandomPerm(16,4), opts), "\n"); 
fi;
if ((which=142) or (which=-1)) then
   Print("--- test 142: bitwidth ", bitwidth, " Streaming random perm(16), w=8: ", HDLVerify2(StreamRandomPerm(16,8), opts), "\n"); 
fi;
if ((which=143) or (which=-1)) then
   Print("--- test 143: bitwidth ", bitwidth, " Streaming random perm(16), w=16: ", HDLVerify2(StreamRandomPerm(16,16), opts), "\n"); 
fi;
if ((which=144) or (which=-1)) then
   Print("--- test 144: bitwidth ", bitwidth, " Streaming random perm(20), w=2: ", HDLVerify2(StreamRandomPerm(20,2), opts), "\n"); 
fi;
if ((which=145) or (which=-1)) then
   Print("--- test 145: bitwidth ", bitwidth, " Streaming random perm(20), w=4: ", HDLVerify2(StreamRandomPerm(20,4), opts), "\n"); 
fi;
if ((which=146) or (which=-1)) then
   Print("--- test 146: bitwidth ", bitwidth, " Streaming random perm(84), w=2: ", HDLVerify2(StreamRandomPerm(84,2), opts), "\n"); 
fi;
if ((which=147) or (which=-1)) then
   Print("--- test 147: bitwidth ", bitwidth, " Streaming random perm(84), w=3: ", HDLVerify2(StreamRandomPerm(84,3), opts), "\n"); 
fi;
if ((which=148) or (which=-1)) then
   Print("--- test 148: bitwidth ", bitwidth, " Streaming random perm(84), w=4: ", HDLVerify2(StreamRandomPerm(84,4), opts), "\n"); 
fi;
if ((which=149) or (which=-1)) then
   Print("--- test 149: bitwidth ", bitwidth, " Streaming random perm(84), w=6: ", HDLVerify2(StreamRandomPerm(84,6), opts), "\n"); 
fi;
if ((which=150) or (which=-1)) then
   Print("--- test 150: bitwidth ", bitwidth, " Streaming random perm(84), w=7: ", HDLVerify2(StreamRandomPerm(84,7), opts), "\n"); 
fi;
if ((which=151) or (which=-1)) then
   Print("--- test 151: bitwidth ", bitwidth, " Streaming random perm(84), w=12: ", HDLVerify2(StreamRandomPerm(84,12), opts), "\n"); 
fi;
if ((which=152) or (which=-1)) then
   Print("--- test 152: bitwidth ", bitwidth, " Streaming random perm(84), w=14: ", HDLVerify2(StreamRandomPerm(84,14), opts), "\n"); 
fi;
if ((which=153) or (which=-1)) then
   Print("--- test 153: bitwidth ", bitwidth, " Streaming random perm(84), w=21: ", HDLVerify2(StreamRandomPerm(84,21), opts), "\n"); 
fi;
if ((which=154) or (which=-1)) then
   Print("--- test 154: bitwidth ", bitwidth, " Streaming random perm(84), w=28: ", HDLVerify2(StreamRandomPerm(84,28), opts), "\n"); 
fi;
if ((which=155) or (which=-1)) then
   Print("--- test 155: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 4, 1): ", HDLVerify(streamRDFT(32,2,4,1,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=156) or (which=-1)) then
   Print("--- test 156: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 8, 1): ", HDLVerify(streamRDFT(32,2,8,1,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=157) or (which=-1)) then
   Print("--- test 157: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 16, 1): ", HDLVerify(streamRDFT(32,2,16,1,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=158) or (which=-1)) then
   Print("--- test 158: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 32, 1): ", HDLVerify(streamRDFT(32,2,32,1,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=159) or (which=-1)) then
   Print("--- test 159: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 4, 2): ", HDLVerify(streamRDFT(32,2,4,2,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=160) or (which=-1)) then
   Print("--- test 160: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 8, 2): ", HDLVerify(streamRDFT(32,2,8,2,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=161) or (which=-1)) then
   Print("--- test 161: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 16, 2): ", HDLVerify(streamRDFT(32,2,16,2,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=162) or (which=-1)) then
   Print("--- test 162: bitwidth ", bitwidth, " Pease CHS RDFT(32, 2, 32, 2): ", HDLVerify(streamRDFT(32,2,32,2,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=163) or (which=-1)) then
   Print("--- test 163: bitwidth ", bitwidth, " Strm  CHS RDFT(32, 2, 4, 4): ", HDLVerify(streamRDFT(32,2,4,4,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=164) or (which=-1)) then
   Print("--- test 164: bitwidth ", bitwidth, " Strm  CHS RDFT(32, 2, 8, 4): ", HDLVerify(streamRDFT(32,2,8,4,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=165) or (which=-1)) then
   Print("--- test 165: bitwidth ", bitwidth, " Strm  CHS RDFT(32, 2, 16, 4): ", HDLVerify(streamRDFT(32,2,16,4,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=166) or (which=-1)) then
   Print("--- test 166: bitwidth ", bitwidth, " Strm  CHS RDFT(32, 2, 32, 4): ", HDLVerify(streamRDFT(32,2,32,4,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=167) or (which=-1)) then
   Print("--- test 167: bitwidth ", bitwidth, " Pease CHS RDFT(32, 4, 8, 1): ", HDLVerify(streamRDFT(32,4,8,1,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=168) or (which=-1)) then
   Print("--- test 168: bitwidth ", bitwidth, " Pease CHS RDFT(32, 4, 16, 1): ", HDLVerify(streamRDFT(32,4,16,1,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=169) or (which=-1)) then
   Print("--- test 169: bitwidth ", bitwidth, " Pease CHS RDFT(32, 4, 32, 1): ", HDLVerify(streamRDFT(32,4,32,1,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=170) or (which=-1)) then
   Print("--- test 170: bitwidth ", bitwidth, " Strm  CHS RDFT(32, 4, 8, 2): ", HDLVerify(streamRDFT(32,4,8,2,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=171) or (which=-1)) then
   Print("--- test 171: bitwidth ", bitwidth, " Strm  CHS RDFT(32, 4, 16, 2): ", HDLVerify(streamRDFT(32,4,16,2,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=172) or (which=-1)) then
   Print("--- test 172: bitwidth ", bitwidth, " Strm  CHS RDFT(32, 4, 32, 2): ", HDLVerify(streamRDFT(32,4,32,2,2), PkRDFT1(32,1), opts), "\n");
fi;
if ((which=173) or (which=-1)) then
   Print("--- test 173: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 4, 1): ", HDLVerify(streamRDFT(128,2,4,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=174) or (which=-1)) then
   Print("--- test 174: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 8, 1): ", HDLVerify(streamRDFT(128,2,8,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=175) or (which=-1)) then
   Print("--- test 175: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 16, 1): ", HDLVerify(streamRDFT(128,2,16,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=176) or (which=-1)) then
   Print("--- test 176: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 32, 1): ", HDLVerify(streamRDFT(128,2,32,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=177) or (which=-1)) then
   Print("--- test 177: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 4, 2): ", HDLVerify(streamRDFT(128,2,4,2,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=178) or (which=-1)) then
   Print("--- test 178: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 8, 2): ", HDLVerify(streamRDFT(128,2,8,2,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=179) or (which=-1)) then
   Print("--- test 179: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 16, 2): ", HDLVerify(streamRDFT(128,2,16,2,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=180) or (which=-1)) then
   Print("--- test 180: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 32, 2): ", HDLVerify(streamRDFT(128,2,32,2,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=181) or (which=-1)) then
   Print("--- test 181: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 4, 3): ", HDLVerify(streamRDFT(128,2,4,3,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=182) or (which=-1)) then
   Print("--- test 182: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 8, 3): ", HDLVerify(streamRDFT(128,2,8,3,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=183) or (which=-1)) then
   Print("--- test 183: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 16, 3): ", HDLVerify(streamRDFT(128,2,16,3,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=184) or (which=-1)) then
   Print("--- test 184: bitwidth ", bitwidth, " Pease CHS RDFT(128, 2, 32, 3): ", HDLVerify(streamRDFT(128,2,32,3,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=185) or (which=-1)) then
   Print("--- test 185: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 2, 4, 6): ", HDLVerify(streamRDFT(128,2,4,6,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=186) or (which=-1)) then
   Print("--- test 186: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 2, 8, 6): ", HDLVerify(streamRDFT(128,2,8,6,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=187) or (which=-1)) then
   Print("--- test 187: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 2, 16, 6): ", HDLVerify(streamRDFT(128,2,16,6,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=188) or (which=-1)) then
   Print("--- test 188: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 2, 32, 6): ", HDLVerify(streamRDFT(128,2,32,6,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=189) or (which=-1)) then
   Print("--- test 189: bitwidth ", bitwidth, " Pease CHS RDFT(128, 4, 8, 1): ", HDLVerify(streamRDFT(128,4,8,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=190) or (which=-1)) then
   Print("--- test 190: bitwidth ", bitwidth, " Pease CHS RDFT(128, 4, 16, 1): ", HDLVerify(streamRDFT(128,4,16,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=191) or (which=-1)) then
   Print("--- test 191: bitwidth ", bitwidth, " Pease CHS RDFT(128, 4, 32, 1): ", HDLVerify(streamRDFT(128,4,32,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=192) or (which=-1)) then
   Print("--- test 192: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 4, 8, 3): ", HDLVerify(streamRDFT(128,4,8,3,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=193) or (which=-1)) then
   Print("--- test 193: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 4, 16, 3): ", HDLVerify(streamRDFT(128,4,16,3,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=194) or (which=-1)) then
   Print("--- test 194: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 4, 32, 3): ", HDLVerify(streamRDFT(128,4,32,3,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=195) or (which=-1)) then
   Print("--- test 195: bitwidth ", bitwidth, " Pease CHS RDFT(128, 8, 16, 1): ", HDLVerify(streamRDFT(128,8,16,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=196) or (which=-1)) then
   Print("--- test 196: bitwidth ", bitwidth, " Pease CHS RDFT(128, 8, 32, 1): ", HDLVerify(streamRDFT(128,8,32,1,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=197) or (which=-1)) then
   Print("--- test 197: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 8, 16, 2): ", HDLVerify(streamRDFT(128,8,16,2,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=198) or (which=-1)) then
   Print("--- test 198: bitwidth ", bitwidth, " Strm  CHS RDFT(128, 8, 32, 2): ", HDLVerify(streamRDFT(128,8,32,2,2), PkRDFT1(128,1), opts), "\n");
fi;
if ((which=199) or (which=-1)) then
   Print("--- test 199: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=4, dout=1, din=1: ", HDLVerify(streamBluesteinDFT(7,2,4,1,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=200) or (which=-1)) then
   Print("--- test 200: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=4, dout=1, din=2: ", HDLVerify(streamBluesteinDFT(7,2,4,1,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=201) or (which=-1)) then
   Print("--- test 201: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=4, dout=1, din=4: ", HDLVerify(streamBluesteinDFT(7,2,4,1,4), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=202) or (which=-1)) then
   Print("--- test 202: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=4, dout=2, din=1: ", HDLVerify(streamBluesteinDFT(7,2,4,2,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=203) or (which=-1)) then
   Print("--- test 203: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=4, dout=2, din=2: ", HDLVerify(streamBluesteinDFT(7,2,4,2,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=204) or (which=-1)) then
   Print("--- test 204: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=4, dout=2, din=4: ", HDLVerify(streamBluesteinDFT(7,2,4,2,4), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=205) or (which=-1)) then
   Print("--- test 205: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=8, dout=1, din=1: ", HDLVerify(streamBluesteinDFT(7,2,8,1,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=206) or (which=-1)) then
   Print("--- test 206: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=8, dout=1, din=2: ", HDLVerify(streamBluesteinDFT(7,2,8,1,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=207) or (which=-1)) then
   Print("--- test 207: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=8, dout=1, din=4: ", HDLVerify(streamBluesteinDFT(7,2,8,1,4), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=208) or (which=-1)) then
   Print("--- test 208: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=8, dout=2, din=1: ", HDLVerify(streamBluesteinDFT(7,2,8,2,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=209) or (which=-1)) then
   Print("--- test 209: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=8, dout=2, din=2: ", HDLVerify(streamBluesteinDFT(7,2,8,2,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=210) or (which=-1)) then
   Print("--- test 210: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=8, dout=2, din=4: ", HDLVerify(streamBluesteinDFT(7,2,8,2,4), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=211) or (which=-1)) then
   Print("--- test 211: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=16, dout=1, din=1: ", HDLVerify(streamBluesteinDFT(7,2,16,1,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=212) or (which=-1)) then
   Print("--- test 212: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=16, dout=1, din=2: ", HDLVerify(streamBluesteinDFT(7,2,16,1,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=213) or (which=-1)) then
   Print("--- test 213: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=16, dout=1, din=4: ", HDLVerify(streamBluesteinDFT(7,2,16,1,4), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=214) or (which=-1)) then
   Print("--- test 214: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=16, dout=2, din=1: ", HDLVerify(streamBluesteinDFT(7,2,16,2,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=215) or (which=-1)) then
   Print("--- test 215: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=16, dout=2, din=2: ", HDLVerify(streamBluesteinDFT(7,2,16,2,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=216) or (which=-1)) then
   Print("--- test 216: bitwidth ", bitwidth, " Bluestein DFT7 r=2, w=16, dout=2, din=4: ", HDLVerify(streamBluesteinDFT(7,2,16,2,4), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=217) or (which=-1)) then
   Print("--- test 217: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=8, dout=1, din=1: ", HDLVerify(streamBluesteinDFT(7,4,8,1,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=218) or (which=-1)) then
   Print("--- test 218: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=8, dout=1, din=2: ", HDLVerify(streamBluesteinDFT(7,4,8,1,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=219) or (which=-1)) then
   Print("--- test 219: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=8, dout=2, din=1: ", HDLVerify(streamBluesteinDFT(7,4,8,2,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=220) or (which=-1)) then
   Print("--- test 220: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=8, dout=2, din=2: ", HDLVerify(streamBluesteinDFT(7,4,8,2,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=221) or (which=-1)) then
   Print("--- test 221: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=16, dout=1, din=1: ", HDLVerify(streamBluesteinDFT(7,4,16,1,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=222) or (which=-1)) then
   Print("--- test 222: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=16, dout=1, din=2: ", HDLVerify(streamBluesteinDFT(7,4,16,1,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=223) or (which=-1)) then
   Print("--- test 223: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=16, dout=2, din=1: ", HDLVerify(streamBluesteinDFT(7,4,16,2,1), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=224) or (which=-1)) then
   Print("--- test 224: bitwidth ", bitwidth, " Bluestein DFT7 r=4, w=16, dout=2, din=2: ", HDLVerify(streamBluesteinDFT(7,4,16,2,2), RC(DFT(7)), opts), "\n"); 
fi;
if ((which=225) or (which=-1)) then
   Print("--- test 225: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=2, w=12, d=1: ", HDLVerify(streamDFTPease(48, 2, 12, 1), RC(DFT(48)), opts), "\n");
fi;
if ((which=226) or (which=-1)) then
   Print("--- test 226: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=2, w=24, d=1: ", HDLVerify(streamDFTPease(48, 2, 24, 1), RC(DFT(48)), opts), "\n");
fi;
if ((which=227) or (which=-1)) then
   Print("--- test 227: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=2, w=48, d=1: ", HDLVerify(streamDFTPease(48, 2, 48, 1), RC(DFT(48)), opts), "\n");
fi;
if ((which=228) or (which=-1)) then
   Print("--- test 228: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=4, w=24, d=1: ", HDLVerify(streamDFTPease(48, 4, 24, 1), RC(DFT(48)), opts), "\n");
fi;
if ((which=229) or (which=-1)) then
   Print("--- test 229: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=4, w=48, d=1: ", HDLVerify(streamDFTPease(48, 4, 48, 1), RC(DFT(48)), opts), "\n");
fi;
if ((which=230) or (which=-1)) then
   Print("--- test 230: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=2, w=12, d=4: ", HDLVerify(streamDFTUnroll(48, 2, 12), RC(DFT(48)), opts), "\n");
fi;
if ((which=231) or (which=-1)) then
   Print("--- test 231: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=2, w=24, d=4: ", HDLVerify(streamDFTUnroll(48, 2, 24), RC(DFT(48)), opts), "\n");
fi;
if ((which=232) or (which=-1)) then
   Print("--- test 232: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=2, w=48, d=4: ", HDLVerify(streamDFTUnroll(48, 2, 48), RC(DFT(48)), opts), "\n");
fi;
if ((which=233) or (which=-1)) then
   Print("--- test 233: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=4, w=24, d=4: ", HDLVerify(streamDFTUnroll(48, 4, 24), RC(DFT(48)), opts), "\n");
fi;
if ((which=234) or (which=-1)) then
   Print("--- test 234: bitwidth ", bitwidth, " Mixed-Radix DFT48 r=4, w=48, d=4: ", HDLVerify(streamDFTUnroll(48, 4, 48), RC(DFT(48)), opts), "\n");
fi;
if ((which=235) or (which=-1)) then
   Print("--- test 235: bitwidth ", bitwidth, " Pease DFT27 r=3, w=6: ", HDLVerify(streamDFTPease(27,3,6,1), RC(DFT(27)), opts), "\n"); 
fi;
if ((which=236) or (which=-1)) then
   Print("--- test 236: bitwidth ", bitwidth, " Pease DFT27 r=3, w=18: ", HDLVerify(streamDFTPease(27,3,18,1), RC(DFT(27)), opts), "\n"); 
fi;
if ((which=237) or (which=-1)) then
   Print("--- test 237: bitwidth ", bitwidth, " Pease DFT81 r=3, w=6, d=1: ", HDLVerify(streamDFTPease(81,3,6,1), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=238) or (which=-1)) then
   Print("--- test 238: bitwidth ", bitwidth, " Pease DFT81 r=3, w=18, d=1: ", HDLVerify(streamDFTPease(81,3,18,1), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=239) or (which=-1)) then
   Print("--- test 239: bitwidth ", bitwidth, " Pease DFT81 r=3, w=54, d=1: ", HDLVerify(streamDFTPease(81,3,54,1), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=240) or (which=-1)) then
   Print("--- test 240: bitwidth ", bitwidth, " Pease DFT81 r=3, w=6, d=2: ", HDLVerify(streamDFTPease(81,3,6,2), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=241) or (which=-1)) then
   Print("--- test 241: bitwidth ", bitwidth, " Pease DFT81 r=3, w=18, d=2: ", HDLVerify(streamDFTPease(81,3,18,2), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=242) or (which=-1)) then
   Print("--- test 242: bitwidth ", bitwidth, " Pease DFT81 r=3, w=54, d=2: ", HDLVerify(streamDFTPease(81,3,54,2), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=243) or (which=-1)) then
   Print("--- test 243: bitwidth ", bitwidth, " Pease DFT81 r=9, w=18, d=1: ", HDLVerify(streamDFTPease(81,9,18,1), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=244) or (which=-1)) then
   Print("--- test 244: bitwidth ", bitwidth, " Pease DFT81 r=9, w=54, d=1: ", HDLVerify(streamDFTPease(81,9,54,1), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=245) or (which=-1)) then
   Print("--- test 245: bitwidth ", bitwidth, " Streaming DFT27 r=3, w=6: ", HDLVerify(streamDFTUnroll(27,3,6), RC(DFT(27)), opts), "\n"); 
fi;
if ((which=246) or (which=-1)) then
   Print("--- test 246: bitwidth ", bitwidth, " Streaming DFT27 r=3, w=18: ", HDLVerify(streamDFTUnroll(27,3,18), RC(DFT(27)), opts), "\n"); 
fi;
if ((which=247) or (which=-1)) then
   Print("--- test 247: bitwidth ", bitwidth, " Streaming DFT81 r=3, w=6: ", HDLVerify(streamDFTUnroll(81,3,6), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=248) or (which=-1)) then
   Print("--- test 248: bitwidth ", bitwidth, " Streaming DFT81 r=3, w=18: ", HDLVerify(streamDFTUnroll(81,3,18), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=249) or (which=-1)) then
   Print("--- test 249: bitwidth ", bitwidth, " Streaming DFT81 r=3, w=54: ", HDLVerify(streamDFTUnroll(81,3,54), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=250) or (which=-1)) then
   Print("--- test 250: bitwidth ", bitwidth, " Streaming DFT81 r=9, w=18: ", HDLVerify(streamDFTUnroll(81,9,18), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=251) or (which=-1)) then
   Print("--- test 251: bitwidth ", bitwidth, " Streaming DFT81 r=9, w=54: ", HDLVerify(streamDFTUnroll(81,9,54), RC(DFT(81)), opts), "\n"); 
fi;
if ((which=252) or (which=-1)) then
   Print("--- test 252: bitwidth ", bitwidth, " Streaming IxL(16,2), w=1: ", HDLVerify(streamGen(TL(16,2,4,1).withTags([AStream(1)]), opts), TL(16,2,4,1), opts), "\n"); 
fi;
if ((which=253) or (which=-1)) then
   Print("--- test 253: bitwidth ", bitwidth, " Streaming IxL(16,2), w=2: ", HDLVerify(streamGen(TL(16,2,4,1).withTags([AStream(2)]), opts), TL(16,2,4,1), opts), "\n"); 
fi;
if ((which=254) or (which=-1)) then
   Print("--- test 254: bitwidth ", bitwidth, " Streaming IxL(16,2), w=4: ", HDLVerify(streamGen(TL(16,2,4,1).withTags([AStream(4)]), opts), TL(16,2,4,1), opts), "\n"); 
fi;
if ((which=255) or (which=-1)) then
   Print("--- test 255: bitwidth ", bitwidth, " Streaming IxL(16,2), w=8: ", HDLVerify(streamGen(TL(16,2,4,1).withTags([AStream(8)]), opts), TL(16,2,4,1), opts), "\n"); 
fi;
if ((which=256) or (which=-1)) then
   Print("--- test 256: bitwidth ", bitwidth, " Streaming IxL(16,2), w=16: ", HDLVerify(streamGen(TTensorI(TPrm(L(16,2)), 4, APar, APar).withTags([AStream(16)]), opts), TL(16,2,4,1), opts), "\n"); 
fi;
if ((which=257) or (which=-1)) then
   Print("--- test 257: bitwidth ", bitwidth, " Streaming IxRC(L(16,2)), w=4: ", HDLVerify(streamGen(TRC(TL(16,2,4,1)).withTags([AStream(4)]), opts), TRC(TL(16,2,4,1)), opts), "\n");
fi;
if ((which=258) or (which=-1)) then
   Print("--- test 258: bitwidth ", bitwidth, " Streaming IxRC(L(16,2)), w=32: ", HDLVerify(streamGen(TRC(TTensorI(TPrm(L(16,2)), 4, APar, APar)).withTags([AStream(32)]), opts), TRC(TL(16,2,4,1)), opts), "\n");
fi;
if ((which=259) or (which=-1)) then
   Print("--- test 259: bitwidth ", bitwidth, " Streaming IxR(16,2), w=2: ", HDLVerify(streamGen(TTensorI(TDR(16,2), 4, APar, APar).withTags([AStream(2)]), opts), TTensorI(TDR(16,2),4, APar, APar), opts), "\n"); 
fi;
if ((which=260) or (which=-1)) then
   Print("--- test 260: bitwidth ", bitwidth, " Streaming IxR(16,2), w=8: ", HDLVerify(streamGen(TTensorI(TDR(16,2), 4, APar, APar).withTags([AStream(8)]), opts), TTensorI(TDR(16,2),4,APar,APar), opts), "\n"); 
fi;
if ((which=261) or (which=-1)) then
   Print("--- test 261: bitwidth ", bitwidth, " Streaming IxR(16,2), w=16: ", HDLVerify(streamGen(TTensorI(TPrm(DR(16,2)), 4, APar, APar).withTags([AStream(16)]), opts), TTensorI(TDR(16,2),4,APar,APar), opts), "\n");
fi;
if ((which=262) or (which=-1)) then
   Print("--- test 262: bitwidth ", bitwidth, " Streaming IxRC(R(16,2)), w=32: ", HDLVerify(streamGen(TRC(TTensorI(TPrm(DR(16,2)), 4, APar, APar)).withTags([AStream(32)]), opts), TRC(TTensorI(TPrm(DR(16,2)), 4, APar, APar)), opts), "\n");
fi;
if ((which=263) or (which=-1)) then
   Print("--- test 263: bitwidth ", bitwidth, " Streaming I x random perm(16), w=1: ", HDLVerify2(STensor(StreamRandomPerm(16,1),4,1), opts), "\n"); 
fi;
if ((which=264) or (which=-1)) then
   Print("--- test 264: bitwidth ", bitwidth, " Streaming I x random perm(16), w=2: ", HDLVerify2(STensor(StreamRandomPerm(16,2),4,2), opts), "\n"); 
fi;
if ((which=265) or (which=-1)) then
   Print("--- test 265: bitwidth ", bitwidth, " Streaming I x random perm(16), w=4: ", HDLVerify2(STensor(StreamRandomPerm(16,4),4,4), opts), "\n"); 
fi;
if ((which=266) or (which=-1)) then
   Print("--- test 266: bitwidth ", bitwidth, " Streaming I x random perm(16), w=8: ", HDLVerify2(STensor(StreamRandomPerm(16,8),4,8), opts), "\n"); 
fi;
if ((which=267) or (which=-1)) then
   Print("--- test 267: bitwidth ", bitwidth, " Streaming I x random perm(16), w=16: ", HDLVerify2(STensor(StreamRandomPerm(16,16),4,16), opts), "\n"); 
fi;
if ((which=268) or (which=-1)) then
   Print("--- test 268: bitwidth ", bitwidth, " Streaming I x random perm(20), w=2: ", HDLVerify2(STensor(StreamRandomPerm(20,2),4,2), opts), "\n"); 
fi;
if ((which=269) or (which=-1)) then
   Print("--- test 269: bitwidth ", bitwidth, " Streaming I x random perm(20), w=4: ", HDLVerify2(STensor(StreamRandomPerm(20,4),4,4), opts), "\n"); 
fi;
if ((which=270) or (which=-1)) then
   Print("--- test 270: bitwidth ", bitwidth, " Streaming I x random perm(84), w=2: ", HDLVerify2(STensor(StreamRandomPerm(84,2),4,2), opts), "\n"); 
fi;
if ((which=271) or (which=-1)) then
   Print("--- test 271: bitwidth ", bitwidth, " Streaming I x random perm(84), w=3: ", HDLVerify2(STensor(StreamRandomPerm(84,3),4,3), opts), "\n"); 
fi;
if ((which=272) or (which=-1)) then
   Print("--- test 272: bitwidth ", bitwidth, " Streaming I x random perm(84), w=4: ", HDLVerify2(STensor(StreamRandomPerm(84,4),4,4), opts), "\n"); 
fi;
if ((which=273) or (which=-1)) then
   Print("--- test 273: bitwidth ", bitwidth, " Streaming I x random perm(84), w=6: ", HDLVerify2(STensor(StreamRandomPerm(84,6),4,6), opts), "\n"); 
fi;
if ((which=274) or (which=-1)) then
   Print("--- test 274: bitwidth ", bitwidth, " Streaming I x random perm(84), w=7: ", HDLVerify2(STensor(StreamRandomPerm(84,7),4,7), opts), "\n"); 
fi;
if ((which=275) or (which=-1)) then
   Print("--- test 275: bitwidth ", bitwidth, " Streaming I x random perm(84), w=12: ", HDLVerify2(STensor(StreamRandomPerm(84,12),4,12), opts), "\n"); 
fi;
if ((which=276) or (which=-1)) then
   Print("--- test 276: bitwidth ", bitwidth, " Streaming I x random perm(84), w=14: ", HDLVerify2(STensor(StreamRandomPerm(84,14),4,14), opts), "\n"); 
fi;
if ((which=277) or (which=-1)) then
   Print("--- test 277: bitwidth ", bitwidth, " Streaming I x random perm(84), w=21: ", HDLVerify2(STensor(StreamRandomPerm(84,21),4,21), opts), "\n"); 
fi;
if ((which=278) or (which=-1)) then
   Print("--- test 278: bitwidth ", bitwidth, " Streaming I x random perm(84), w=28: ", HDLVerify2(STensor(StreamRandomPerm(84,28),4,28), opts), "\n"); 
fi;
end;
