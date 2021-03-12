
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Load(sym);
Import(sym);
k:=4;
a:=1/8;

#Most BRDFT1, even, (x2-1)U(k-1)(q)
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(I(1),PolyDTT(DST1(k-1)),PolyDTT(DST1(k-1)),I(1)))^-1* Mat_BRDFT1(2*k,1)); #double base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(PolyDTT(DCT2(k)),PolyDTT(DST1(k-1)),I(1)))^-1* Mat_BRDFT1(2*k)); #base V and base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(I(1),PolyDTT(DST1(k-1)),PolyDTT(DST2(k))))^-1* Mat_BRDFT1(2*k)); #base U and base W
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(PolyDTT(DCT2(k)),PolyDTT(DST2(k))))^-1* Mat_BRDFT1(2*k)); #base V and base W

#Most BRDFT2, even, (x2-1)U(k-1)(q)
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(I(1),PolyDTT(DST1(k-1)),PolyDTT(DST1(k-1)),I(1)))^-1* Mat_BRDFT2(2*k, true)); #double base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(PolyDTT(DCT2(k)),PolyDTT(DST1(k-1)),I(1)))^-1* Mat_BRDFT2(2*k, true)); #base V and base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(I(1),PolyDTT(DST1(k-1)),PolyDTT(DST2(k))))^-1* Mat_BRDFT2(2*k,true)); #base U and base W
PrintMat(MatSPL(DirectSum(I(1),L(2*(k-1),k-1),I(1))*DirectSum(PolyDTT(DCT2(k)),PolyDTT(DST2(k))))^-1* Mat_BRDFT2(2*k,true)); #base V and base W

#Most BRDFT1, odd, (x-1)Wk(q)
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT5(k+1)),PolyDTT(DST5(k))))^-1* Mat_BRDFT1(2*k+1)); #base T and base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT5(k+1)),PolyDTT(DST6(k))))^-1* Mat_BRDFT1(2*k+1)); #base T and base W
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(I(1),PolyDTT(DST5(k)),PolyDTT(DST5(k))))^-1* Mat_BRDFT1(2*k+1)); #double base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT6(k+1)),PolyDTT(DST5(k))))^-1* Mat_BRDFT1(2*k+1)); #base V and base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT6(k+1)),PolyDTT(DST6(k))))^-1* Mat_BRDFT1(2*k+1)); #base V and base W
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(I(1),PolyDTT(DST6(k)),PolyDTT(DST6(k))))^-1* Mat_BRDFT1(2*k+1)); #double base W

#Most BRDFT2, odd, (x-1)Wk(q)
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT5(k+1)),PolyDTT(DST5(k))))^-1* Mat_BRDFT2(2*k+1, true)); #base T and base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT5(k+1)),PolyDTT(DST6(k))))^-1* Mat_BRDFT2(2*k+1, true)); #base T and base W
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(I(1),PolyDTT(DST5(k)),PolyDTT(DST5(k))))^-1* Mat_BRDFT2(2*k+1, true)); #double base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT6(k+1)),PolyDTT(DST5(k))))^-1* Mat_BRDFT2(2*k+1, true)); #base V and base U
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(PolyDTT(DCT6(k+1)),PolyDTT(DST6(k))))^-1* Mat_BRDFT2(2*k+1, true)); #base V and base W
PrintMat(MatSPL(DirectSum(I(1),L(2*(k),k))*DirectSum(I(1),PolyDTT(DST6(k)),PolyDTT(DST6(k))))^-1* Mat_BRDFT2(2*k+1, true)); #double base W

#Most BRDFT3, even, Tk(q)
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DCT3(k)),PolyDTT(DCT3(k))))^-1* Mat_BRDFT3(2*k,1/4)); #double base T 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DST3(k)),PolyDTT(DST3(k))))^-1* Mat_BRDFT3(2*k,1/4)); #double base U 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DCT4(k)),PolyDTT(DCT4(k))))^-1* Mat_BRDFT3(2*k,1/4)); #double base V 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DST4(k)),PolyDTT(DST4(k))))^-1* Mat_BRDFT3(2*k,1/4)); #double base W

#Most BRDFT4, even, Tk(q)
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DCT3(k)),PolyDTT(DCT3(k))))^-1* Mat_BRDFT4(2*k,1/4, true)); #double base T 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DST3(k)),PolyDTT(DST3(k))))^-1* Mat_BRDFT4(2*k,1/4, true)); #double base U 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DCT4(k)),PolyDTT(DCT4(k))))^-1* Mat_BRDFT4(2*k,1/4, true)); #double base V 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(DST4(k)),PolyDTT(DST4(k))))^-1* Mat_BRDFT4(2*k,1/4, true)); #double base W

#Most BRDFT3, odd, (x+1)Vk(q)
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DST7(k)),PolyDTT(DCT7(k+1))))^-1* Mat_BRDFT3(2*k+1,1/4)); #base U and base T  
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DCT8(k)),PolyDTT(DCT7(k+1))))^-1* Mat_BRDFT3(2*k+1,1/4)); #base V and base T  
PrintMat(MatSPL(DirectSum(L(2*k,k)*DirectSum(PolyDTT(DST7(k)),PolyDTT(DST7(k))),I(1)))^-1* Mat_BRDFT3(2*k+1,1/4));   #double base U
PrintMat(MatSPL(DirectSum(L(2*k,k)*DirectSum(PolyDTT(DCT8(k)),PolyDTT(DCT8(k))),I(1)))^-1* Mat_BRDFT3(2*k+1,1/4));   #double base V
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DST7(k)),PolyDTT(DST8(k+1))))^-1* Mat_BRDFT3(2*k+1,1/4)); #base U and base W
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DCT8(k)),PolyDTT(DST8(k+1))))^-1* Mat_BRDFT3(2*k+1,1/4)); #base V and base W

#Most BRDFT4, odd, (x+1)Vk(q)
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DST7(k)),PolyDTT(DCT7(k+1))))^-1* Mat_BRDFT4(2*k+1,1/4, true)); #base U and base T  
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DCT8(k)),PolyDTT(DCT7(k+1))))^-1* Mat_BRDFT4(2*k+1,1/4, true)); #base V and base T  
PrintMat(MatSPL(DirectSum(L(2*k,k)*DirectSum(PolyDTT(DST7(k)),PolyDTT(DST7(k))),I(1)))^-1* Mat_BRDFT4(2*k+1,1/4, true));   #double base U
PrintMat(MatSPL(DirectSum(L(2*k,k)*DirectSum(PolyDTT(DCT8(k)),PolyDTT(DCT8(k))),I(1)))^-1* Mat_BRDFT4(2*k+1,1/4, true));   #double base V
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DST7(k)),PolyDTT(DST8(k+1))))^-1* Mat_BRDFT4(2*k+1,1/4, true)); #base U and base W
PrintMat(MatSPL(DirectSum(L(2*k,k),I(1))*DirectSum(PolyDTT(DCT8(k)),PolyDTT(DST8(k+1))))^-1* Mat_BRDFT4(2*k+1,1/4, true)); #base V and base W

#Skew BRDFTs
#weird cyclotomics are 2*cospi(2*a)
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(SkewDTT(DCT3(k),2*a)),PolyDTT(SkewDTT(DCT3(k),2*a))))^-1* MatSPL(BRDFT3(2*k,a))); #double base T 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(SkewDTT(DST3(k),2*a)),PolyDTT(SkewDTT(DST3(k),2*a))))^-1* MatSPL(BRDFT3(2*k,a))); #double base U 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(SkewDTT(DCT4(k),2*a)),PolyDTT(SkewDTT(DCT4(k),2*a))))^-1* MatSPL(BRDFT3(2*k,a))); #double base V 
PrintMat(MatSPL(L(2*k,k)*DirectSum(PolyDTT(SkewDTT(DST4(k),2*a)),PolyDTT(SkewDTT(DST4(k),2*a))))^-1* MatSPL(BRDFT3(2*k,a))); #double base W

#Skew PRDFTs
PrintMat(MatSPL(L(2*k,k)*DirectSum(SkewDTT(DCT3(k),2*a),SkewDTT(DST3(k),2*a)))^-1* MatSPL(BSkewPRDFT(2*k,a))); #base T and base U

#PRDFT3, even
PrintMat(MatSPL(L(2*k,k)*DirectSum(DCT3(k),DST3(k)))^-1* MatSPL(PRDFT3(2*k))); #base T and base U

#PRDFT4, even
PrintMat(MatSPL(L(2*k,k)*DirectSum(DCT4(k),DST4(k)))^-1* MatSPL(PRDFT4(2*k))); #base V and base W
