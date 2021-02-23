
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F Perm_CCS(<n>) - SPL objects for CCS -> Perm IPP data format conversion
#F  
#F Returns an SPL object that converts the result of an RDFT in CCS format into
#F Perm format. Both of these formats are defined in the IPP manual. 
#F IPP manual does not define Perm format for odd RDFT size, so in this case
#F we defined it to coincide with CCS.
#F
#F The output of SPIRAL's PRDFT transform is equivalent to IPP CCS format. 
#F Suggested uses are:
#F     Perm_CCS(n) * PRDFT(n) 
#F     IPRDFT(n) * Perm_CCS(n).transpose()
#F 
Perm_CCS := n -> When(IsEvenInt(n), 
    Gath(H(n+2,n,0,1))*Perm((2,n+1),n+2),
    DirectSum(Mat([[1,0]]), I(n-1))
);


#F Pack_CCS(<n>) - SPL objects for CCS -> Pack IPP data format conversion of RDFT of type 1
#F  
#F Returns an SPL object that converts the result of an RDFT in CCS format into
#F Pack format. Both of these formats are defined in the IPP manual. 
#F
#F The output of SPIRAL's PRDFT transform is equivalent to IPP CCS format. 
#F Suggested uses are:
#F     Pack_CCS(n) * PRDFT(n) 
#F     IPRDFT(n) * Pack_CCS(n).transpose()
#F 
Pack_CCS := n -> When(IsEvenInt(n), 
    DirectSum(Mat([[1,0]]), When(n=2, [], I(n-2)), Mat([[1,0]])),
    DirectSum(Mat([[1,0]]), I(n-1))
);

#F Pack_CCS(<n>) - SPL objects for CCS -> Pack IPP data format conversion of RDFT of type 3
#F
#F [ note that IPP does not have RDFT of type 3, we just use their format terminology ]
#F Returns an SPL object that converts the result of an RDFT in CCS format into
#F Pack format. Both of these formats are defined in the IPP manual. 
#F Perm format does not make sense for PRDFT3.
#F
#F The output of SPIRAL's PRDFT3 transform is equivalent to IPP CCS format. 
#F Suggested uses are:
#F     Pack_CCS3(n) * PRDFT3(n) 
#F     PRDFT3(n).inverse() * Pack_CCS3(n).transpose()
#F 
Pack_CCS3 := n -> When(IsEvenInt(n), 
    I(n), 
    DirectSum(I(n-1), Mat([[1,0]]))
);
