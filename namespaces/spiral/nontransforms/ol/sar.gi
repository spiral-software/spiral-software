
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#expand FData to report range as size
Class(FDataIndirect,FData,rec (
    range := self >> self.var.t.size,
));

Class(ICFDataIndirect,FDataIndirect,rec (
    range := self >> 2*self.var.t.size,
    domain := self >> 2*self.var.t.size,
));

Class(ExpandMult,NonTerminal,rec (
        abbrevs := [(height,rows,isSplit) -> [height,rows,isSplit,[]]],
        dmn := self >> let(x:=When(self.params[3],2,1),[TArray(TReal,x*self.params[1] * self.params[2]),TArray(TReal,x*self.params[2])]),
        rng := self >> let(x:=When(self.params[3],2,1),[TArray(TReal,x*self.params[1]* self.params[2])]),
        isReal := self >> true,
        tagpos := 4,
        transpose := self >> Copy(self)));


NewRulesFor(ExpandMult,rec(

       ExpandMult_Base:= rec (
        forTransposition := false,
        applicable := (self,nt) >> nt.params[1]=1,
        apply := (nt,c,cnt) -> When(nt.params[3],ICMultiplication(2,2*nt.params[2],true),ICMultiplication(2,nt.params[2]))),

       ExpandMult_One := rec (
        forTransposition := false,
        applicable := (self,nt) >> nt.params[1] > 1,
        freedoms := nt -> [[1]],
        child := (self,nt,freedoms) >> [TTensorI_OL(ExpandMult(1,nt.params[2],nt.params[3]),[APar,AOne],[[1,0]],
                        [nt.params[1]/freedoms[1],1])],
        apply := (nt,c,cnt) -> c[1]
       )));






Class(SAR_Interpolation,NonTerminal,rec(
        abbrevs := [(iters,height,rows,isSplit) -> Checked(IsPosInt(rows) and IsPosInt(iters) and
                     IsPosInt(height),[iters,height,rows,isSplit,[]]),
                    (iters,height,rows,isSplit,t) -> Checked(IsPosInt(rows) and IsPosInt(iters) and
                        IsPosInt(height), [iters,height,rows,isSplit,t])],
        dmn := self >> [TArray(TReal,2*self.params[1] * self.params[2] * self.params[3]),
                        TArray(TReal,2*self.params[1] * self.params[3])],
        rng := self >> [TArray(TReal,2*self.params[1] * self.params[3])],
        isReal := self >> true,
        isSplit := self >> self.params[4],
        numops := self >> self.params[1] * self.params[2] * self.params[3] * self.params[1],
        tagpos := 5,
        transpose := self >> Copy(self)));



NewRulesFor(SAR_Interpolation,rec(


     SAR_Interpolation_One := rec (
        forTransposition := false,
        applicable := (self,nt) >> nt.params[1] > 1 and nt.params[2] > 1,
        freedoms := nt -> [[1]],
        child := function(nt,freedoms)
		local size,tt,fd;
                size:=(nt.params[1]*nt.params[2]*nt.params[3]);
		fd:=When(nt.params[4],
			ICFDataIndirect(param(TArray(TDouble,size),"indirect")),FDataIndirect(param(TArray(TDouble,size),"indirect")));
                tt:=When(nt.params[4],ExpandMult(nt.params[2],nt.params[3],true),
                        ExpandMult(nt.params[2],nt.params[3],false));
                nt.params[5] := When(nt.params[4],nt.params[1],nt.params[2]);
		return [ICScatAcc(fPrecompute(fd))*
			   TTensorI_OL(tt,[APar,AMul(1,APar)],[[1,0]],
                        [nt.params[1]/freedoms[1],nt.params[5]/freedoms[1]])];
                end,
        apply := (nt,c,cnt) ->  let(len1:=2*nt.params[1]*nt.params[2]*nt.params[3],
			            len2:=2*nt.params[1]*nt.params[3],
						When(nt.isSplit(),Split(len1,2)*Prm(L(len1,2))*c[1]*
                                			Cross(Prm(L(len1,len1/2)),Prm(L(len2,len2/2)))*
                                			Cross(Glue(2,len1/2),Glue(2,len2/2)),c[1])),


     ) 
   )
);
  




Class(SAR2,TaggedNonTerminal,rec(
       abbrevs := [(n,s) -> Checked(IsPosInt(n),[n,s])],
       dmn := self >> [TArray(TReal,When(self.params[1]=1,2,self.params[1])),TArray(TReal,When(self.params[1]=1,2,self.params[1]))],
       rng := self >> [TArray(TReal,When(self.params[1]=1,2,self.params[1]))],
       isSplit := self >> self.params[2], 
       isReal := self >> true,
       transpose := self >> Copy(self)));

NewRulesFor(SAR2,rec(

      SAR_Base := rec (
      	forTransposition := false,
      	applicable := (self,nt) >> (nt.params[1]=1),
      	apply := (nt,c,cnt) -> ICMultiplication(2,2)),

      SAR_MatchFilter := rec (
	forTransposition := false,
        applicable := (self,nt) >> nt.params[1] >= 2, 
        freedoms := nt -> [[2]],
        child := (self,nt,freedoms) >> [TTensorI_OL(SAR2(1,nt.params[2]),[APar,AMul(1,APar)],[[1,0]],
                        [nt.params[1]/freedoms[1],nt.params[1]/freedoms[1]])],
        apply := (nt,c,cnt) -> When(nt.isSplit(),Split(nt.params[1],2)*Prm(L(nt.params[1],2))*c[1]*
				Cross(Prm(L(nt.params[1],nt.params[1]/2)),Prm(L(nt.params[1],nt.params[1]/2)))*
				Cross(Glue(2,nt.params[1]/2),Glue(2,nt.params[1]/2)),c[1]),
                            


      ),

      SAR_MatchFilterSMP := rec (
      	forTransposition := false,
 	#applicable := (self,nt) >> nt.params[1] >= 1 and nt.hasTags() and nt.isTag(1,AParSMP),
        applicable := (self,nt) >>  false,
        freedoms := nt -> Cond(nt.isTag(1, AParSMP) and
               nt.params[1] mod nt.firstTag().params[1]=0,
               [[nt.params[1]/nt.firstTag().params[1]]],
               [[1]]),
        child := (self,nt,freedoms) >> [TTensorI_OL(SAR2(freedoms[1],nt.params[2]),[APar,AMul(1,APar)],[[1,0]],
			[nt.params[1]/freedoms[1],nt.params[1]/freedoms[1]]).withTags(nt.getTags())],
        apply := (nt,c,cnt) -> c[1]
     ), 
     
     
     SAR_MatchFilterVec := rec (
         forTransposition := false,
         #applicable := (self,nt) >> nt.hasTags() and nt.isTag(1,AVecReg) and nt.params[1] > nt.firstTag().v,
         applicable := (self,nt) >> false,
         freedoms := nt -> Cond(nt.isTag(1,AVecReg) and Mod(nt.params[1],nt.firstTag().v) = 0,
			[[nt.firstTag().v]],[[1]]),
         child := (self,nt,freedoms) >> [TTensorI_OL(SAR2(nt.firstTag().v,nt.firstTag()),[AVec,AMul(1,AVec)],[[0,1]],
			 [nt.params[1]/freedoms[1],nt.params[1]/freedoms[1]]).withTags(nt.getTags())],
         apply := (nt,c,cnt) -> c[1]
     ),
   

    SAR_MatchFilterVecBase := rec (
        forTransposition := false,
        #applicable := (self,nt) >> nt.hasTags() and nt.isTag(1,AVecReg) and nt.params[1] = nt.firstTag().v,
        applicable := (self,nt) >> false,
        apply := (nt,c,cnt) -> 
                        VTensor_OL(Multiplication(2,1),nt.firstTag().v)
                        
        )


    )

 
);

     
