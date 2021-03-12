
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#D TL.withTags := (self, tags) >> CopyFields(self, rec(params := [self.params[1], self.params[2], self.params[3], self.params[4], tags]));
#D TL.getTags := self >> self.params[self.tagpos];

DoubleDivisorPairs:=function(n,m)
    local a,b,l;
    l:=[];
    for a in DivisorPairs(n) do
        Append(l,[Concatenation(a,[1,m]),Concatenation(a,[m,1])]);
        for b in DivisorPairs(m) do
            Append(l,[Concatenation(a,b)]);
        od;
    od;
    for b in DivisorPairs(m) do
        Append(l,[Concatenation([n,1],b),Concatenation([1,n],b)]);
    od;
    return l;
end;

VectorDivisible:=function(m,c,k,b,v)
#not yet taken into account : b or c =1
    local leftpart,rightpart;
    leftpart := v in DivisorsInt(c) or m=1 or (b=1 and k=1);
    rightpart:= v in DivisorsInt(b) or c=1 or k=1;
    return leftpart and rightpart;
end;

DoubleDivisorPairsVector:=function(n,m,v)
    return Filtered(DoubleDivisorPairs(n,m) , a -> VectorDivisible(a[1],a[2],a[3],a[4],v));
end;

Blocking:=function(m,c,k,b,v)
    local t, nv, PageSize, CacheLineSize;
    t:=128/v;
    PageSize:=4*1024*8*(2);
    nv:=m*c*k*b*t;
    if (nv>PageSize) then
        return When((PageSize/8<=b*c*t)and(b*c*t<=PageSize),true,false);

        #CacheLineSize:=64*8*(4);
    #else
        #if (nv>CacheLineSize) then
            #return When((CacheLineSize/8<=b*c*t)and(b*c*t<=CacheLineSize),true,false);
        #fi;
    fi;
    return true;
end;

DoubleDivisorPairsVectorBlocking:=function(n,m,v)
    return Filtered(DoubleDivisorPairsVector(n,m,v), a->Blocking(a[1],a[2],a[3],a[4],v));
end;

NewRulesFor(TL,rec(
    L_cx_real := rec(
        switch:=false,
        forTransposition := false,
        applicable := t ->  t.isTag(1, AVecRegCx) and let(P := t.params, P[1] = P[2]^2), #FF: NOTE!! can only terminate symmetrical ones...
        children := nt -> let(P:=nt.params, [[TL(P[1], P[2], P[3], P[4]*2).withTags([AVecReg(nt.getTags()[1].isa)])]]),
        apply := (t, C, Nonterms) -> CR(When(t.params[1]=t.params[2]^2, SymSPL(C[1]), C[1]))
    ),

    L_base_vec := rec(
        switch:=false,
        forTransposition := false,
        applicable := t -> t.isTag(1, AVecReg) or t.isTag(1, AVecRegCx),
        apply := (t, C, Nonterms) -> let(
            C1:=When(t.params[3]=1, [], [I(t.params[3])]),
            C2:=When(t.params[4]=1, [], [I(t.params[4])]),
            Tensor(Concat(C1, [L(t.params[1], t.params[2])], C2))
        )
#D        applicable := (self, t) >> FirstTagEq(t, AVecReg) or FirstTagEq(t, AVecRegCx),
#D        apply := (t, C, Nonterms) -> let(C1:=When(t.params[3]=1, [], [I(t.params[3])]), 
#D            C2:=When(t.params[4]=1, [], [I(t.params[4])]), Tensor(Concat(C1, [L(t.params[1], t.params[2])], C2)))),
    ),


   #L^mn_m -> (L^mn/v_m x I_v)(I_mn/v2 x L^v2_v)(I_n/v x L^m_m/v x I_v)
    L_mn_m_vec := rec(
        forTransposition := false,
        freedoms := (self, t) >> [],
        applicable := t ->
            t.params[3] = 1
            and t.params[4] = 1
            and (t.isTag(1, AVecReg) or t.isTag(1, AVecRegCx))
            and let(
                v := t.firstTag().v,
                m := t.params[2],
                n := t.params[1] / m,
                IsInt(m/v)
                and IsInt(n/v)
                and not (
                    t.params[1] = v*v
                    and t.params[2] = v
                )
            ),
        child := (nt,freedoms) -> let (v := nt.firstTag().v, [ TL(v*v, v, 1, 1).withTags(nt.getTags()) ] ),

        apply := (nt, C, cnt) -> let(
            v := nt.firstTag().v,
            v2 := v*v,
            m := nt.params[2],
            n := nt.params[1]/m,
            VTensor(L(m*n/v, m), v)
            * SymSPL(BlockVPerm(m*n/v2, v, C[1], L(v^2,v)))
            * VTensor(Tensor(I(n/v), L(m,m/v)), v)
        )

#D        applicable := (self, t) >> t.params[3] = 1 and t.params[4] = 1 and FirstTagEq(t, AVecReg) and
#D                        let (v := GetFirstTag(t).v, m:=t.params[2], n:= t.params[1]/m,
#D                               IsInt(m/v) and IsInt(n/v) and not(t.params[1]=v*v and t.params[2]=v))  ,
#D
#D        child := (nt,freedoms) -> let (v := GetFirstTag(nt).v, [TL(v*v, v, 1, 1, GetTags(nt))]),
#D
#D        apply := (nt,C,cnt) -> let(v:=GetFirstTag(nt).v, v2:=v*v, m:=nt.params[2], n:= nt.params[1]/m,
#D                VTensor(L(m*n/v, m), v) *
#D                SymSPL(BlockVPerm(m*n/v2, v, C[1], L(v^2,v))) *
#D                VTensor(Tensor(I(n/v), L(m,m/v)), v))),
    ),

#SymSPL-BlockVPerm can be replaced by Tensor(I(m*n/v2), C[1])

#L^mn_m could probably also be splitted like that:

#   #L^mn_m -> (L^mn/v_m x I_v)(I_mn/v2 x L^v2_v)(I_n/v x L^m_m/v x I_v)
#   L_mn_m_vec := rec(
#       forTransposition := false,
#       applicable := (self, t) >> t.params[3] = 1 and t.params[4] = 1 and FirstTagEq(t, AVecReg) and
#                        let (v := GetFirstTag(t).v, m:=t.params[2], n:= t.params[1]/m,
#                                  IsInt(m/v) and IsInt(n/v)),

#       freedoms := (self, t) >> [],
#       child := (nt,freedoms) -> let (v := GetFirstTag(nt).v, m:=nt.params[2], n:= nt.params[1]/m,
#         [TL(m*n/v, m, 1, v, GetTags(nt)),
#                TL(v*v, v, m*n/(v*v), 1, GetTags(nt)),
#                   TL(m,m/v,n/v, v, GetTags(nt))]),
#       apply := (nt,C,cnt) -> let(v:=GetFirstTag(nt).v, v2:=v*v, m:=nt.params[2], n:= nt.params[1]/m, C[1] * C[2] * C[3])),

#   #I x L -> BlockVPerm
#   IxLxI_BlockVPerm := rec(
#       forTransposition := false,
#       applicable := (self, t) >> FirstTagEq(t, AVecReg) and (t.params[4]=1)
#                   and let (v := GetFirstTag(t).v, (t.params[1]=v*v) and (t.params[2]=v)),
#       freedoms := (self, t) >> [],
#       child := (nt,freedoms) -> let (v := GetFirstTag(nt).v,[TL(v*v, v, 1, 1, GetTags(nt))]),
#       apply := (t, C, cnt) -> let(v:=GetFirstTag(t).v,
#SymSPL(BlockVPerm(t.params[3], v, C[1], DropTag(cnt[1]))))),


   #A x I_v -> VTensor
   IxLxI_vtensor := rec(
       forTransposition := false,
       applicable := t ->
            (
                t.isTag(1, AVecReg)
                or t.isTag(1, AVecRegCx)
            )
            and IsPosInt(t.params[4] / t.firstTag().v),
       apply := (t, C, Nonterms) -> let(
            v:=t.firstTag().v,
            l1 := When(t.params[3] > 1, [I(t.params[3])], []),
            l2 := When(t.params[4]/v > 1, [I(t.params[4]/v)], []),
            VTensor(Tensor(Concat(l1, [L(t.params[1], t.params[2])], l2)), v)
        )
#D       applicable := (self, t) >> (FirstTagEq(t, AVecReg) or FirstTagEq(t, AVecRegCx)) and
#D                              let(v:=GetFirstTag(t).v, IsPosInt(t.params[4]/v)),
#D       apply := (t, C, Nonterms) -> let(v:=GetFirstTag(t).v, l1:= When(t.params[3] > 1, [I(t.params[3])], []), l2:=When(t.params[4]/v > 1, [I(t.params[4]/v)], []) , VTensor(Tensor(Concat(l1, [L(t.params[1], t.params[2])], l2)), v))),
    ),

#GV1 shouldn't be enabled for stuff that are not strides!
#It is highly disruptive because it doesn't respect the VTensor VPerm nomenclatura
#F L_GV1: TL(kmbc,kb,r,s) = I_r tensor ((L(kbm,bk) tensor I_c) * (I_m tensor TL(bc,b,k,1)) * (I_m tensor L(kc,k) tensor I_b)) tensor I_s
    L_GV1 := rec(
        switch:=false,
        forTransposition := false,
        freedoms := nt -> [
            MapN(
                DoubleDivisorPairsVector(nt.params[1] / nt.params[2], nt.params[2], nt.firstTag().v),
                (m,c,k,b) -> [b,c]
            )
        ],
        child := (nt,freedoms) -> let(b:=freedoms[1][1],c:=freedoms[1][2], [TL(b*c,b,1,1).withTags(nt.getTags())]),
        apply := (nt,C,cnt) -> let(
            n:=nt.params[1],
            b:=cnt[1].params[2],
            c:=cnt[1].params[1]/b,
            k:=nt.params[2]/b,
            m:=nt.params[1]/(k*b*c),
            r:=nt.params[3],
            s:=nt.params[4],
            Tensor(
                I(r),
                Tensor(
                    Tensor(L(k*b*m,b*k),I(c))
                    * Tensor(I(m),Tensor(I(k),C[1]))
                    * Tensor(Tensor(I(m),L(k*c,k)),I(b)),
                    I(s))
            )
        )
#D       freedoms := nt -> [MapN(DoubleDivisorPairsVector(nt.params[1]/nt.params[2],nt.params[2],GetFirstTag(nt).v),(m,c,k,b) -> [b,c])],
#D       child := (nt,freedoms) -> let(b:=freedoms[1][1],c:=freedoms[1][2], [TL(b*c,b,1,1,GetTags(nt))]),
#D       apply := (nt,C,cnt) -> let(
#D       n:=nt.params[1],b:=cnt[1].params[2],c:=cnt[1].params[1]/b,k:=nt.params[2]/b,m:=nt.params[1]/(k*b*c),r:=nt.params[3],s:=nt.params[4],
#D       Tensor(I(r),(Tensor(
#D               Tensor(L(k*b*m,b*k),I(c))*
#D               Tensor(I(m),Tensor(I(k),C[1]))*
#D               Tensor(Tensor(I(m),L(k*c,k)),I(b))
#D               , I(s)))))),
    ),


    L_GV1_vtensor := rec(
        switch:=false,
        forTransposition := false,
        applicable := nt -> let(p := nt.params,
            nt.isTag(1, AVecReg)
            and p[1] <> p[2]
            and p[2] <> 1
            and Length(
                DoubleDivisorPairsVectorBlocking(p[1] / p[2], p[2], nt.firstTag().v)
            ) > 0
        ),
        freedoms := nt -> let(p := nt.params, [
            MapN(
                DoubleDivisorPairsVectorBlocking(p[1] / p[2], p[2], nt.firstTag().v),
                (m,c,k,b) -> [b,c]
            )
        ]),
        child := (nt, freedoms) -> let(
            b:=freedoms[1][1],
            c:=freedoms[1][2],
            [ TL(b*c,b,1,1).withTags(nt.getTags()) ]
        ),
        apply := (nt,C,cnt) -> let(
            n := nt.params[1],
            b := cnt[1].params[2],
            c := cnt[1].params[1]/b,
            k := nt.params[2]/b,
            m := nt.params[1]/(k*b*c),
            r := nt.params[3],
            s := nt.params[4],
            v := nt.firstTag().v,
#            garbage:=fPrint(["n:",n," b:",b," c:",c," k:",k," m:",m," r:",r," s:",s, " v:",v,"\n"]),
            prefactor := When(r = 1, [], [I(r)]),
            firstfactor := When(m = 1 or (b = 1 and k = 1),
                [],
                [When(c=1,L(k*b*m,b*k), VTensor(Tensor(L(k*b*m,b*k),I(c/v)),v))]
            ),
            middlefactor := [ Tensor(I(m), Tensor(I(k), C[1])) ],
#            middlefactor:=When(k>1 and b*c<1024,[Tensor(I(m),I(k/2),BB(Tensor(I(2),C[1])))],[Tensor(I(m),Tensor(I(k),C[1]))]),
            thirdfactor := When(c=1 or k=1,
                [],
                [
                    When(b=1,
                        Tensor(I(m),L(k*c,k)),
                        VTensor(Tensor(I(m),L(k*c,k),I(b/v)),v)
                    )
                ]
            ),
            postfactor := When(r=s, [], [I(s)]),

            Tensor(Concat(
                prefactor,
                [Compose(Concat(
                    firstfactor,
                    middlefactor,
                    thirdfactor
                ))],
                postfactor
            ))
        )
#D       applicable := (self, nt)>> FirstTagEq(nt, AVecReg) and (nt.params[1]<>nt.params[2]) and (nt.params[2]<>1) and
#D           (Length(DoubleDivisorPairsVectorBlocking(nt.params[1]/nt.params[2],nt.params[2],GetFirstTag(nt).v))>0),
#D       freedoms := nt -> [MapN(DoubleDivisorPairsVectorBlocking(nt.params[1]/nt.params[2],nt.params[2],GetFirstTag(nt).v),(m,c,k,b) -> [b,c])],
#D       child := (nt,freedoms) -> let(b:=freedoms[1][1],c:=freedoms[1][2], [TL(b*c,b,1,1,GetTags(nt))]),
#D       apply := (nt,C,cnt) -> let(
#D       n:=nt.params[1],b:=cnt[1].params[2],c:=cnt[1].params[1]/b,k:=nt.params[2]/b,m:=nt.params[1]/(k*b*c),r:=nt.params[3],s:=nt.params[4],v:=GetFirstTag(nt).v,
#D#       garbage:=fPrint(["n:",n," b:",b," c:",c," k:",k," m:",m," r:",r," s:",s, " v:",v,"\n"]),
#D       prefactor  :=When(r=1, [], [I(r)]),
#D       firstfactor:=When(m=1 or (b=1 and k=1), [],[When(c=1,L(k*b*m,b*k), VTensor(Tensor(L(k*b*m,b*k),I(c/v)),v))]),
#D       middlefactor:=[Tensor(I(m),Tensor(I(k),C[1]))],
#D#      middlefactor:=When(k>1 and b*c<1024,[Tensor(I(m),I(k/2),BB(Tensor(I(2),C[1])))],[Tensor(I(m),Tensor(I(k),C[1]))]),
#D       thirdfactor:=When(c=1 or k=1, [], [When(b=1,Tensor(I(m),L(k*c,k)), VTensor(Tensor(I(m),L(k*c,k),I(b/v)),v))]),
#D       postfactor :=When(r=s, [], [I(s)]),
#D       Tensor(Concat(
#D           prefactor,
#D           [Compose(Concat(
#D               firstfactor,
#D               middlefactor,
#D               thirdfactor))
#D           ],
#D           postfactor))))
    )
));






######################################################################
#  OBSOLETE RULES
######################################################################

NewRulesFor(TL,rec(

    # L^nv_n -> (I_n/v x L^v2_v)(L^n_n/v x I_v)
    # superseeded by L_mn_m_vec with n=v
    L_nv_n_vec := rec(
        switch := false,
        forTransposition := false,
        applicable := t ->
            t.params[3] = 1
            and t.params[4] = 1
            and (t.isTag(1, AVecReg) or t.isTag(1, AVecRegCx))
            and let (v := t.firstTag().v,
                t.params[1] = t.params[2] * v
                and t.params[2] <> v
                and IsInt(t.params[2]/v)
            ),
        freedoms := (self, t) >> [],
        child := (nt,freedoms) -> let(
            v := nt.firstTag().v,
            [ TL(v*v, v, 1, 1).withTags(nt.getTags()) ]
        ),
        apply := (nt, C, cnt) -> let (
            v := nt.firstTag().v,
            n := nt.params[2],
            SymSPL(
                BlockVPerm(n/v, v, C[1], L(v^2,v))
            ) * VTensor(L(n, n/v), v)
        )
#D       applicable := (self, t) >> t.params[3] = 1 and t.params[4] = 1 and FirstTagEq(t, AVecReg) and
#D                         let (v := GetFirstTag(t).v, t.params[1] = t.params[2] * v and t.params[2] <> v and IsInt(t.params[2]/v)),
#D       freedoms := (self, t) >> [],
#D       child := (nt,freedoms) -> let (v := GetFirstTag(nt).v, [TL(v*v, v, 1, 1, GetTags(nt))]),
#D       apply := (nt,C,cnt) -> let (v := GetFirstTag(nt).v, n:= nt.params[2], SymSPL(BlockVPerm(n/v, v, C[1], L(v^2,v))) * VTensor(L(n, n/v), v))),
   #SymSPL-BlockVPerm can be replaced by Tensor(I(n/v), C[1])
    ),



   #L^nv_v -> (L^n_v x I_v)(I_n/v x L^v2_v)
   # superseeded by L_mn_m_vec with m=v
   L_nv_v_vec := rec(
       switch := false,
       forTransposition := false,
       applicable := t ->
            t.params[3] = 1
            and t.params[4] = 1
            and (t.isTag(1, AVecReg) or t.isTag(1, AVecRegCx))
            and let(v := t.firstTag().v,
                t.params[2] = v
                and t.params[1] <> v * v
                and IsInt(t.params[1]/(v*v))
            ),
       freedoms := (self, t) >> [],
       child := (nt, freedoms) -> let(v := nt.firstTag().v, [ TL(v*v, v, 1, 1).withTags(nt.getTags()) ]),
       apply := (nt, C, cnt) -> let(
            v := nt.firstTag().v,
            n:= nt.params[1]/v,
            VTensor(L(n, v), v)
            * SymSPL(BlockVPerm(n/v, v, C[1], L(v^2,v)))
        )
#D       applicable := (self, t) >> t.params[3] = 1 and t.params[4] = 1 and FirstTagEq(t, AVecReg) and
#D                        let (v := GetFirstTag(t).v, t.params[2] = v and t.params[1] <> v * v and IsInt(t.params[1]/(v*v))),
#D       freedoms := (self, t) >> [],
#D       child := (nt,freedoms) -> let (v := GetFirstTag(nt).v, [TL(v*v, v, 1, 1, GetTags(nt))]),
#D       apply := (nt,C,cnt) -> let (v := GetFirstTag(nt).v, n:= nt.params[1]/v, VTensor(L(n, v), v) * SymSPL(BlockVPerm(n/v, v, C[1], L(v^2,v))))),
   #SymSPL-BlockVPerm can be replaced by Tensor(I(n/v), C[1])))
    )

));
