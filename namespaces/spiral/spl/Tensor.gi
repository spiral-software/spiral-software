
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(Tensor, I, L);

# ==========================================================================
# Tensor(<spl1>, <spl2>, ...) . . . . . . . . . .  Kronecker product of SPLs
#    Arguments can be lists of spls of any depth. So that the following is
#    legal:
#       Tensor(I(2), [[F(2), [I(2)]]])
#    and is equivalent to
#       Tensor(I(2), F(2), I(2))
# ==========================================================================
Class(Tensor, BaseOperation, rec(
    abbrevs := [ arg -> [Flat(arg)] ],

    new := meth(self, L)
        local scl,mat;
        Constraint(IsList(L) and Length(L) >= 1);
        L:=Filtered(L,x->x<>I(1));
        if (Length(L)=0) then return I(1); fi;
        #DoForAll(L, x -> Constraint(IsSPL(x)));
	if Length(L)=1 then return L[1]; fi;

	#if Length(L)=2 then
	#    scl := Cond(Dimensions(L[1])=[1,1], 1,
	#	        Dimensions(L[2])=[1,1], 2,
	#		false);
	#    if scl <> false then
	#	mat := L [ When(scl=1, 2, 1) ];
	#	scl := MatSPL(L[scl])[1][1];
	#	
	#	return When(scl=1, mat, spiral.spl.operators.Scale(scl, mat));
	#    fi;
	#fi;

        return SPL(WithBases( self, 
	    rec( _children := L,
		dimensions := [ Product(L, t -> t.dims()[1]),
		                Product(L, t -> t.dims()[2]) ] )));
    end,
    #-----------------------------------------------------------------------
    dims := self >> [ Product(self._children, t -> t.dims()[1]),
	              Product(self._children, t -> t.dims()[2]) ],
    #-----------------------------------------------------------------------
    isPermutation := self >> ForAll(self._children, IsPermutationSPL),
    #-----------------------------------------------------------------------
    isSymmetric := self >> ForAll(self._children, c -> c.isSymmetric()),
    #-----------------------------------------------------------------------
    toAMat := self >> TensorProductAMat(List(self._children, AMatSPL)),
    #-----------------------------------------------------------------------
    transpose := self >>  
        CopyFields(self, rec(_children := List(self._children, x->x.transpose()),
		          dimensions := Reversed(self.dimensions))),
    conjTranspose := self >>  
        CopyFields(self, rec(_children := List(self._children, x->x.conjTranspose()),
		          dimensions := Reversed(self.dimensions))),
    inverse := self >>  
        CopyFields(self, rec(_children := List(self._children, x->x.inverse()),
		          dimensions := Reversed(self.dimensions))),
    #-----------------------------------------------------------------------
    rightBinary  := self >> FoldR1(self._children, (p,x) -> let(base:=self.__bases__[1], base(x, p))),

    leftBinary := self >> FoldL1(self._children, (p,x) -> let(base:=self.__bases__[1], base(p, x))),

    split := meth(self)
        local i, lft, rt, ch, res;
        ch := self.children();
        rt := Product(Drop(ch, 1), Rows);
        lft := 1;
        res := Tensor(ch[1], I(rt));
        for i in [2..self.numChildren()] do
            lft := lft * Cols(ch[i-1]);
            rt := rt / Rows(ch[i]);
            res := res * Tensor(I(lft), ch[i], I(rt));
        od;
        return res;
    end,

    vectorForm := self >> Checked(self.numChildren() = 2,
	let(c1 := self.child(1), c2 := self.child(2), 

	    Cond(IsIdentitySPL(c1) and IsIdentitySPL(c2), self,
		 IsIdentitySPL(c1), L(Rows(c2)*Rows(c1), Rows(c1)) * 
		                    Tensor(c2, I(Rows(c1))) * 
				    L(Cols(c2)*Cols(c1), Cols(c2)),
		 IsIdentitySPL(c2), self, 
	         # else
		 Tensor(c1, I(Rows(c2)) * Tensor(I(Cols(c1)), c2).vectorForm()
	    )))),

    parallelForm := self >> Checked(self.numChildren() = 2,
	let(c1 := self.child(1), c2 := self.child(2), 

	    Cond(IsIdentitySPL(c1) and IsIdentitySPL(c2), self,
		 IsIdentitySPL(c2), L(Rows(c2)*Rows(c1), Rows(c1)) * 
		                    Tensor(I(Rows(c2)), c1) * 
				    L(Cols(c2)*Cols(c1), Cols(c2)),
		 IsIdentitySPL(c1), self,
	         # else rc1*rc2 X cc1*cc2
		 Tensor(c1, I(Rows(c2))).parallelForm() * Tensor(I(Cols(c1)), c2)
	    ))),

    
    #-----------------------------------------------------------------------
    arithmeticCost := meth(self, costMul, costAddMul)
        local t, left, right, cost;
        left  := 1;
	right := self.dimensions[1];
	cost  := costMul(0) - costMul(0); # will work even when costMul(0) <> 0
	for t in self.children() do
	    right := right / t.dimensions[1];
	    cost  := cost + left * right * t.arithmeticCost(costMul, costAddMul);
	    left  := left * t.dimensions[2];
	od;
	return cost;
    end,
    #-----------------------------------------------------------------------
    latexSymbol := "\\tensor"
));
