
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


##    extractP3(P, k) takes a bit matrix P and a boundary k, and returns 
##    submatrix P3 where P is partitioned as 
##
##    P4 | P3
##    -------
##    P2 | P1
##    
##    If P is of size N by N, P3 is size N-k by k.
##
extractP3 := meth(P, k)
   local dim, row, i, j, res;
   dim := Length(P);
   res := [];
   for i in [1..dim-k] do
      row := [];
      for j in [dim-k+1..dim] do
         Append(row, [P[i][j]]);
      od;
      Append(res, [row]);
   od;
   return res;
end;


##    extractP1(P, k) takes a bit matrix P and a boundary k, and returns 
##    submatrix P1 where P is partitioned as 
##
##    P4 | P3
##    -------
##    P2 | P1
##    
##    P1 is size k by k.
##
extractP1 := meth(P, k)
   local dim, row, i, j, res;
   dim := Length(P);
   res := [];
   for i in [dim-k+1..dim] do
      row := [];
      for j in [dim-k+1..dim] do
         Append(row, [P[i][j]]);
      od;
      Append(res, [row]);
   od;
   return res;
end;



##    NonZeroRows(P) determines which rows of P are not all-zero.  It returns a 
##       vector (indexed starting from 1) that indicates these rows.
##
NonZeroRows := meth(P2)
   local i, l, t, res;
   res := [];

   for i in [1..Length(P2)] do
      t := 0;

      for l in [1..Length(P2[i])] do
         if (P2[i][l] <> 0 * GF(2).one) then t := t + 1; fi;
      od;

      if (t <> 0) then Append(res, [i]); fi;

   od;
   return res;
end;

##    ZeroRows(P) determines which rows of P are all zero.  It returns a 
##       vector (indexed starting from 1) that indicates these rows.
##
ZeroRows := meth(P1)
   local i, l, t, res;
   res := [];

   for i in [1..Length(P1)] do
      t := 0;

      for l in [1..Length(P1[i])] do
         if (P1[i][l] <> 0 * GF(2).one) then t := t + 1; fi;
      od;

      if (t = 0) then Append(res, [i]); fi;

   od;
   return res;
end;


##   buildH2(i, j, n, k) builds matrix H2 given i, j, n, and k as in
##      Algorithm 5.2, Case 1, Line 3, in the Streaming Permutation
##      paper.
##
buildH2 := meth(i, j, n, k)
   local res, l, c;
   res := [];
   c := 1;
   for l in [1..n-k] do
      if l in j then
         Append(res, [BasisVec(k, i[c]-1) * GF(2).one]);
	 c := c+1;
      else
	 Append(res, [ List([1..k], x -> 0 * GF(2).one) ]);
      fi;
   od;

   return TransposedMat(res);
end;


##   buildH(n, k, H2) constructs matrix H given size n, partition
##      location k, and submatrix H2.
##
##      I(n-k) |  0
##      -------|-----
##        H2   | I(k)
##
buildH := meth(n, k, H2)
   local res, i;
   res := [];
   for i in [1..n-k] do
      Append(res, [BasisVec(n, i-1) * GF(2).one]);
   od;
   for i in [n-k+1..n] do
      Append(res, [Concatenation(H2[i-(n-k)], BasisVec(k, i-(n-k)-1) * GF(2).one )]);
   od;
  
   return res;
end;


##   GetTopLeftOnes(P, k) counts the number of ones in the top 
##      left corner.  
##
GetTopLeftOnes := meth(P, k)

#    local count, x, size;

#    size := Length(P);
#    count := 1;

#    repeat x := P[count][count]; count := count+1; until x <> GF(2).one;

#    return Min2(count-2, size-k); 
    
    local size, count, x, y, Pt;

    Pt := TransposedMat(P);
    size := Length(P);
    count := 0;


    repeat 
	count := count+1;
        x := P[count];
	y := Pt[count];
    until ((x <> BasisVec(size, count-1) * GF(2).one) or (y <> BasisVec(size, count-1) * GF(2).one) or (count = size));

    return Min2(count-1, size-k);


end;

##   AddTopLeftOnes(P, l) adds l ones to the top left corner.  
##
AddTopLeftOnes := meth(P, l)
   local size, res, i, t;
   size := Length(P);
   res := [];

   for i in [1..l] do
      Append(res, [BasisVec(size+l, i-1)]*GF(2).one);
   od;

   t := List([1..l], e->0*GF(2).one);
   
   for i in [1..size] do
      Append(res, [Concatenation(t, P[i])]);
   od;

   return res;

end;


##   TransposeVector(x) transposes 1d 'horizontal' vector x into
##      a 2d 'vertical' vector.
##
TransposeVector := meth(x)
   local i, res;
   res := [];
   if IsList(x[1]) then
       for i in [1 .. Length(x)] do res[i] := x[i][1]; od;
   else
      for i in [1 .. Length(x)] do res[i] := [x[i]]; od;
   fi;
   return res;
end;



##   CheckPerm(P) returns 1 if P is a permutation matrix, 
##      and 0 otherwise.
##
CheckPerm := meth(P)
    local i, j, t, row;
    P := BitMatrixToInts(P);

    for i in [1 .. Length(P)] do
       if ( Sum( P[i]{ [ 1..Length(P[i]) ] } ) <> 1) then
	   return 0;
       fi;    
    od;

    return 1;
end;


switchCols := meth(P, a, b)
   local ptrans, row;
   ptrans := TransposedMat(P);
   row := ptrans[a];
   ptrans[a] := ptrans[b];
   ptrans[b] := row;

   return TransposedMat(ptrans);
end;

Declare(BRAMPerm);
Declare(STensor);
Declare(MyTriangulizeMat);
#Declare(BRAMPermStreamOne);
Declare(PermMatrixRowVals);

# ===============================================
# A container for a streaming permutation
# StreamPermBit(<SPL>, <q>)
#   where <SPL> is the permutation as an SPL object, and <q> 
#   is the streaming width.
# ===============================================
Class(StreamPermBit, BaseMat, SumsBase, rec(
	abbrevs := [(s, q) -> [s, q]], 

	new := (self, prm, ss) >> SPL(WithBases(self, rec(
	     _children := [prm],
	     streamSize := ss, 

	     dimensions := prm.dims(),
	     ))),

	rChildren := self >> self._children,
	rSetChild := meth(self, n, what) self._children[n] := what; end,
	child := (self, n) >> self._children[1],
	children := self >> self._children,

	createCode := meth(self) 
	   local res, p3, p1, r, k, M, N, P, x, i, j, l, K, size, count, ptrans, z, G, q1, q3, q3b, p1b, G;

       # if (self.streamSize = 1) then
#           return BRAMPermStreamOne(PermMatrixRowVals(MatSPL(self.child(1))), 
#                       self.child(1));
#       fi;               

	   k := Log2Int(self.streamSize);
	   P := self._children[1]._children[1];
	   size := Log2Int(self.dimensions[1]);

       # If our streaming width is 1, then we are done.  P = I*P.
       #    We no longer do this, because we use the general version above.
	   if (self.streamSize = 1) then
	      #return BRAMPerm(P, MatSPL(I(size))*GF(2).one, 1);
           return BRAMPerm(MatSPL(I(size))*GF(2).one, P, 1);
	   fi;

	   # if P is the bit form of I tensor P',
	   # we can remove the top-left corner bits
	   # (up to (size-k) bits, and replace them
	   # with a streaming tensor product of a 
	   # smaller permutation
	   count := GetTopLeftOnes(P,k);
	   P := P{[count+1..Length(P)]}{[count+1..Length(P)]};

	   size := size-count;
	   
	   p3 := extractP3(P, k);
	   p1 := extractP1(P, k);

	   r := Rank(p1);

	   if (r = k) then
	       M := P;
	       N := MatSPL(I(Length(M))) * GF(2).one;
	   else 
           if (CheckPerm(P) = 1) then	
	           j := NonZeroRows(p3);
	           i := ZeroRows(p1);
	       else
		       p1b := TransposedMat(Copy(p1));
		       G := MyTriangulizeMat(p1b);
		       G := TransposedMat(G);
		       q1 := p1 * G;
		       q3 := p3 * G;
		       
 		       z := LinearIndependentColumns(TransposedMat(q1));
 		       i := [];
               
 		       for l in [1 .. k] do
   		       if (l in z) then
 		               ;
               else
 		           Add(i, l);
 		       fi;
 		       od;
               
 		       q3b := [];
		       
 		       for l in [1 .. k] do
 		       Append(q3b, q3{[r+1..k]});
 		       od;
               
 		       j := LinearIndependentColumns(TransposedMat(q3b));
           fi;
           
	       K := buildH2(i, j, size, k);
           
	       N := buildH(size, k, K) * GF(2).one; 
	       M := N*P;	  
	   fi;
	   
	   if (Maximum(Maximum(N * M - P)) <> 0 * GF(2).one) then
	       Print("ERROR: Could not factor permutation matrix.\n");
	   fi;

	   if (count = 0) then
	       return BRAMPerm(N, M, self.streamSize);
	   else
	       return STensor(BRAMPerm(N,M,self.streamSize), 2^count, self.streamSize);
	   fi;
        end,
        
	print := (self,i,is) >> Print(self.name, "(", self._children[1], ", ", self.streamSize, ")"),

	toAMat := self >> AMatMat(BitsToPermMatrix(self._children[1]._children[1])),
	dims := self >> self._children[1].dims(),

	sums := self >> self,

));


# ===============================================
# A streaming permutation that has been factored into reading and writing 
# into a dual-ported RAM
# BRAMPerm(M, N)
#   where M and N are bit representations of the factorized perm
# 
# (see Pueschel, Milder, and Hoe.  "Permuting Streaming Data Using RAMs.")
# ===============================================
Class(BRAMPerm, BaseMat, SumsBase, rec(
	abbrevs := [(m,n,q)-> [m,n,q]], 

	new := (self, m, n,q) >> SPL(WithBases(self, rec(
		    _children  := [m,n],
		    dimensions := [2^Length(m), 2^Length(m)],
		    streamSize := q
	     ))),

	rChildren := self >> self._children,
	rSetChild := meth(self, n, what) self._children[n] := what; end,
	child     := (self, n) >> self._children[n],
	children  := self >> self._children,

	createCode := self >> self,
        print      := (self,i,is) >> Print(self.name, "(", BitMatrixToInts(self._children[1]), ", ", BitMatrixToInts(self._children[2]), ", ", self.streamSize, ")"), 

	toAMat := self >> AMatMat(BitsToPermMatrix(self.child(1)) * BitsToPermMatrix(self.child(2))),
	dims   := self >> [2^(Length(self._children[1])), 2^(Length(self._children[1]))],

	sums   := self >> self
));

## PermMatrixRowVals(p) takes permutation matrix p and returns a list
##   of the '1' location in each row of the matrix.
PermMatrixRowVals := meth(x)
   local length, i, res;

   res := [];
   length := Length(x);
   for i in [1 .. length] do
      res[i] := BasisVecToNum(x[i]);
   od;
   return res;
end;


#############################################################################
##
#F  MyTriangulizeMat( <mat> ) . . . . . bring a matrix in upper triangular form
##
##    This is a modified version of TriangulizeMat included with Gap.
##       I have modified it to keep track of the elementary row
##       operations performed, and return their product.
MyTriangulizeMat := function ( mat )
    local   m, n, i, j, k, row, row2, zero, G;

    if mat <> [] then 

       # get the size of the matrix
       m := Length(mat);
       n := Length(mat[1]);
       zero := 0*mat[1][1];
       G := MatSPL(I(Length(mat)));
   
       # run through all columns of the matrix
       i := 0;
       for k  in [1..n]  do
   
           # find a nonzero entry in this column
           j := i + 1;
           while j <= m and mat[j][k] = zero  do j := j + 1;  od;
   
           # if there is a nonzero entry
           if j <= m  then
   
               # increment the rank
               InfoMatrix2(k," \c");
               i := i + 1;
   
               # make its row the current row and normalize it
               row := mat[j];  mat[j] := mat[i];  mat[i] := row[k]^-1 * row;
	       row2 := G[j]; G[j] := G[i]; G[i] := row[k]^-1 * row2;
	       # Print("aG = ", G, "\n");
	       # Print("amat = ", mat, "\n");

               # clear all entries in this column
               for j  in [1..m] do
                   if  i <> j  and mat[j][k] <> zero  then
		       G[j] := G[j] - mat[j][k] * G[i];
                       mat[j] := mat[j] - mat[j][k] * mat[i];
		       # Print("bG = ", G, "\n");
		       # Print("G[i] = ", G[i], "\n");
		       # Print("mat[j][k] = ", mat[j][k], "\n");
		       # Print("bmat = ", mat, "\n");
                   fi;
               od;
   
           fi;
   
       od;

    fi;

    return G;

end;
