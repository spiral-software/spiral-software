
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



Declare(CodeBlock, BRAMPermStreamOne, TPrmMulti);



Class(TPrmMulti, Tagged_tSPL_Container, rec(
    abbrevs :=  [ (tspl_list, which) -> [tspl_list, which] ],
    dims := self >> Cond(IsBound(self.params[1][1].domain), [self.params[1][1].domain(), self.params[1][1].range()], self.params[1][1].dims()),

    terminate := meth(self)
       local prms, res, len, p, i;
       prms := self.params[1];
       len := Length(prms);
       res := prms[len];
       for i in Reversed([1..len-1]) do
          p := Cond(IsBound(prms[i].terminate), prms[i].terminate(), Prm(prms[i]));
          res := COND(eq(self.params[2], (i-1)), p, res);
       od;

       return res.terminate();
    end,

    transpose := self >> TPrmMulti(List(self.params[1], i->i.transpose()), self.params[2]).withTags(self.getTags()),

    isReal := True,
    doNotMeasure := true,
    noCodelet := true,
    normalizedArithCost := self >> 0
));



#F PermutationModMatrix ( <perm>, <n>, <m> )
#F   constructs for the permutation <perm> on <n> points an
#F   <m> x <m> matrix M. The (i,j) entry of M contains the number
#F   of pairs (k, k^<perm>) with k mod m = i and k^{perm> mod m = j.
#F   <m> has to divide <n>.
#F   We view <perm> as a permutation of the points {0,.., n-1} even
#F   though in gap it is represented as permutation on {1,..,n} 
#F   (i.e shifted by 1).
#F

PermutationModMatrix := function ( p, n, m)
  local M, i;

  # error checking
  if not IsPerm(p) then
    Error("<p> is not a permutation");
  fi;
  if not ( IsInt(n) and IsInt(m) ) then
    Error("<n> and <m> must be integers");
  fi;
  if not n mod m = 0 then
    Error("<m> must divide <n>");
  fi;
  if not ( p = () or LargestMovedPointPerm(p) <= n ) then
    Error("<n> is not a valid degree for <perm>");
  fi;

  # initialize m x m matrix M
  M := List([1..m], r -> List([1..m], c -> 0));

  # construct M
  for i in [1..n] do
    M[(i-1) mod m + 1][(i^p-1) mod m + 1] := 
      M[(i-1) mod m + 1][(i^p-1) mod m + 1] + 1;
  od;

  return M;
end;

## ============================================================
##
##  PermMatrixRowVals(p) takes permutation matrix p and returns a list
##    of the '1' location in each row of the matrix.
##
##  Thus, the value a in list element b means that the input in 
##    location b will be permuted to location a in the output.
##
## ============================================================
PermMatrixRowVals := meth(x)




   return List(Prm(FPerm(x)).func.lambda().tolist(), i->i.ev());
end;

## ============================================================
##
##  PermMatrixRowVals(p) takes permutation matrix p and returns a list
##    of the '1' location in each row of the matrix.
##
##  Thus, the value a in list element b means that the input in 
##    location b will be permuted to location a in the output.
##
## ============================================================
PermMatrixRowValsOld := meth(x)
   local length, i, res;

   res := [];
   length := Length(x);
   for i in [1 .. length] do
      res[i] := BasisVecToNum(x[i]);
   od;
   return res;
end;


## ============================================================
##
##  min_list(l) returns the smallest value in l, a list of lists.
##
## ============================================================
min_list := meth(l)
   local t, i;   
   t := [];   
   for i in [1 .. Length(l)] do      
      t[i] := Minimum(l[i]);   
   od;   
   return Minimum(t);
end;


## ============================================================
##
##  max_list(l) returns the largest value in l, a list of lists.
##
## ============================================================
max_list := meth(l)   
   local t, i;
   t := [];
   for i in [1 .. Length(l)] do
      t[i] := Maximum(l[i]);
   od;
   return Maximum(t);
end;


## ============================================================
##
##  RandomPerm(n) returns a random permutation on n points.
##
##  RandomPermSPL(n) is not used becuase 1/3 of the time it
##  returns a stride permutation, and 1/3 of the time it 
##  returns J(n).  
##
## ============================================================
RandomPerm := meth(n)
   return Perm(Random(SymmetricGroup(n)), n);
end;

## ============================================================
##
##   IsPermMult(p) returns 1 if matrix p is a permutation of a
##   constant multiple of a permutation, and 0 otherwise.
##
## ============================================================

IsPermMult := meth(p)
   local n, i, j, total;
   n := Length(p);
   for i in [1..n] do
      total := 0;
      for j in [1..n] do
         if (p[i][j] <> 0) then
            total := total+1;
	    if (total > 1) then
		return 0;
	    fi;
         fi;
      od;
   od;
   return 1;
end; 


## ============================================================
##   OnesVector(n)
##   Returns a vector of 1s of length n
## ============================================================
OnesVector := meth(n)
   local res, i;
   res := [];
   for i in [1..n] do
      res[i] := 1;
   od;
   return res;
end;


## ============================================================
##  Given the mux addresses, i.e., the multiplexor settings
##  that say "given an output location, where does it get its
##  input from on each cycle," determine the inverse, i.e.,
##  "given an input location, determine where its output will 
##  go for each cycle.
##
##  This function is intended as a helper function, used in
##  StreamPermGen.createCode().
## ============================================================
PermHelperFindPortAddresses := meth(muxaddr)
    local ports, i, row, j, rowres;
    ports := [];
    for i in [1..Length(muxaddr)] do
       rowres := [];
       row := muxaddr[i];       
       for j in [1..Length(row)] do
          rowres[row[j]+1] := j-1;
       od;
       Append(ports, [rowres]);
    od;
    return ports;
end;


_optsInColumn := meth(col, rows_left)
    local res, i, size;
    size := Length(col);
    res := [];
    for i in [1..size] do
        if ((rows_left[i] = 1) and (col[i] > 0)) then
            Append(res, [i]);
        fi;
    od;
    return res;
end;

# Used in debugging.
#_findPermBacktracks := 0;

try_column := meth(whichCol, board, rows_left)
    local size, col, options, i, spec_rows_left, tres;
    

    size := Length(board[1]);

    col := TransposedMat(board)[whichCol];

    options := _optsInColumn(col, rows_left);

    for i in options do
        if (whichCol = size) then
            return [i];
        fi;

        rows_left[i] := 0;
        tres := try_column(whichCol+1, board, rows_left);
        if (tres <> [-1]) then
            Append(tres, [i]); # backwards
            return tres; 
        fi;
        rows_left[i] := 1;
    od;

    # Debugging
    #_findPermBacktracks := _findPermBacktracks+1;
    return [-1];    

end;

_randomPermGood := meth(board)
    local rows_left, permVec, size, permMat, col, row, choices, failed, choice, i;
    size      := Length(board[1]);
    rows_left := List([1..size], i->1);
    permVec := [];

    failed := false;
    col := 1;
    while ((col <= size) and (not failed)) do        

        # Collect list of choices in this column.
        choices := [];        
        for row in [1..size] do
            if ((rows_left[row] <> 0) and (board[row][col] > 0)) then
                Append(choices, [row]);
            fi;
        od;
        
        if (Length(choices) = 0) then
            failed := true;
        else
            choice := Random(choices);
            rows_left[choice] := 0;
            Append(permVec, [choice]);
        fi;

        col := col + 1;
    od;
    
    if (failed) then
        # If we fail, then we can assume that the square contains at least one 0 entry.  So,
        # we can send back the all 1 matrix, and be guaranteed that P - [1] < 0 for some value.
        return(List([1..size], r -> List([1..size], c -> 1)));
    else
        permMat   := [];
        for i in [1..size] do
            permMat[i] := BasisVec(size, permVec[i]-1);
        od;
        
        permMat := TransposedMat(permMat);

        return permMat;
    fi;
end;



_findPerm := meth(board)
    local rows_left, permVec, size, permMat, i;
    size      := Length(board[1]);
    rows_left := List([1..size], i->1);
    permVec   := Reversed(try_column(1, board, rows_left));
    permMat   := [];
    
    for i in [1..size] do
        permMat[i] := BasisVec(size, permVec[i]-1);
    od;

    permMat := TransposedMat(permMat);

    return permMat;
end;

_minisat_path := "";

_findPermUsingSat := meth(board)
    local size, r, c, zeros, dir, i, zerofile, outfile, res, permmat;
    
    size := Length(board[1]);
    zeros := [];

    for r in [1..size] do
       for c in [1..size] do
           if (board[r][c] = 0) then
               Append(zeros, [ (r-1)*size + c ]);
           fi;
       od;
    od;
    
    dir := Concat("/tmp/spiral/", String(GetPid()));
    MakeDir(dir);
    zerofile := Concat(dir, "/zeros");
    outfile := Concat(dir, "/out");
    PrintTo(zerofile, "");

    for i in [1..Length(zeros)] do
        AppendTo(zerofile, zeros[i], "\n");
    od;

    if (_minisat_path = "") then
        Error("Error: path to SAT solver not set: paradigms.stream._minisat_path is not bound.");
    fi;

#    IntExec(Concat("/Users/pam/minisat/core/minisat ", String(size), " ", zerofile, " ", outfile));
    IntExec(Concat(_minisat_path, " ", String(size), " ", zerofile, " ", outfile));

    res := ReadVal(outfile);
    
    permmat := [];
    for i in [1..size] do
       permmat[i] := BasisVec(size, res[i]);
    od;

    return permmat;
end;

_SumList := meth(m)
    local res, i;
    res := 0;
    for i in [1..Length(m)] do
        res := res + m[i];
    od;
    return res;
end;

_whichMeth := 1;

Declare(routePerm);

## ============================================================
##     listofperms = FactorSemiMagicSquare(square)
##
##  Given a semi-magic square of size n, find a (non-unique) 
##  decomposition into a sum of permutations.  Returns the 
##  decomposition as a list of permutations (with each perm
##  being represented in the "row value" form).
##
##  Each semi-magic square can be decomposed into at most
##  (n-1)^2 + 1 different permutations.  See:
##      Leep et al. Marriage, Magic, and Solitaire. The American 
##      Mathematical Monthly (1999) vol. 106 (5) pp. 419-429.
##  
##  This method uses randomly generated permutations.  A better
##  method would construct them based upon the square.  A more
##  realistic approach would be a hybrid of the two.
##
##  This function is intended as a helper function, used in
##  StreamPermGen.createCode().
## ============================================================
FactorSemiMagicSquare := meth(square)
   local length, multplr, rndprm, j, res, rows_left, cols_left, startTime, endTime, tries, which, ct;
   length := Length(square);
   res := [];

   # Debugging
   startTime := TimeInSecs();
   tries := 0;

   ct := 0;

   which := stream._whichMeth;

   ##  While the semi-magic square is not all zero...
   while (max_list(square) > 0) do

	  ## What is left?  Is it a multiple of a permutation matrix?
	  ## (This step not needed for correctness, but is a performance
	  ## optimization.)
	  if (IsPermMult(square) = 1) then

	      ## If what is left is a constant times a perm matrix,
	      ## set that perm as the random step, and complete the 
	      ## final step.
	      rndprm := square / max_list(square);
	      multplr := 0;
	      while (min_list(square - multplr*rndprm) >= 0) do
	         multplr := multplr+1;
	      od;
	      multplr := multplr-1;
	      square := square - multplr*rndprm;		 
	      for j in [1..multplr] do
#! fails sometimes	         Append(res, [routePerm(PermMatrixRowValsOld(rndprm))]);  
	         Append(res, [PermMatrixRowValsOld(rndprm)]);
	      od;

          ct := ct + 1;

      else
          
          ##  Generate a random permutation.
          # set 'which' as: paradigms.stream.whichMeth := 0;

#          rndprm := _randomPermGood(square);
          tries := tries+1;

#            if (which = 0) then
#                rndprm := MatSPL(RandomPerm(length));
#                #tries := tries + 1;
#            else
#                if (which = 1) then
#                    rndprm := _findPerm(square);
#                else
                    rndprm := _findPermUsingSat(square);
#                fi;
# 	       fi;
      fi;


      ##  How many times can this permutation be 
      ##  subtracted from the square?
      multplr := 0;
      while (min_list(square - multplr*rndprm) >= 0) do
         multplr := multplr+1;
      od;
      multplr := multplr-1;

      ## If this permutation has been subtracted from the square,
      ## record the decision, and check if we are done.
      if (multplr > 0) then

          ## Subtract the permutation the correct number of times.
          square := square - multplr*rndprm;

     	  ## Record the decision
	      for j in [1..multplr] do
	          Append(res, [PermMatrixRowValsOld(rndprm)]);
	      od;
          
          endTime := TimeInSecs();
#          Print(which, ", ", (endTime - startTime), ", ", tries, "\n");
          startTime := endTime;
          tries := 0;
          ct := ct + 1;

      fi;
   od;

   return res;
end;


## ============================================================
##  row = PermHelperFindRow(permrows, x (readdport), 
##                           y (writeport), streaming-width)
##
##  Given a permutation as a set of row values, and a selected
##  read and write port (x, y), find which permutation word can 
##  be scheduled that will lead to a read from port x to a 
##  write in port y.
##
##  As the algorithm progresses, words that have been scheduled
##  are removed from permrows and replaced with -1.
##
##  This function is intended as a helper function, used in
##  StreamPermGen.createCode().
## ============================================================
PermHelperFindRow := meth(p, x, y, w)
   local a, b;

   for b in [1 .. Length(p)] do
      a := p[b];
      
      # The a <> -1 isn't needed for correctness but it makes the
      # code more clear.
      if ((a <> -1) and (Mod(a,w) = x) and (Mod(b-1,w) = y)) then
         return b;
      fi;
   od;

   Error("Cannot find a word to schedule.");
end;


## ============================================================
##  [rd_addr, wr_addr] = PermHelperFindAddresses(permrows, 
##                          muxaddr, w);
##
##  Given a permutation as a set of row values and a cycle-by-
##  cycle read/write port mapping (muxaddr), and streaming width 
##  w, determine the assoicated memory read and write addresses.
##
##  This function is intended as a helper function, used in
##  StreamPermGen.createCode().
## ============================================================
PermHelperFindAddresses := meth(permrows, muxaddr, w)
   local rd_addr, wr_addr, i, x, y, b, a, rdcyc, wrcyc, wrcyc2, ports;

   ports := PermHelperFindPortAddresses(muxaddr);
   rd_addr := []; wr_addr := [];

   for i in [1..Length(permrows)/w] do
      rdcyc := []; wrcyc := []; wrcyc2 := [];

      for x in [1..w] do
         y := ports[i][x]+1;
	 b := PermHelperFindRow(permrows, x-1, y-1, w);
	 a := permrows[b];
	 Append(rdcyc, [floor(a/w).v]);
	 Append(wrcyc, [floor((b-1)/w).v]);
	 Append(wrcyc2, []);
	 permrows[b] := -1;
      od;
      Append(rd_addr, [rdcyc]);

      # Here, wrcyc tells the write address for each *read port* at this
      # time.  I need to permute these values so it tells the write address
      # for each *write port* at this time.
      # So, I just need to assign wrcyc to wrcyc2 based upon this mapping.
      for x in [1..w] do 
         wrcyc2[x] := wrcyc[muxaddr[i][x]+1]; 
      od;

      Append(wr_addr, [wrcyc2]);
   od;
   return [rd_addr, wr_addr];
end;
      


## ==================================================================
##  BRAMPermGeneral([rdaddr], [swnetctrl], [wraddr], [perm], it)
##    where rdaddr is the set of read addresses, swnetctrl is the 
##    control bits for the \Omega^{-1}\Omega network, wraddr is the 
##    set of write addresses, and perm is the permutation as an SPL 
##    object.  it is the iteration variable that specifies which perm 
##    to perform.
##
##  This represents a streaming multi-permutation structure that has 
##  been factored using our "general" method.
##   
##  This is not ideal for permutations that are linear on the bits.  
##
##  Do not attempt to build this object by hand. Instead use 
##  StreamPerm(perm, streamsize) to build a generic streaming 
##  perm.  Then, StreamPerm.createCode() will construct the
##  BRAMPermGeneral given the permutation and streaming width.
## ==================================================================
Class(BRAMPermGeneral, BaseMat, SumsBase, rec(
	abbrevs := [(r,m,w,p,it)-> [r,m,w,p,it]], 

	new := (self, r, m, w, p, it) >> SPL(WithBases(self, rec(
		    _children  := [r, m, w, p, it],
		    dimensions := [ Length(r[1])*Length(r[1][1]), Length(r[1])*Length(r[1][1])],
		    streamSize := Length(r[1][1]),
	     ))),

	rChildren := self >> self._children,
	rSetChild := meth(self, n, what) self._children[n] := what; end,
	child     := (self, n) >> self._children[n],
	children  := self >> self._children,

	createCode := self >> self,
    print      := (self,i,is) >> Print(self.name, "(", self.dims()[1], 
	    ", ", self.streamSize, ", ", self.child(1), ", ", self.child(2), 
	    ", ", self.child(3), ", ", self.child(5), ")"),


    toAMat := meth(self)
        local prms, len, it, i, res;
        prms := self.child(4);
        len  := Length(prms);
        it   := self.child(5);
        res  := prms[len];

        for i in Reversed([1..len-1]) do
            res := COND(eq(it, i-1), prms[i], res);
        od;
        
        return res.toAMat();
    end,

	dims   := self >> [Length(self._children[1][1])*Length(self._children[1][1][1]), 
	                   Length(self._children[1][1])*Length(self._children[1][1][1])],

	sums   := self >> self
));


PadToTwoPowSize := meth(muxaddr)
   local n, n2p, i, res, dif, line;
   n := Length(muxaddr[1]);
   
   if (2^Log2Int(n) = n) then
       return muxaddr;
   fi;

   n2p := 2^(Log2Int(n)+1);
   dif := n2p-n;


   res := [];

   for i in [1..Length(muxaddr)] do
       line := Concatenation(muxaddr[i], [n..n2p-1]);       
       Append(res, [line]);
   od;

   return res;
end;


RandomVectorVector := meth(domain, inner, outer)
   local res, i, j;
   res := [];
   for i in [1..outer] do
      res[i] := [];
      for j in [1..inner] do
         res[i][j] := RandomList([0..domain-1]);
      od;
   od;
   return res;
end;

## ======================================================
##  A container for a streaming multi-permutation structure
##
##  StreamPerm([<SPL>], <m>, <w>, <i>)
##
##  where the permutations are I(<M>) x <SPL>, <SPL> is 
##  the permutation as an SPL object, <w> is the 
##  streaming width, and <i> selects between the <SPL>.
## ======================================================

Class(StreamPerm, BaseMat, SumsBase, rec(
	abbrevs := [(s, m, w, i) -> [s, m, w, i]], 

	new := (self, prm, m, ss, i) >> SPL(WithBases(self, rec(
	     _children := [prm, m, ss, i],
	     streamSize := ss, 
         par := m,
         it := i,
	     dimensions := Cond(IsBound(prm[1].domain), [m * prm[1].domain(), m * prm[1].range()], m * prm[1].dims()),
	     ))),

	rChildren := self >> self._children,
	rSetChild := meth(self, n, what) self._children[n] := what; end,
	child := (self, n) >> self._children[n],
	children := self >> self._children,
    
    createCode := meth(self)
        local lin, prm, w, P, length, permobj, permrows, modmatrix, muxaddr, muxaddr2, ports, addrs, 
              rdaddr, wraddr, network, res, p3, p1, r, k, M, N, P, x, i, j, l, K, size, count, ptrans, 
              z, G, q1, q3, q3b, p1b, G, dm, plist, bitlist, prms, Pt, thisPerm, thisPrm, prmCode, it,
              prmTens, dm_no_tens, prm_no_tens;

        prm := List(self.child(1), p->Tensor(I(self.child(2)), p));
        prm_no_tens := self.child(1);
        w := self.child(3);
        it := self.child(4);

        P := [];

        # If the permutation is a composition of multiple perms, it is more efficient to
        # compute the bit representation of each individually, and then combine them.
        for i in [1..Length(self.child(1))] do
            thisPerm := self.child(1)[i];
            thisPrm := prm[i];

	    if (ObjId(thisPrm) = L) then
		thisPrm := TL(thisPerm.params[1], thisPerm.params[2]);
	    fi;

	    if (ObjId(thisPrm) = Tensor and ObjId(thisPrm.child(1)) = I and ObjId(thisPrm.child(2)) = L) then
		thisPrm := TL(thisPrm.child(2).params[1], thisPrm.child(2).params[2], thisPrm.child(1).params[1], 1);
	    fi;

            if (ObjId(thisPerm) = Compose) then
                plist := thisPerm.children();

		# If we just have an L, turn it into a TL so we can use its .permBits() function
		plist := List(plist, i -> Cond(ObjId(i) = L, TL(i.params[1], i.params[2]), i));

		# If we have I x L, turn it into TL
		plist := List(plist, i -> Cond(ObjId(i) = Tensor and ObjId(i.child(1)) = I and ObjId(i.child(2)) = L, TL(i.child(2).params[1], i.child(2).params[2], i.child(1).params[1], 1), i));

                bitlist := List(plist, i -> 
		    Cond(IsBound(i.permBits), 
			i.permBits(), 
			Cond(ObjId(i) = Tensor and ObjId(i.child(1)) = I and IsBound(i.child(2).permBits), 
				DirectSumMat(MatSPL(I(Log2Int(i.child(1).params[1])))*GF(2).one, i.child(2).permBits()),
				PermMatrixToBits(MatSPL(i))
				
			)
		    )
		);

                if (-1 in bitlist) then
                    Pt := -1;
                else
                    Pt := Product(bitlist);
                    if (self.child(2) > 1) then
                        Pt := DirectSumMat(MatSPL(I(Log2Int(self.child(2))))*GF(2).one, Pt);
                    fi;
                fi;
            else
		Pt := Cond(IsBound(thisPrm.permBits),
                    # If P.permBits defined, use it
		    thisPrm.permBits(),  
	            # If P = Tensor(I, Q) and Q.permBits defined, use it. 
		    Cond(ObjId(thisPrm) = Tensor and ObjId(thisPrm.child(1)) = I and IsBound(thisPrm.child(2).permBits),
			DirectSumMat(MatSPL(I(Log2Int(thisPrm.child(1).params[1])))*GF(2).one, thisPrm.child(2).permBits()),

			# Otherise, use PermMatrixToBits function (slow)
			PermMatrixToBits(MatSPL(thisPrm))
		    )
		);
            fi;

            Append(P, [Pt]);
        od;

        # so, Make a list of Ps, one for each perm.  then set lin if all are <> -1.
        lin := Cond(ForAny(P, i->i=-1) or 2^Log2Int(w) <> w or Length(prm) > 1, 0, 1);

        # Find the overall dimensions and the dimension of the "non-tensor" part       
        dm := Cond(IsBound(prm[1].domain), prm[1].domain(), prm[1].dims()[1]);

	# PM: This is not very robust.
        dm_no_tens := Cond(IsBound(self.child(1)[1].domain), self.child(1)[1].domain(), self.child(1)[1].dims()[1]);


        # If the permutations are on w words, then we can generate code.
        if (self.child(3) = dm_no_tens) then
            
            ## If this perm is multiple permutation functions composed, separate them to make the
            ## code generation work as expected.
            
            prmCode := List(prm_no_tens, j ->
                Cond(ObjId(j) = Compose,
                    Compose(List(j.children(), i->Prm(FPerm(i)))),
                    Prm(FPerm(j))
                    )
                );
            
            res := prmCode[Length(prmCode)];
            
            for i in Reversed([1..Length(prmCode)-1]) do
            res := COND(eq(it, (i-1)), prmCode[i], res);
            od;
            
            if (self.child(2) > 1) then                
                return STensor(CodeBlock(res).createCode(), self.child(2), self.child(3));
            fi;

            return CodeBlock(res).createCode();
            
        fi;

        
        # If our perm is not linear on the bits, or if we have multiple permutations, or
        # if we have manually overridden this to avoid the patented permutations:
        if ((lin = 0) or (stream.avoidPermPatent = 1)) then


            # We want to ignore the tensor product until the end
            prm := self.child(1);
            prmTens := List(self.child(1), p->Tensor(I(self.child(2)), p));

            it := self.child(4);

            if (self.child(3) = 1) then
                return BRAMPermStreamOne(
#                           List(prmTens, i->PermMatrixRowValsOld(MatSPL(i))),
                           List(prmTens, i->PermMatrixRowVals(i)),
                           prmTens,
                           it);
            fi;


	        length    := self._children[1][1].dims()[1];

	        permobj   := List(self._children[1], i -> PermSPL(i));
	        permrows  := List(permobj, i -> PermMatrixRowValsOld(MatPerm(i, length)));
	        modmatrix := List(permobj, i -> PermutationModMatrix(i, length, self.child(3)));
	        muxaddr   := List(modmatrix, i -> FactorSemiMagicSquare(i));
            
            muxaddr2  := List(muxaddr, i -> PadToTwoPowSize(i));


            network   := List(muxaddr2, j -> List([1..Length(j)], i -> routePerm(j[i])));
            
	        addrs     := List([1..Length(permobj)], i -> PermHelperFindAddresses(permrows[i], muxaddr[i], self.child(3)));
	        rdaddr    := List(addrs, i -> i[1]);
	        wraddr    := List(addrs, i -> i[2]);

            if (self.child(2) > 1) then
                return STensor(BRAMPermGeneral(rdaddr, network, wraddr, self._children[1], it), self.child(2), w);
            else
                return BRAMPermGeneral(rdaddr, network, wraddr, self._children[1], it);
            fi;
            
        else
            # If we are here, we only have one permutation
            P := P[1];
           

	        k := Log2Int(self.child(3));
	        size := Log2Int(self.dimensions[1]);
            
            # If our streaming width is 1, then we are done.  P = I*P.
	        if (self.child(3) = 1) then
                return BRAMPerm(MatSPL(I(size))*GF(2).one, P, 1);
	        fi;

	        # if P is the bit form of I tensor P', we can remove the top-left corner bits
	        # (up to (size-k) bits, and replace them with a streaming tensor product of a 
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

	        else # This is algorithm 5.2 of J. ACM 2009 paper.

                     # Is P a permutation matrix?
                     if (CheckPerm(P) = 1) then	
	                  j := NonZeroRows(p3);
	                  i := ZeroRows(p1);
	             else
		          p1b := TransposedMat(Copy(p1));
		          G := MyTriangulizeMat(p1b);
		          G := TransposedMat(G);
		          G := G * GF(2).one;
		          q1 := p1 * G;
		          q3 := p3 * G;
		            
 		          z := LinearIndependentColumns(TransposedMat(q1));
 		          i := [];
                    
 		          for l in [1 .. k] do
   		               if l in z then
 		                    ;
                               else
 		                    Add(i, l);
 		               fi;
 		          od;
               
 		          q3b := [];
		       
 		          for l in [1 .. Length(q3)] do
 		               Append(q3b, [q3[l]{[r+1..k]}]);
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
	            return BRAMPerm(N, M, self.child(3));
	        else
	            return STensor(BRAMPerm(N,M,self.child(3)), 2^count, self.child(3));
	        fi;            
        fi;
    end,

    print := (self,i,is) >> Print(self.name, "(", self._children[1], ", ", self.child(2), ", ", self.child(3), ", ", self._children[4], ")"),

    toAMat := meth(self)
        local prms, len, it, i, res;
        prms := self._children[1];
        len  := Length(prms);
        it   := self._children[4];
        res  := Tensor(I(self.child(2)), prms[len]);
        

        for i in Reversed([1..len-1]) do
            res := COND(eq(it, (i-1)), Tensor(I(self.child(2)), prms[i]), res);
        od;

        return res.toAMat();
    end,

	dims := self >> self.child(2) * self._children[1][1].dims(),
	sums := self >> self
        
));



## ======================================================
##  A container for a streaming RC(permutation)
##
##  RCStreamPerm([<SPL>], <m>, <w>, <i>)
##
##  where the permutations is I(<M>) x [<SPL>] x I(2), [<SPL>] is 
##  list of permutations as SPL objects, <w> is the 
##  streaming width, and <i> selects which of the perms.
## ======================================================
Class(RCStreamPerm, BaseMat, SumsBase, rec(
	abbrevs := [(s, m, w, i) -> [s, m, w, i]], 

	new := (self, prm, m, ss, i) >> SPL(WithBases(self, rec(
	     _children := [prm, m, ss, i],
	     streamSize := ss, 
         par := m,
         it := i,
	     dimensions := Cond(IsBound(prm[1].domain), [2 * m * prm[1].domain(), 2 * m * prm[1].range()], 2 * m * prm[1].dims()),
	     ))),

	rChildren := self >> self._children,
	rSetChild := meth(self, n, what) self._children[n] := what; end,
	child := (self, n) >> self._children[n],
	children := self >> self._children,
    
    createCode := meth(self)
        local r;
        r := StreamPerm(self.child(1), self.par, self.streamSize, self.it).createCode(); 
        if (ObjId(r) = STensor) then
            return STensor(RC(r.child(1)), r.p, r.bs*2);
        else
             return RC(r);
        fi;
    end,

    print := (self,i,is) >> Print(self.name, "(", self._children[1], ", ", self.par, ", ", self.streamSize, ", ", self._children[4], ")"),

	toAMat := self >> RC(StreamPerm(self._children[1], self.par, self.streamSize, self.it)).toAMat(),

	dims := self >> 2 * self.par * self._children[1][1].dims(),
	sums := self >> self

        
));



## ======================================================
##  A container for a general streaming permutation (not 
##  necessarily linear on the bits)
##
##  StreamPermGen(<SPL>, <q>)
##
##  where <SPL> is the permutation as an SPL object, and 
##  <q> is the streaming width.
## ======================================================
## NB: Deprecated.  Use StreamPerm, which covers all cases
# Class(StreamPermGen, BaseMat, SumsBase, rec(
# 	abbrevs := [(s, q) -> [s, q]], 

# 	new := (self, prm, ss) >> SPL(WithBases(self, rec(
# 	     _children := [prm],
# 	     streamSize := ss, 


# 	     dimensions := Cond(IsBound(prm.domain), [prm.domain(), prm.range()], prm.dims()),

# 	     ))),

# 	rChildren := self >> self._children,
# 	rSetChild := meth(self, n, what) self._children[n] := what; end,
# 	child := (self, n) >> self._children[1],
# 	children := self >> self._children,

# 	createCode := meth(self) 
# 	   local length, permobj, permrows, modmatrix, muxaddr, muxaddr2, ports, 
#              addrs, rdaddr, wraddr, network, dms, prms;
             
#        dms := Cond(IsBound(self.child(1).domain), [self.child(1).domain(), self.child(1).range()], self.child(1).dims());

#        if (self.streamSize = 1) then
#            return BRAMPermStreamOne(
#                     [PermMatrixRowVals(MatSPL(self._children[1]))],
#                     [self._children[1]], 0);
#        fi;

#        if (self.streamSize = dms[1]) then

#            ## If this perm is multiple permutation functions composed, separate them to make the
#            ## code generation work as expected.
#            if (ObjId(self.child(1)) = Compose) then
#                prms := Compose(List(self.child(1).children(), i->Prm(FPerm(i))));
#            else
#                prms := Prm(FPerm(self.child(1)));
#            fi;
           
#            return CodeBlock(prms).createCode();

#        fi;

# 	   length    := self._children[1].dims()[1];
# 	   permobj   := PermSPL(self._children[1]);
# 	   permrows  := PermMatrixRowVals(MatPerm(permobj, length));
# 	   modmatrix := PermutationModMatrix(permobj, length, self.streamSize);
# 	   muxaddr   := FactorSemiMagicSquare(modmatrix);

#        muxaddr2  := PadToTwoPowSize(muxaddr);

# #       Print("muxaddr = ", muxaddr, "\nmuxaddr2 = ", muxaddr2);

#        network   := List([1..Length(muxaddr2)], i -> routePerm(muxaddr2[i]));

# 	   addrs     := PermHelperFindAddresses(permrows, muxaddr, 
# 	                   self.streamSize);
# 	   rdaddr    := addrs[1];
# 	   wraddr    := addrs[2];

# #!	   rdaddr := RandomVectorVector(length/self.streamSize, self.streamSize, length/self.streamSize);
# #!	   wraddr := RandomVectorVector(length/self.streamSize, self.streamSize, length/self.streamSize);
# #!	   muxaddr := RandomVectorVector(self.streamSize, self.streamSize, length/self.streamSize);

# # Previously, we used 'muxaddr.'
# #	   return BRAMPermGeneral(rdaddr, muxaddr, wraddr, self._children[1]);
# # Now I am switching to the omega^-1 omega network control bits
#        return BRAMPermGeneral(rdaddr, network, wraddr, self._children[1]);

#         end,
        
# 	print := (self,i,is) >> Print(self.name, "(", self._children[1], ", ", 
# 	                               self.streamSize, ")"),

# 	toAMat := self >> self._children[1].toAMat(),
# 	dims := self >> self._children[1].dims(),
# 	sums := self >> self

# ));




## ======================================================
##  A container for a streaming multi-permutation structure
##
##  StreamPermGen(<l>, <i>, <q>)
##
##  where <l> is a list of permutations as SPL objects, 
##  <i> is a variable that selects which permutation to 
##  perform, and <q> is the streaming width.
## ======================================================
## Deprecated.  Use StreamPerm, which supports multiple perms (or will soon!)

##  !!!!!!!!!!! IMPORTANT !!!!!!!!
#     Do not delete this code until I have successfully rolled its multiple perm
#     support into StreamPermGen.
##  !!!!!!!!!!! IMPORTANT !!!!!!!!

# Class(StreamPermGenMult, BaseMat, SumsBase, rec(
# 	abbrevs := [(s, i, q) -> [s, i, q]], 

# 	new := (self, prm, i, ss) >> SPL(WithBases(self, rec(
# 	     _children := [prm, i],
# 	     streamSize := ss, 

# 	     dimensions := Cond(IsBound(prm[1].domain), [prm[1].domain(), prm[1].range()], prm[1].dims()),

# 	     ))),

# 	rChildren := self >> self._children,
# 	rSetChild := meth(self, n, what) self._children[n] := what; end,
# 	child := (self, n) >> self._children[n],
# 	children := self >> self._children,

# 	createCode := meth(self) 
# 	   local length, permobj, permrows, modmatrix, muxaddr, muxaddr2, ports, 
#              addrs, rdaddr, wraddr, network, dms, prms, it, prmcode, i, res;
             
#        dms := Cond(IsBound(self.child(1)[1].domain), [self.child(1)[1].domain(), self.child(1)[1].range()], self.child(1)[1].dims());

#        it   := self.child(2);
#        prms := self.child(1);

#        if (self.streamSize = 1) then
#            return BRAMPermStreamOne(List(prms, i->PermMatrixRowVals(MatSPL(i))),
#                       prms,
#                       it);
#        fi;

#        if (self.streamSize = dms[1]) then

#            ## If this perm is multiple permutation functions composed, separate them to make the
#            ## code generation work as expected.

#            prmcode := List(self.child(1), j -> 
#                         Cond(ObjId(j)=Compose,
#                            Compose(List(self.child(1).children(), i->Prm(FPerm(i)))),
#                            Prm(FPerm(j))
#                         )
#            );

#            res := prmcode[Length(prmcode)];

#            for i in Reversed([1..Length(prmcode)-1]) do
#               res := COND(eq(it, i), prmcode[i], res);
#            od;
           
#            return CodeBlock(res).createCode();

#        fi;

# 	   length    := self._children[1].dims()[1];
# 	   permobj   := PermSPL(self._children[1]);
# 	   permrows  := PermMatrixRowVals(MatPerm(permobj, length));
# 	   modmatrix := PermutationModMatrix(permobj, length, self.streamSize);
# 	   muxaddr   := FactorSemiMagicSquare(modmatrix);

#        muxaddr2  := PadToTwoPowSize(muxaddr);

#        network   := List([1..Length(muxaddr2)], i -> routePerm(muxaddr2[i]));

# 	   addrs     := PermHelperFindAddresses(permrows, muxaddr, 
# 	                   self.streamSize);
# 	   rdaddr    := addrs[1];
# 	   wraddr    := addrs[2];


# # Previously, we used 'muxaddr.'
# #	   return BRAMPermGeneral(rdaddr, muxaddr, wraddr, self._children[1]);
# # Now I am switching to the omega^-1 omega network control bits
#        return BRAMPermGeneral(rdaddr, network, wraddr, self._children[1]);

#         end,
        
# 	print := (self,i,is) >> Print(self.name, "(", self._children[1], ", ", self._children[2], ", ", 
# 	                               self.streamSize, ")"),

#     toAMat := meth(self)
#         local prms, len, it, i, res;

#          prms := self._children[1];
#          len  := Length(prms);
#          it   := self._children[2];
#          res  := prms[len];

#          for i in Reversed([1..len-1]) do
#              res := COND(eq(it, i), prms[i], res);
#          od;

#          return res.toAMat();
#     end,

# #    toAMat := self >> COND(eq(self._children[2], 0), self._children[1][1], self._children[1][2]).toAMat(),
# 	dims := self >> self._children[1][1].dims(),
# 	sums := self >> self

# ));


## ==================================================================
##   BRAMPermStreamOne([rd-addr, ...], [<SPL>, ... ], i)
##     A container for a streaming multi-permutation structure where width = 1.
##     This merits its own class because the hardware implementation
##     is greatly simplified.
## 
##     The permutations are given as a list of vectors of read addresses, and
##     as a list of SPL objects.
##
##     i is the variable that selects which permutation to perform
## ==================================================================
Class(BRAMPermStreamOne, BaseMat, SumsBase, rec(
	abbrevs := [(r, p, i)-> [r, p, i]], 

	new := (self, r, p, i) >> SPL(WithBases(self, rec(
		    _children  := [r, p, i],
		    dimensions := [ Length(r[1]), Length(r[1])]
	     ))),

	rChildren := self >> self._children,
	rSetChild := meth(self, n, what) self._children[n] := what; end,
	child     := (self, n) >> self._children[n],
	children  := self >> self._children,

	createCode := self >> self,
    
    print      := (self,i,is) >> Print(self.name, "(", self.child(1), ", ", self.child(3), ")"), 

#	toAMat := self >> self.child(2).toAMat(),

    toAMat := meth(self)
        local prms, len, it, i, res;

         prms := self._children[2];
         len  := Length(prms);
         it   := self._children[3];
         res  := prms[len];

         for i in Reversed([1..len-1]) do
             res := COND(eq(it, i-1), prms[i], res);
         od;

         return res.toAMat();
    end,


	dims   := self >> [Length(self._children[1][1]), 
	                   Length(self._children[1][1])],

	sums   := self >> self
));

Declare(InitStreamUnrollHw, SumStreamStrategy, StreamStrategy);


# CountPermSwitches(t)
# Returns number of switches needed for permutation t.
# We assume t is a TPrm().withTags([AStream(w)]);
CountPermSwitches := function (t)
  local i, M, M2, M2_t, N, N2, N2_t, n, k, r, s, s2, s3, s_m, s_n, opts;

  opts := InitStreamUnrollHw();
  r := RandomRuleTree(t, opts);
  s := SumsRuleTreeStrategy(r, SumStreamStrategy, opts);
  s2 := ApplyStrategy(s, StreamStrategy, UntilDone, opts);
  s3 := s2.createCode();

  k := Log2Int(s3.streamSize);
  M := s3.child(2);
  n := Length(M);

  # Extract M2.  First just get the bottom k rows.
  M2_t := Sublist(M, [n-k+1 .. n]);
  
  M2 := [];
  # Now get the first n-k columns
  for i in [1..k] do
     Append(M2, [Sublist(M2_t[i], [1..k])]);
  od;

  s_m := Rank(M2);

  N := s3.child(1);
  # Extract N2.  First just get the bottom k rows.
  N2_t := Sublist(N, [n-k+1 .. n]);
  
  N2 := [];
  # Now get the first n-k columns
  for i in [1..k] do
     Append(N2, [Sublist(N2_t[i], [1..k])]);
  od;

  s_n := Rank(N2);

  return (s_m + s_n) * s3.streamSize/2;
end;
