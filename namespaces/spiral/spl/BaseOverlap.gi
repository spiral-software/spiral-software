
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(BaseOverlap);

# BaseOverlap
# subclasses should define these fields:
#    _transpose_class
#
Class(BaseOverlap, BaseOperation, rec(

    new := (self, isize, overlap, spls) >>
       self._new(isize,overlap,spls),

    _new := meth(self, isize, overlap, spls)
        local result;
    if not IsList(spls) then spls := [spls]; fi;
    #Constraint(IsInt(overlap));
    #Constraint(IsInt(isize));
        DoForAll(spls, s -> Constraint(IsSPL(s)));
    result := WithBases(self, rec(isize := isize,
                            overlap := overlap,
                    _children := spls));
    result.dimensions := result.dims();
    return SPL(result);
    end,
    #-----------------------------------------------------------------------
    rChildren := self >> Concat([self.isize, self.overlap, self._children], When(IsBound(self._setDims), [self._setDims],[])),
    rSetChild := rSetChildFields("isize", "overlap", "_children", "_setDims"),

    #-----------------------------------------------------------------------
    isPermutation := self >> false,
    #-----------------------------------------------------------------------
    transpose := self >> CopyFields(self, 
	self._transpose_class._new(
            self.isize,
            self.overlap,
            List(self._children, c -> c.transpose())
	)
    ),
    conjTranspose := self >> CopyFields(self, 
	self._transpose_class._new(
            self.isize,
            self.overlap,
            List(self._children, c -> c.conjTranspose())
	)
    ),
    #----------------------------- Helpers ----------------------------------

    # Create SparseSPL list for AMatSPL conversion of type "rowDirectSum" and
    # "rowtensor" that represent a row overlapped direct sum of identity
    # matrices of sizes given in list N with overlap v.
    _rowoverlap := function(N, v)
        local i,j,k,l,min,L;
	L:=[];
	i:=1;
	j:=1;
	for k in [1..Length(N)] do
            for l in [1..N[k]] do
                Add(L, [i, j, 1]);
		i := i+1;
		j := j+1;
	    od;
            j := j-v;
        od;
        # Make sure there are no negative indices
	min := Minimum(List([1..Length(L)],i->L[i][2]));
	if min < 1 then
            for k in [1..Length(L)] do
                L[k][2] := L[k][2] + 1 - min;
            od;
	fi;
	return L;
   end,

   # Create SparseSPL list for AMatSPL conversion of type "colDirectSum" and
   # "coltensor" that represent a column overlapped direct sum of identity
   # matrices of sizes given in list N with overlap v.
   _coloverlap := function(N, v)
       local i,j,k,L;
       L := BaseOverlap._rowoverlap(N, v);
       for k in [1..Length(L)] do
           L[k]{[1..2]}:=Reversed(L[k]{[1..2]});
       od;
       return L;
   end,

   ovDim := (ov, dims) ->
       FoldL(dims,
         (p, dim) -> let(dd := p[1] + dim - ov,
                   [ dd, Min2(p[2], dd-dim+1), Max2(p[3], dd) ]),
         [ov, 1, 1]),

   ovDimList := (ov, dims) -> Drop(
       ScanL(dims,
         (p, dim) -> let(dd := p[1] + dim - ov,
                   [ dd, Min2(p[2], dd-dim+1), Max2(p[3], dd) ]),
         [ov, 1, 1]),
       1),

   spans := (ov, dims) -> let(
       dl := BaseOverlap.ovDimList(ov,dims),
       ofs := 1 - dl[1][2],
       List([1..Length(dims)], i -> [ofs+dl[i][1]-dims[i]+1, ofs+dl[i][1]]))
));
