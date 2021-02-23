
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(_numBlocks);

#F NumBlocks(<sums>, <blocksize>, <comp>)
#F
#F counts the number of blocks of <blocksize> in <sums>, and <comp> is
#F a boolean that tells the function to only count blocks that actually do
#F some computation. It checks for computation by testing the presence of the 
#F Blk() structure.

NumBlocks := (S, blocksize, comp) -> 
    _numBlocks(S, blocksize, S.dims()[1], comp);

_numBlocks := function(S, blocksize, size, comp)

    if ObjId(S) = ISum then
        size := size / S.var.range;

        if size <= blocksize then
            return When(comp = false or Collect(S, Blk) <> [], 1, 0);
        fi;
    fi;
        
    return Sum(S.rChildren(), e -> _numBlocks(e, blocksize, size, comp));
end;

Declare(_numWBlocks);

NumWeightedBlocks := (S, blocksize, comp) ->
    _numWBlocks(S, blocksize, S.dims()[1], S.dims()[1], comp);

_numWBlocks := function(S, blocksize, size, fullsize, comp)
    if ObjId(S) = ISum then
        size := size / S.var.range;

        if size <= blocksize then
            return When(comp = false or Collect(S, Blk) <> [],
                fullsize / size,
                0
            );
        fi;
    fi;

    return Sum(S.rChildren(), e -> _numWBlocks(e, blocksize, size, fullsize, comp));
end;

Declare(_numW2Blocks);

#F NumW2Blocks(<sums>, <blocksize>, <comp>)
#F
#F
#F
NumW2Blocks := (S, blocksize, comp) ->
    _numW2Blocks(S, blocksize, S.dims()[1], S.dims()[1], comp);

_numW2Blocks := function(S, blocksize, size, fullsize, comp)

    if ObjId(S) = ISum then
        size := size / S.var.range;

        # <fullsize / size> gives the number of times a block of size
        # <size> is executed.  <2 * size> is the number of out of
        # block accesses a <size> sized block requires. <(2 * size) *
        # (fullsize / size)> is the total number of out-of-block
        # accesses required by this block.
        #
        # <size * fullsize> is an algebraic simplification.
        #
        # since fullsize is a constant factor, we drop it. 
        #
        if size <= blocksize then
            return When(comp = false or Collect(S, Blk) <> [],
                size,
                0
            );
        fi;
    fi;

    return Sum(S.rChildren(), e -> _numW2Blocks(e, blocksize, size, fullsize, comp));
end;
