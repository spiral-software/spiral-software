
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

#F standard cache params structure, it takes 
#F
#F blksize (block size in bytes)
#F nsets   (number of sets)
#F assoc   (associativity eg. 2-way,4-way,etc)
#F rpolicy  (replacement policy string eg, "FIFO","LRU", etc)
#F
#F and computes a hidden value, csize (cache size)

Declare(CacheDesc);

Class(CacheDesc, rec(
    __call__ := (self, blksize, nsets, assoc, rpolicy) >>
        Checked(Is2Power(blksize),
            Is2Power(nsets),
            Is2Power(assoc),
            IsString(rpolicy),
            WithBases(self, rec(
                blksize := blksize,
                nsets := nsets,
                assoc := assoc,
                rpolicy := rpolicy,
                csize := blksize * nsets * assoc,
            ))
        ),

    operations := rec(
        Print := self >> When(ObjId(self) = CacheDesc,
            Print("CacheDesc(", 
                PrintCS([
                    self.blksize, 
                    self.nsets, 
                    self.assoc, 
                    Concat("\"", self.rpolicy, "\"")
                ]), ")"
            ),
            Print("CacheDesc(<blksize>, <nsets>, <assoc>, <rpolicy>)")
        )
    ),
));

# Cache tag. This is a generic tag, which means it takes any arguments.
# However, the correct arguments are:
#
# arg1: positive integer symbolizing cache level
# arg2: a CacheSpec()

Class(ACache, AGenericTag);

# Inplace tag. Takes one argument, maximum BB size.
#
# use this tag to specify that a transform should be computed inplace
Class(AInplace, AGenericTag);

# Right expansion tag
#
# one argument, max left kernel size
#
Class(AExpRight, AGenericTag);

# AMem tag. 
#
# params are:
# 1: size
# 2: packet/block size
# 3: per use defined "fit" function
#
# what fit-function's parameters are is TBD

Class(AMem, AGenericTag, rec(
	__call__ := meth(arg)
		local self, params;
		self := arg[1];
		params := Drop(arg, 1);
		return WithBases(self, 
			rec(
				params := params,
				size := (self) >> self.params[1], 
				blocks := (self) >> self.params[2], 
				assoc := (self) >> self.params[3],
				operations := RewritableObjectOps
			)
		);
	end,

));

#
# Basic block tag.
#
# param1 is K, size of the basic block
# param2 is mu, the block/row/whatever size
#
# we assume K >= mu
#  
Class(ABB, AGenericTag);
