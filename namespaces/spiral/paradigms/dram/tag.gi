
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# ADram tag, generic class for DRAM data layout optimizations
# params[1] = rb = row-buffer size in words
# params[2] = m = size of local memory in words
Class(ADram, AGenericTag, rec(
	updateParams := meth(self)
		Checked(Length(self.params)=2);
		self.rb := self.params[1];
		self.m := self.params[2];
	end,
));


# ACube tag: Tag meant for cubic memory mapping for DRAM,
# params[1] = k = cube-edge in words (k^3 = row-buffer size)
# params[2] = m = size of local memory in words
Class(ACube, AGenericTag, rec(
	isDMARead := false,
	isDMAWrite := false,
	updateParams := meth(self)
		Checked(Length(self.params)=2);
		self.k := self.params[1];
		self.m := self.params[2];
	end,
));

Class(ACubeRd, AGenericTag, rec(
	isDMARead := true,
	isDMAWrite := false,
	updateParams := meth(self)
		Checked(Length(self.params)=2);
		self.k := self.params[1];
		self.m := self.params[2];
	end,
));

Class(ACubeWr, AGenericTag, rec(
	isDMARead := false,
	isDMAWrite := true,
	updateParams := meth(self)
		Checked(Length(self.params)=2);
		self.k := self.params[1];
		self.m := self.params[2];
	end,
));



# ATile tag: Tag meant for 2D-tiled memory mapping for DRAM,
# params[1] = k = tile-width in words (k^2 = row-buffer size)
# params[2] = m = size of local memory in words
Class(ATile, AGenericTag, rec(
	isDMARead := false,
	isDMAWrite := false,
	updateParams := meth(self)
		Checked(Length(self.params)=2);
		self.k := self.params[1];
		self.m := self.params[2];
	end,
));

Class(ATileRd, ATile, rec(
	isDMARead := true,
	isDMAWrite := false,
	updateParams := meth(self)
		Checked(Length(self.params)=2);
		self.k := self.params[1];
		self.m := self.params[2];
	end,
));

Class(ATileWr, ATile, rec(
	isDMARead := false,
	isDMAWrite := true,
	updateParams := meth(self)
		Checked(Length(self.params)=2);
		self.k := self.params[1];
		self.m := self.params[2];
	end,
));

#  # size = scratchpad segment size
#  # nsgmts = number of segments, defaults to 1.
#  Class(ALStore, AGenericTag, rec(
#      updateParams := meth(self)
#          Checked(Length(self.params)=3);
#          self.size := self.params[1];
#          self.nsgmts := self.params[2];
#          self.linesize := self.params[3];
#      end,
#      isRegCx := false
#  ));
#  
#  Class(ALStoreCx, ALStore, rec(
#      updateParams := meth(self)
#          Checked(Length(self.params)=3);
#          self.size := self.params[1]/2;
#          self.nsgmts := self.params[2];
#          self.linesize := self.params[3];
#      end,
#      isRegCx := true
#  ));
#  
#  # APad tag.
#  # 
#  # APad is a special buffering tag meant for things like scratchpads, hence
#  # the name. It takes four parameters. In order, they are:
#  # b - the block size in number of elements
#  # s - the segment size, in number of blocks
#  # u - the number of segments
#  # n - a string identifier
#  #
#  # The block size is the SMALLEST allowed transfer size when copying data.
#  # The breakdown rules which propagate the APad tag insure that sub-block
#  # numbers of contiguous elements are never moved when the tag is present.
#  #
#  # A segment is a discrete memory separate from other segments, in the case
#  # that the 'u' parameter is >1. In the case of DPA, each segment is
#  # connected to a different compute processor, and we do parallelization
#  # after the APad tag is dropped.
#  #
#  # The string identifier is for the platform writer. You can label things
#  # like "local memory" or "vector register file."
#  #
#  # Also, the software pipelining loop (rather than a standard ISum) is
#  # automatically used with the APad tag.
#  
#  Class(APad, AGenericTag, rec(
#      applied := false,
#  
#      b := (self) >> self.params[1],
#      s := (self) >> self.params[2],
#      u := (self) >> self.params[3],
#      bs := (self) >> self.params[1] * self.params[2],
#      bsu := (self) >> Product(DropLast(self.params, 1)),
#      n := (self) >> self.params[4],
#  
#      apply := self >> CopyFields(self, rec(applied := true)),
#  
#      updateParams := meth(self)
#          Checked(Length(self.params)=4);
#      end,
#  
#      print := meth(self) 
#          Print(self.name, "(", PrintCS([self.b(), self.s(), self.u()]),
#              ", \"", self.n(), "\")");
#      end
#  ));