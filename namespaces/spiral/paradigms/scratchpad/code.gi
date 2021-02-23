
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(swp_loop, loop);

Class(dma_signal, ExpCommand);
Class(dma_wait, ExpCommand);

Class(cpu_signal, ExpCommand);
Class(cpu_wait, ExpCommand);

Class(barrier_cmd, ExpCommand);
Class(nop_cmd, ExpCommand);

Class(dma_fence, skip);

Class(dma_transfer, Command, rec(
    isAssign := false, # transfers are not assignments, since they take >2 params
    __call__ := (self, loc, exp, size) >> WithBases(self, rec(
        operations := CmdOps,
        loc := toAssignTarget(loc),
                exp := toExpArg(exp),
        size := toExpArg(size)
   )),

   rChildren := self >> [self.loc, self.exp, self.size],
   rSetChild := rSetChildFields("loc", "exp", "size"),
   unroll := self >> self,

   print := (self,i,is) >> let(name := Cond(IsBound(self.isCompute) and self.isCompute,
                                            gap.colors.DarkYellow(self.__name__),
                                            IsBound(self.isLoad) and self.isLoad,
                                            gap.colors.DarkRed(self.__name__),
                                            IsBound(self.isStore) and self.isStore,
                                            gap.colors.DarkGreen(self.__name__),
                                            self.__name__),
                                Print(name, "(", self.loc, ", ", self.exp, ", ", self.size, ")"))
));

Class(dma_load, dma_transfer);
Class(dma_store, dma_transfer);

# wrapper for dma_xfer, includes the dma transfer size.
# dma_size( dma_xfer(a,b), c);
Class(dma_size, ExpCommand);
Class(par_exec, chain);

#The next two dma_jump and cpu_jump are used only in the CM stuff in ScratchX86
Class(dma_jump,ExpCommand);
Class(cpu_jump,ExpCommand);

#For thread registration using the Fast Barrier
Class(register,ExpCommand);
Class(initialization, ExpCommand);
Class(add_buffer,skip);
