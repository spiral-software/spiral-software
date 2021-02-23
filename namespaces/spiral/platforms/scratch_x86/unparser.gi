
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(BarrierScratchUnparserProg, CScratchUnparserProg, rec(
    barrier_cmd := (self,o,i,is) >> Print(Blanks(i), self.opts.barrierCMD(self.opts), self.pinfix(o.args, ", "), ";\n"),
    nop_cmd := (self,o,i,is) >> Print(""),
	register := (self,o,i,is) >> Print(Blanks(i), "if(count == 0) ", self.opts.register(self.opts), self.pinfix(o.args, ", "), ";\n"),
    initialization := (self, o, i, is) >> Print(Blanks(i), self.opts.initialization(self.opts), self.pinfix(o.args, ", "), ";\n"),
	par_exec := (self,o,i,is) >> Print(Blanks(i), "parallel((void*)&sub_cpu) \n",
                              Blanks(i), "{\n", self(o.cmds[1],i+is,is),
                              Blanks(i),"}\n"),
));
