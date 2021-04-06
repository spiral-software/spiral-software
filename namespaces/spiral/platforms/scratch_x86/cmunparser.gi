
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

Class(CCMContextScratchUnparserProg, CScratchUnparserProg, rec(
	par_exec := (self,o,i,is) >> Print(Blanks(i), "parallel(&sub_cpu) \n",
                              Blanks(i), "{\n", self(o.cmds[1],i+is,is),
                              Blanks(i+is), "if(isFinished) {\n",
                              Blanks(i+is+is), "setFinished;\n",
                              Blanks(i), self(o.cmds[2],i+is,is),
                              Blanks(i+is),"}\n",
                              Blanks(i),"}\n"),
));
