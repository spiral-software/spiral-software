
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Here we define shorter print functions for common constructs.
# Loading this file will cause more compact printing of formulas,
# but the output can not be pasted back into GAP.
#

Import(spl, code);
var.print := self >> Print(self.id);
fBase.print := (self,i,is) >> Print("[", self.params[2], "]_", self.N); 
fTensor.print := FuncClassOper.infix_print(" X ");
fDirsum.print := FuncClassOper.infix_print(" + ");
fCompose.print := FuncClassOper.infix_print(" o ");
gammaTensor.print := FuncClassOper.infix_print(" [#] ");

Value.print := self >> Cond(IsString(self.v), Print("\"",self.v, "\""), Print(self.v));
mId.print := (self,i,is) >> Print("I", self.params[1], ":", self.params[2], "");
fId.print := (self,i,is) >> Print("I", self.params[1]);
Alt.print := (self,i,is) >> Print("<", self.params[1], "|", self.params[2], ">_", self.params[3]);
#Exp.print := self >> self.cprint();
