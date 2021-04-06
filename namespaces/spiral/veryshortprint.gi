
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Here we define shorter print functions for common constructs.
# Loading this file will cause more compact printing of formulas,
# but the output can not be pasted back into GAP.
#

Import(formgen, code);

Diag.print := (s,i,is) >> Print("D");
Blk.print := (s,i,is) >> Print("B");

Gath._sym := true;
Scat._sym := true;
Diag._sym := true;
Blk._sym := true;

HideRTWrap := function()
   RTWrap._origprint := RTWrap.print;
   RTWrap.print := (self,i,is) >> Print(self.rt.node);
end;

ShowRTWrap := function()
   if IsBound(RTWrap._origprint)
       then RTWrap.print := RTWrap._origprint;
   fi;
end;
