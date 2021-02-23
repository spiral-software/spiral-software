
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Class(DistMixin1, rec(
# #   includes := ["<include/threads.h>", "<include/smp2.h>"],
#     dist_loop := meth(self, o, i, is)
#         local v, lo, hi;
#         v := o.var;
#         lo := o.range[1];
#         hi := Last(o.range);
#         Print(Blanks(i), "for(int ", v, " = tid + ", lo, "; ", v, " <= ", hi, "; ", v, "+= ", o.p, ") {\n");
#         Print(Blanks(i+is), "int ", o.tidvar, " = ", self(o.tidexp, i+is+is, is), ";\n");
#         self(o.cmd,i+is,is);
#         Print(Blanks(i), "}\n");
#         Print(Blanks(i), When(IsBound(self.opts.smp), 
#                               self.opts.smp.barrier, 
#                               "//barrier(num_threads, tid, &GLOBAL_BARRIER);\n"));
#     end,
# 
#     threadId := (self, o, i, is) >> Print("tid")
# 
# ));

Class(DistMixin, rec(
#  includes := ["<spumacros.h>"],

   dist_loop := meth(self, o, i, is)
       Print(Blanks(i),    "{\n");
       #Print(Blanks(i+is), "unsigned int ", o.var, " = spe_info.spuid;\n");
       self(o.cmd,i+is,is);
       Print(Blanks(i),    "}\n");
       #Print(Blanks(i),    "//ALL_TO_ALL_BARRIER;\n");
       #Print(Blanks(i),    "BLOCK_ON_READ();\n");
    end,

    dist_barrier := (self,o,i,is) >> Print(Blanks(i), "BLOCK_ON_CPUDMA; ALL_TO_ALL_BARRIER;\n"),

    call := (self, o, i, is) >> Print(Blanks(i), o.args[1].id, self.pinfix(Drop(o.args, 1), ", "), ";\n"),

    fcall := (self, o, i, is) >> Print(self(o.args[1],0,0), "(", self.infix(Drop(o.args, 1), ", "), ")"),


));

Class(DistUnparser, DistMixin, CUnparserProg);
#Class(DistCellUnparser, DistMixin, CellUnparser);
