
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(SMP_UnparseMixin, rec(
    includes := ["<include/threads.h>", "<include/smp2.h>"],

    # nothing to do here, threads must be started outside
    smp_fork := (self, o, i, is) >> Print(
        Blanks(i), "/* region must be executed concurrently by ", o.nthreads, " threads */ \n",
        Blanks(i), "{\n",
        self(o.cmd, i+is, is),
        Blanks(i), "}\n"
    ),

    smp_loop := meth(self, o, i, is)
        local v, lo, hi;
        v := o.var;
        lo := 0;
        hi := o.range-1;
        Print(Blanks(i), "{ /* begin parallel loop */\n");
        Print(Blanks(i+is), self.printf("int $1 = $2; \n", [o.tidvar, o.tidexp]));
        Print(Blanks(i+is), self.printf("for(int $1 = $2; $1 <= $3; $1 += $4) {\n", [v, o.tidvar + lo, hi, o.nthreads]));
        self(o.cmd,i+is+is,is);
        Print(Blanks(i+is), "}\n");
        Print(Blanks(i), "} /* end parallel loop */\n");
    end,

    threadId := (self, o, i, is) >> Print("tid")   # Error ??
));

Class(OpenMP_UnparseMixin, SMP_UnparseMixin, rec(
    includes := ["<omp.h>"],

    # start threads using 'omp parallel' pragma
    smp_fork := (self, o, i, is) >> Print(
        Blanks(i), "#pragma omp parallel num_threads(", o.nthreads, ")\n",
        Blanks(i), "{\n",
        self(o.cmd, i+is, is),
        Blanks(i), "}\n"
    ),

    threadId := (self,o,i,is) >> Print("omp_get_thread_num()"),
    barrier := (self,o,i, is) >> Print("#pragma omp barrier\n")
));

Class(OpenMP_UnparseMixin_ParFor, SMP_UnparseMixin, rec(
    includes := ["<omp.h>"],

    # start threads using 'omp parallel' pragma
    smp_fork := (self, o, i, is) >>
        Print(Blanks(i), "/* SMP fork */\n",
        Blanks(i), "{\n",
#   NOTE: why do I need to do that??
        self.opts.unparser(o.cmd,i+is,is),
#        self(o.cmd, i+is, is),
        Blanks(i), "}\n"
    ),

    threadId := (self,o,i,is) >> Print("omp_get_thread_num()"),
    barrier := (self,o,i, is) >> Print(Blanks(i), "/* SMP barrier */\n"),

    smp_loop := (self,o,i,is) >> let(v := o.var, lo := 0, hi := o.range,
            Print(Blanks(i), "#pragma omp parallel for schedule(static, ", Int((hi+1)/2), ")\n",
            Blanks(i), "for(int ", v, " = ", lo, "; ", v, " < ", hi, "; ", v, "++) {\n",
                Blanks(i + is), "int ", o.tidvar, " = ", v, ";\n",
#   NOTE: why do I need to do that??
                self.opts.unparser(o.cmd,i+is,is),
#                self(o.cmd,i+is,is),
                Blanks(i), "}\n")),
));
