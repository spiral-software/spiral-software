.. smp_openmp:

SMP/OpenMP
==========

.. only:: html

   .. contents::

SMP Tagged Code Objects
+++++++++++++++++++++++

Parallel Loop == smp_for
------------------------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\smp\code.gi
	Class(smp_loop, loop_base, rec(
	   __call__ := meth(self, nthreads, tidvar, tidexp, 
						loopvar, range, cmd) 
			local result;
			range := toRange(range);
			loopvar.setRange(range);
			loopvar.isLoopIndex := true;
			return WithBases(self, rec(
				operations := CmdOps, 
				nthreads := nthreads, 
				cmd := cmd, 
				var := loopvar, 
				tidvar := Checked(IsLoc(tidvar), tidvar),
				tidexp := toExpArg(tidexp),
				range := range));
		end,
		print := (self, i, is) >> Print(self.name,"(",self.nthreads,", ", 
		   self.tidvar, ", ", self.tidexp, ", ", self.var, ", ", 
		   self.range, ",\n", Blanks(i+is),
		   self.cmd.print(i+is, is),
		   Print("\n", Blanks(i), ")")),
	));


SMP Code Objects and Code Generator
+++++++++++++++++++++++++++++++++++

Thread ID
---------

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\smp\sigmaspl.gi
	Class(threadId, Exp, rec(computeType := self >> TInt));

Barrier
-------

.. code-block:: none

	Class(barrier, call, rec(visitAs := call));

SMP Codegenerator
-----------------

.. code-block:: none

	Class(SMPCodegenMixin, Codegen, rec(
		SMPBarrier := (self, o, y, x, opts) >> chain(
			self(o.child(1), y, x, opts), 
			barrier(o.nthreads, o.tid, "&GLOBAL_BARRIER")),
		SMPSum := (self, o, y, x, opts) >> let(
			outer_tid     := When(IsBound(opts._tid), opts._tid, 0),
			outer_num_thr := When(IsBound(opts._tid), opts._tid.range, 1),
			tid := var.fresh("tid", TInt, o.nthreads * outer_num_thr),
			smp_loop(o.nthreads, tid, (outer_tid * outer_num_thr) + o.tid,
				o.var, o.domain, 
				self(o.child(1), y, x, 
					CopyFields(opts, rec(_tid := tid))))
			)
	));


OpenMP Unparser
+++++++++++++++

Parallel For
------------

Definition
##########

.. code-block:: none

	# Unparser for #pragma omp parallel for
	Class(OpenMP_Unparser, OpenMP_UnparseMixin_ParFor, CUnparserProg);

Mixin
#####

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\smp\unparsed.gi
	Class(OpenMP_UnparseMixin_ParFor, SMP_UnparseMixin, rec(
		includes := ["<omp.h>"],
		threadId := (self,o,i,is) >> Print("omp_get_thread_num()"),
		barrier := (self,o,i, is) >> 
			Print(Blanks(i), "/* SMP barrier */\n"),
		smp_loop := (self,o,i,is) >> let(v := o.var, 
			lo := 0, hi := o.range, 
			Print(Blanks(i), 
			"#pragma omp parallel for schedule(static, ",
			Int((hi+1)/2),")\n",
			Blanks(i), "for(int ", v, " = ", lo, "; ", v, 
			" < ", hi, "; ", v, "++) {\n",
			Blanks(i + is), "int ", o.tidvar, " = ", v, ";\n",
			self.opts.unparser(o.cmd,i+is,is),
			Blanks(i), "}\n")),
	));
	
Parallel Region
---------------

Definition
##########

.. code-block:: none

	# Unparser for #pragma omp parallel regions
	# namespaces/spiral/libgen/recgt.gi
	Class(OpenMP_Unparser, OpenMP_UnparseMixin, CUnparserProg);

Mixin
#####

.. code-block:: none

	# spiral-core\namespaces\spiral\paradigms\smp\unparsed.gi
	Class(OpenMP_UnparseMixin, SMP_UnparseMixin, rec(
		includes := ["<omp.h>"],
		smp_fork := (self, o, i, is) >> Print(
			Blanks(i), "#pragma omp parallel num_threads(", 
			o.nthreads, ")\n",
			Blanks(i), "{\n",
			self(o.cmd, i+is, is),
			Blanks(i), "}\n"
		),
		threadId := (self,o,i,is) >> Print("omp_get_thread_num()"),
		barrier := (self,o,i, is) >> Print("#pragma omp barrier\n")
	));









