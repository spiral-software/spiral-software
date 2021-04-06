.. _profiler:


Profiler
========

Top-Level Flow
++++++++++++++

.. code-block:: none

	opts := SpiralDefaults;
	c := CodeRuleTree(RandomRuleTree(DFT(8), opts), opts);
	PrintCode("dft8", c, opts);
	CMeasure(c, opts);		# measure the runtime
	CMatrix(c, opts);		# construct the transform matrix from c

Inspect Profiles
++++++++++++++++

.. code-block:: none

	opts.profile;
	default_profiles;
	Exec("dir spiral-localprofiler\\targets");
	Exec("dir spiral-localprofiler\\targets\\win-x64-icc");
	Exec("type spiral-localprofiler\\targets\\win-x64-icc\\Makefile");

Look At the Disk Contents
+++++++++++++++++++++++++

.. code-block:: none

	# see in which drive we are. Usually C: or D:
	Exec("cd");
	# if no outdir is bound in opts this is the default temp path
	IsBound(opts.outdir);
	Exec("dir \\tmp\\"::StringInt(GetPid()));
	Exec("type \\tmp\\"::StringInt(GetPid())::“\\testcode.c");





