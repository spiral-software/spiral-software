.. _namespaces:

Packages and Namespaces
=======================

Load Tree
+++++++++

init.g

.. code-block:: none

	RequirePackage("arep");
	Package(spiral);
	Include(config);
	Include(trace);

	Load(spiral.rewrite);
	Load(spiral.code);     
	ProtectNamespace(code);
	Declare(CMeasure);


Packages and Namespace Commands
+++++++++++++++++++++++++++++++

.. code-block:: none

	avx.addsub_4x64f;
	Import(avx);
	addsub_4x64f;
	avx.addsub_4x64f := false;
	addsub_4x64f;

	Dir(avx);
	Info(addsub_4x64f);
	Doc(DP);

