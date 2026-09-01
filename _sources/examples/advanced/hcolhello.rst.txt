
HCOL Hello World
================

.. code-block:: none

	Load(hcol);
	Import(hcol);

	opts := HCOLopts.getOpts(rec());
	opts.useCReturn := true;
	opts.YType := TPtr(T_Real(64));

	hcol := Reduction(300, (a, b)->add(a, b), V(0), False);

	icode := CoSynthesize(hcol, opts);

	PrintCode("sum", icode, opts);