
Length of Vector
================

.. code-block:: none

	Load(hcol);
	Import(hcol);

	opts := HCOLopts.getOpts(rec());
	opts.useCReturn := true;
	opts.YType := TPtr(T_Real(64));

	len := 128;
	funcname := "l2norm";
	filename := "l2norm.c";
	x := var("x", T_Real(64));
	i := Ind(1);

	hcol := OLCompose(
		PointWise(1, Lambda([x, i], sqrt(x))),
		Reduction(len, (a, b)->add(a, b), V(0), False),    
		PointWise(len, Lambda([x, i], mul(x, x)))
	);
	icode := CoSynthesize(hcol, opts);
	PrintCode(funcname, icode, opts);
