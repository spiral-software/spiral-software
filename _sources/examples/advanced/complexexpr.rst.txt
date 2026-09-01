
Complex Expression
==================

.. code-block:: none

	Load(hcol);
	Import(hcol);

	opts := HCOLopts.getOpts(rec());
	opts.useCReturn := false;
	opts.includes := ["<math.h>"];
	opts.XType := TPtr(TInt);
	X.t := TPtr(TInt);
	opts.globalUnrolling := 2;
	opts.YType := TPtr(TDouble);
	opts.arrayDataModifier := "";
	opts.arrayBufModifier := "";
	funcname := "F_003";
	filename := "F_003.c";
	tx := var("tx", TInt);
	ty := var("ty", TInt);
	i := var("i", TInt);
	n := var("N", TInt);
	runs := Ind(n);
	opts.params := [n];
	__NUM_VAR__ := 1;

	# code generation
	hcol := IterDirectSum(runs, n,
	  TCond(
		TLess(
		  GathH(__NUM_VAR__, 1, 0, 1),
		  OLCompose(
			PointWise(1, Lambda([tx,i], 0)),  
			GathH(__NUM_VAR__, 1, 0, 1)
		)),
		OLCompose(
		  PointWise(1, Lambda([tx,i], 1.0)), 
		  GathH(__NUM_VAR__, 1, 0, 1)),
		OLCompose(
		  PointWise(1, Lambda([tx,i], sin(tx))),
		  GathH(__NUM_VAR__, 1, 0, 1)
		)
	  )
	);

	icode := CoSynthesize(hcol, opts);
	PrintCode(funcname, icode, opts);
