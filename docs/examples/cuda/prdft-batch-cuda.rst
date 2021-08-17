
Generate a Batch of 1d Real DFTs
++++++++++++++++++++++++++++++++

.. code-block:: none

    Load(fftx);
    ImportAll(fftx);
    Import(realdft);

    conf := LocalConfig.fftx.confGPU();

    n := 2;
    d := 2;
    N := 128;
    fwd := true;                    ## for foward|inverse [ PRDFT | IPRDFT ]

    iter := List([1..d], i->Ind(n));

    pdft := When(fwd,
                 PRDFT,
                 IPRDFT
    );

    name := "grid_"::pdft.name::StringInt(d)::"d_cont";
    PrintLine ( "prdft-batch-cuda: name = ", name, ";\t\t##PICKME##");

    t := let ( TFCall(TMap(pdft(N, -1), iter, APar, APar), 
                      rec(fname := name, params := []))
    );

    opts := conf.getOpts(t);
    tt := opts.tagIt(t);

    c := opts.fftxGen(tt);
    PrintTo ( name::".cu", opts.prettyPrint(c) );
