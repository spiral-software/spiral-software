
Generate a batch of 3D FFTs
+++++++++++++++++++++++++++

.. code-block:: none

    Load(fftx);
    ImportAll(fftx);
    ImportAll(simt);

    conf := LocalConfig.fftx.confGPU();

    nbatch := 2;
    szcube := [ 64, 64, 64 ];

    d := Length(szcube);
    name := "dft"::StringInt(d)::"d_batch_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));

    PrintLine("mddft-batch-cuda: batch = ", nbatch, " d = ", d, " cube = ", szcube, " Name prefix = ", name, ";\t\t##PICKME##");
    
    t := let ( batch := nbatch,
               apat := When ( true, APar, AVec),
               k := -1,
               TFCall ( TRC ( TTensorI ( MDDFT ( szcube, k), batch, apat, apat )), 
                        rec ( fname := name, params := [] ))
    );

    opts := conf.getOpts(t);
    tt := opts.tagIt(t);

    c := opts.fftxGen(tt);
    PrintTo(name::".cu", opts.prettyPrint(c));
