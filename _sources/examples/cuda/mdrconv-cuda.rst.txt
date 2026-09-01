
Real Convolution
++++++++++++++++

.. code-block:: none

    Load(fftx);
    ImportAll(fftx);

    conf := LocalConfig.fftx.confGPU();

    szcube := [ 64, 64, 64 ];
    d := Length(szcube);

    name := "mdrconv"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    PrintLine("mdrconv-cuda: name = ", name, ", cube = ", szcube, ";\t\t##PICKME##");

    symvar := var("sym", TPtr(TReal));

    t := TFCall(IMDPRDFT(szcube, 1) * RCDiag(FDataOfs(symvar, 2*Product(DropLast(szcube, 1))* (Last(szcube)/2+1), 0)) * MDPRDFT(szcube, -1), 
                rec(fname := name, params := [symvar])
    );

    opts := conf.getOpts(t);
    tt := opts.tagIt(t);

    c := opts.fftxGen(tt);
    PrintTo ( name::".cu", opts.prettyPrint(c) );
