
Generate a 3D FFT for NVIDIA GPU
++++++++++++++++++++++++++++++++

This SPIRAL script will generate a 3D FFT of size 64^3 designed to take advantage of the parallel processing cababilities of a GPU.

.. code-block:: none

    Load(fftx);
    ImportAll(fftx);
    ImportAll(simt);

    conf := LocalConfig.fftx.confGPU();

    szcube := [ 64, 64, 64 ];

    d := Length(szcube);
    name := "mddft"::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    PrintLine("mddft-cuda: d = ", d, " cube = ", szcube, "Name prefix = ", name, ";\t\t##PICKME##");
    
    t := TFCall(TRC(MDDFT(szcube, 1)), 
                rec(fname := name, params := []));
    
    opts := conf.getOpts(t);
    tt := opts.tagIt(t);

    c := opts.fftxGen(tt);
    PrintTo(name::".cu", opts.prettyPrint(c));

    
