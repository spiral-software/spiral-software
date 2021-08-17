
Generate a Pruned Real 3D FFT
+++++++++++++++++++++++++++++

.. code-block:: none

    Load(fftx);
    ImportAll(fftx);

    conf := LocalConfig.fftx.confGPU();

    fwd := true;                  # for foward|inverse [ MDPRDFT | IMDPRDFT ]
    szcube := [ 64, 64, 64 ];

    if fwd then
        prdft := MDPRDFT;
        k := 1;
        name := "mdprdft";
    else
        prdft := IMDPRDFT;
        k := -1;
        name := "imdprdft";
    fi;

    d := Length(szcube);
    name := name::StringInt(d)::"d_"::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
    
    PrintLine("mdprdft-cuda: name = ", name, ", cube = ", szcube, ", size = ",
              StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->" x "::StringInt(s))),
              ";\t\t##PICKME##");

    t := TFCall(ApplyFunc(prdft, [szcube, k]), 
                rec(fname := name, params := []));
    
    opts := conf.getOpts(t);
    tt := opts.tagIt(t);

    c := opts.fftxGen(tt);
    PrintTo(name::".cu", opts.prettyPrint(c));

