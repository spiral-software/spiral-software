
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Transforms
# ----------
# This package define the transforms (non-terminals) supported by SPIRAL,
# transform-specific SPL and Sigma-SPL constructs, rewriting rules, etc.
#
# Below are some of the commonly used transforms defined here:
#
# DFT - Discrete Fourier Transform
# PRDFT - Real Discrete Fourier Transform
# Filt - FIR filter                 (in transforms.filtering)
# DWT - Discrete Wavelet Transform  (in transforms.filtering)
#
# DCT1, DCT2, DCT3, DCT4 - Discrete Cosine Transforms of four types
# DST1, DST2, DST3, DST4 - Discrete Sine Transforms of four types
#
#@P

Import(rewrite, code, spl, formgen, sigma);



# PRDFTs needed for dct_dst/mdct.g, RDFT for various DCT rules
Declare(SkewDTT, PolyDTT);
Declare(PRDFT1, PRDFT2, PRDFT3, PRDFT4, RDFT, SRDFT);

Load(spiral.transforms.dct_dst);
Import(dct_dst);

Load(spiral.transforms.dft);
DFT := dft.DFT;
Import(dft);

Load(spiral.transforms.interpolate);

Load(spiral.transforms.dtt);
SkewDTT := dtt.SkewDTT;
PolyDTT := dtt.PolyDTT;

Load(spiral.transforms.realdft);
PRDFT1 := realdft.PRDFT1;
PRDFT2 := realdft.PRDFT2;
PRDFT3 := realdft.PRDFT3;
PRDFT4 := realdft.PRDFT4;
RDFT   := realdft.RDFT;
SRDFT  := realdft.SRDFT;


Load(spiral.transforms.wht);
Include(rht);
Include(inplace);
