
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



myScript := function()
    local k, filename;

    for k in [16, 32, 64, 128] do
        filename := ConcatenationString("fft", String(k));
        HDLSynthesize(streamDFTUnroll(k, 2, 4), 1, 0, 16, 300, 0, filename);
    od;
end;

