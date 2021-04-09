
Find a Good Rule Tree with Dynamic Programming
++++++++++++++++++++++++++++++++++++++++++++++

The previous examples all use RandomRuleTree() to build a tree of breakdown rules for the transform.  It's
a quick way to get a valid rule tree for experimentation, but the random choice means the resulting code 
will probably not perform very well compared to a really good tree.  For larger transforms there are just too many 
possible rule trees to try every one, so SPIRAL provides a dynamic programming-based search that assembles a
good rule tree bottom up by finding the best smaller components.

This example shows how to use the dynamic programming search 
function, DP().  It uses the SPIRAL profiler to measure performance, so make sure the profiler is properly
installed and configured (see :ref:`Installing SPIRAL <installing>`).

The script will run for quite a while and generate a lot of progress messages.  Follow
it along to get an idea of how it works from bottom up to build a good tree.


.. code-block:: none

    opts := SIMDGlobals.getOpts(AVX_4x64f);
    transform := TRC(DFT(512)).withTags(opts.tags);
    best := DP(transform, rec(), opts);
    ruletree := best[1].ruletree;
    icode := CodeRuleTree(ruletree, opts);
    PrintTo("AVX_DFT512.c", PrintCode("AVX_DFT512", icode, opts));

