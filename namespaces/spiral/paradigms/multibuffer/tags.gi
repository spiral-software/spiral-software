
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ParStream(p, u)
#F p=# of procs
#F u = minimum packet size (specified)
Class(ParStream, AGenericTag, rec(isParStream := true));


#F MBufCell_spec(u1, u2, p)
#F u1= PkSize for AxI.
#F u2=PkSize for L(IxA) and (IxA)L
#F  p=# of procs (Optional: default=1)
#F  r=# of iters to leave for the children (experimental). Optional.
Class(MBufCell_spec, AGenericTag, rec(isMBufCell_spec := true));

#F MBufCell_its(b)
#F Performs a multibuffer loop with exactly b multibuffered iterations
#F Not to be called directly
Class(MBufCell_its, AGenericTag, rec(isMBufCell_its := true));


#F ==========================================================================
#F MemCell(<its>) - Cell from main mem tag
Class(MemCell, AGenericTag, rec(isMemCell := true));


#F MBuf wrapper tags
Class(MBuf_maxWrapper_WHT, AGenericTag, rec(isMaxWrapper_WHT := true));
Class(MBuf_maxWrapper_DFT, AGenericTag, rec(isMaxWrapper_DFT := true));
Class(MBuf_maxWrapper_DFT_vecrecur, AGenericTag, rec(isMaxWrapper_DFT_vecrecur := true));


#-----------------------------------------------------------------------------
# Deprecated
#-----------------------------------------------------------------------------

#F MBufCell(<its>) - Cell buffer tag
Class(MBufCell, AGenericTag, rec(isMBufCell := true));
Class(MBufCell_max, AGenericTag, rec(isMBufCell := true));
#Class(MBufCell_mbuf, AGenericTag, rec(isMBufCell_mbuf := true));


#F ==========================================================================
#F 2DBufVecRecur(<its>) - Cell from main mem tag
Class(2DBufVecRecur, AGenericTag, rec(is2DBufVecRecur := true));

