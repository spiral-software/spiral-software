
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F ==========================================================================
#F ParCell(<num_spus>, <pkSize>) - Cell parallelization tag
#F    .params[1] is num_spus
#F    .params[2] is packetSize
#F   
Class(ParCell, AGenericTag, rec(isCell := true));



#F ==========================================================================
#F ParCell_auto (<num_spus>) - Cell parallelization tag
#F    .params[1] is num_spus
#F Highest possible packet size is automatically chosen
Class(ParCell_auto, AGenericTag, rec(isCell := true));


#F ==========================================================================
#F ParDMPCell_old(<num_spus>) - Cell DMP parallelization tag
#F    .params[1] is num_spus
Class(ParCellDMP_old, AGenericTag, rec(isCell := true));

#F ==========================================================================
#F ParDMPCell(<num_spus>) - Cell DMP parallelization tag
#F    .params[1] is num_spus
#F    .params[2] is v
Class(ParCellDMP, AGenericTag, rec(isCell := true));

Class(StickyL, AGenericTag);

