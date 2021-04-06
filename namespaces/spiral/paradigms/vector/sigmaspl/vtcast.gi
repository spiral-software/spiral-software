
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F VTCast(<n>, <to_type>, <from_type>, <v>)
#F
Class(VTCast, TCast, rec( # TCast is a symbol. So all parameters go into .params 
    abbrevs := [],
    def := (n, to_type, from_type, v) -> Perm((), n)
));

