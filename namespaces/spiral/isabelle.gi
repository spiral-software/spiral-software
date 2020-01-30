
# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details

# Functions to export variousl GAP/SPL data structures as Isabelle expressions

Class(IsabelleExport, rec(
       toIsabelle := self >> Error("toIsabelle() not implemented for " :: self.__bases__[1].name),
       IsabelleOpName := self >> Error("IsabelleOpName() not implemented for " :: self.__bases__[1].name)
));
