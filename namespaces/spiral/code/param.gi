
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Declare(param);

#Special ordering for parameters
#Allows control on the interfaces of autolib
Class(ParamOps, ExpOps, rec(
   \<   := (e1,e2) -> Cond(
       IsRec(e1) and IsRec(e2) 
       and IsBound(e1.sort_weight) 
       and IsBound(e2.sort_weight) 
       and e1.sort_weight<>e2.sort_weight,
       e1.sort_weight < e2.sort_weight,
       ObjId(e1) <> ObjId(e2), ObjId(e1) < ObjId(e2),
       e1.rChildren() < e2.rChildren())
));

# Autolib coldness constants. 
# pcCold -  parameter must be available at plan time;
# pcHot - parameter available at compute time only;
# pcAny - parameter can be hot or cold (default);        
#

Class(pcCold, ConstClass);
Class(pcHot,  ConstClass);
Class(pcAny,  ConstClass);

#F param(<type>, <id>, [sort weight, [coldness]])  - symbolic representation of a parameter 
#F                        (eg. of a codelet)
#F Fields: .t  - types
#F         .id - name of the parameter
#F         .sort_weight - weight during sorting
#F         .coldness - parameter coldness in autolib interfaces
Class(param, Loc, rec(
    isParam := true,
    __call__ := (arg) >> let( 
                    self := arg[1],
                    type := arg[2],
                    id   := arg[3],
                    WithBases(self, Checked(IsString(id), IsType(type), rec( 
                            operations := ParamOps,
                            id := id,
                            t := type,
                            sort_weight := When(Length(arg)>=4, arg[4], 0),
                            coldness := When(Length(arg)>=5, arg[5], pcAny))))),

    print := self >> Print(self.name, "(", self.t, ", \"", self.id, "\")"),
    rChildren := self >> [self.t, self.id],
    rSetChild := rSetChildFields("t", "id"),
    from_rChildren := (self, rch) >> CopyFields(ObjId(self)(rch[1], rch[2]), 
            rec(sort_weight := self.sort_weight, coldness := self.coldness)),
    eval := self >> self,
    can_fold := False,

    computeType := self >> self.t,
    setRange := meth(self, r) # when used as a loop counter variable
       self.range := r;
       return self;
    end,
));

IsParam := x -> IsRec(x) and IsBound(x.isParam) and x.isParam;

Class(in_param, param);

Class(Unk, Exp, rec(
    __call__ := (self, t) >> Checked(IsType(t),
	WithBases(self, rec(operations := ExpOps, args:=[t], t := t))),
    computeType := self >> self.args[1],

   isUnk := true, 
));

IsUnk := x -> IsRec(x) and IsBound(x.isUnk) and x.isUnk;

Class(UnkInt, Unk(TInt), rec(print := self >> Print(self.__name__)));


#F allocate(<var>, <type>) -- equivalent of malloc()
#F See also: deallocate(), zallocate()
Class(allocate, assign);

#F zero-allocate(<var>, <type>) -- equivalent of calloc()
#F See also: deallocate(), allocate()
Class(zallocate, assign);

#F deallocate(<var>, <type>) -- equivalent of free()
#F See also: allocate(), zallocate()
Class(deallocate, assign);


# fld(<type>, <loc>, <field_id>)
Class(fld, Loc, rec(
    __call__ := (self, type, loc, id) >> Checked(IsString(id), IsType(type), IsLoc(loc),
	WithBases(self, rec(
		operations := ExpOps, 
		loc := loc, 
		id := id,
		t := type))),
    computeType := self >> self.t,
    print := self >> Print(self.name, "(", self.t, ", ", self.loc, ", \"", self.id, "\")"),
    rChildren := self >> [self.t, self.loc, self.id],
    rSetChild := rSetChildFields("t", "loc", "id"),
    eval := self >> self,
    can_fold := False,
));

# ufld(<loc>, <field_id>)
Class(ufld, fld, rec(
    __call__ := (self, loc, id) >> Checked(IsString(id), IsLoc(loc),
	WithBases(self, rec(
		operations := ExpOps, 
		loc := loc, 
		id := id,
		t := TUnknown))),
    print := self >> Print(self.name, "(", self.loc, ", \"", self.id, "\")"),
    rChildren := self >> [self.loc, self.id],
    rSetChild := rSetChildFields("loc", "id"),
    eval := self >> self
));

# struct(<id>, <fields>)
#   <id> is the string that gives the name of the structure
#   <fields> is a list of <param>s or <var>s that go into the structure
#
Class(struct, Command, rec(
    __call__ := (self, id, fields) >> Checked(IsString(id), IsList(fields), WithBases(self, 
	rec(operations := CmdOps,
	    id := id,
	    fields := fields))),
    rChildren := self >> [self.id, self.fields],
    rSetChild := rSetChildFields("id", "fields"),
    print := (self, i, si) >> Print(self.name, "(\"", self.id, "\", ", self.fields, ")"),
));

Class(ret, ExpCommand);


SubstParams := (s,bindings) -> 
    SubstTopDownNR(s, @(1, param, e -> IsBound(bindings.(e.id))), e -> bindings.(e.id));

SubstParamsCustom := (s,bindings,objid_list) -> 
    SubstTopDownNR(s, @(1, objid_list, e -> IsBound(bindings.(e.id))), e -> bindings.(e.id));

SubstVarsSafe := (s,bindings) -> 
    SubstTopDownNR(s, @(1, var, e -> IsBound(bindings.(e.id))), e -> bindings.(e.id));
