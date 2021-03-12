
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_printattr := x->When((x=[] and TYPE(x)="string") or (x<>[] and IsString(x)), Print("\"", x, "\""), Print(x));

attrTakeA  := (to, from) -> When(IsRec(to) and IsBound(to.takeAobj), to.takeAobj(from), to);

Class(AttrMixin, rec(
    a := rec(),

    setA := meth(arg) 
        local self, i, a, f, val;
        a := rec();
        self := arg[1];
        for i in [2..Length(arg)] do
            if IsVarMap(arg[i]) then
                f := NameOf(arg[i][1]);
                val := Eval(arg[i][2]);
                a.(f) := val;
            elif IsList(arg[i]) and Length(arg[i]) in [0,2] then
                if Length(arg[i])>0 then
                    [f, val] := arg[i];
                    a.(f) := val;
                fi;
            else return Error("arg[i] must be a list or varmap");
            fi;
        od;
        return CopyFields(self, rec(a:=a)); 
    end,

    withA := meth(arg) 
        local self, res, set; 
        self := arg[1];
        set := self.setA;
        res := ApplyFunc(set, arg);
        res.a := CopyFields(self.a, res.a);
        return res;
    end,

    hasA := (self, attr) >> Cond(IsBound(self.a.(attr)), true, false),
    
    # getA(<attr>, <def> = false) - returns attribute value.
    #   Optional <def> parameter is value to return when attribute is not found.

    getA := (arg) >> let( self := arg[1], attr := arg[2],
                        def := When(Length(arg)>2, arg[3], false),
                        Cond(IsBound(self.a.(attr)), self.a.(attr), def)),

    printA := self >> let(flds := UserRecFields(self.a),
        Cond(flds=[], Print(""),
             Print(".setA(",
                 DoForAllButLast(flds, x->Print(x, " => ", _printattr(self.a.(x)), ", ")),
                 Last(flds), " => ", _printattr(self.a.(Last(flds))), ")"))),

    takeA   := meth(self, a) self.a := CopyFields(a); return self; end,

    appendA := meth(self, a) self.a := CopyFields(self.a, a); return self; end,

    takeAobj   := meth(self, obj) self.a := CopyFields(obj.a); return self; end,

    appendAobj   := meth(self, obj) self.a := CopyFields(self.a, obj.a); return self; end,

    attrs := ~.takeAobj,

    # testA( attr, v = true ) returns true if object has attribute <attr> with value equal to <v>
	
    testA := (arg) >> let( self := arg[1], attr := arg[2],
        v := When(IsBound(arg[3]), arg[3], true),
        self.hasA(attr) and self.getA(attr) = v),

    dropA := meth(arg)
        local obj, d, s;
        obj := ShallowCopy(arg[1]);
        d   := Drop(arg, 1);
        if Length(arg)=1 then
            Unbind(obj.a);
        elif ForAll(d, IsString) then
            obj.a := ShallowCopy(obj.a);
            for s in d do Unbind(obj.a.(s)); od;
        else
            Error("Usage: dropA([ <attr_name_string>[, <attr_name_string>[, ...]] ])");
        fi;
        return obj;
    end,
    
    # listA() returns list of attribute names and values in [[name, value],...] form.
	
    listA := (self) >> List(Sort(UserRecFields(self.a)), e -> [e, self.a.(e)]),
));


