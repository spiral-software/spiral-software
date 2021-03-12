
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


_sum2 := function(lst)
    local i, res;
    res := 0;
    for i in [2..Length(lst)] do
        res := res + lst[i][1] + lst[i][2];
    od;
    return res;
end;

# objhead = [ objid_addr, param1_uid, param2_uid, ... ]
# uid is unique position in the hashtable returned by HashAdd
Class(ObjHashBase, HashTable( 
    (objhead,size) -> When(IsInt(objhead), 1 + (objhead mod size),
                       1 + ((objhead[1] + _sum2(objhead)) mod size)),
    (objhead1, objhead2) -> objhead1 = objhead2)
);

ObjHashBase.operations := WithBases(HashOps, rec(
    Print := o -> Print(o.name)
));

Class(ObjHash, ObjHashBase, rec(
    liveEntries := self >> Filtered(self.entries, True),
    numLiveEntries := self >> Sum(List(self.liveEntries(), Length)),

    nonrecLookupAdd := meth(self, o)
        local lkup;
	lkup := HashLookupUID(self, InternalHash(o));
	if Same(lkup, false) then
	    return HashAdd(self, InternalHash(o), o);
	else return lkup;
	fi;
    end,

    listLookupAdd := meth(self, o)
        local lkup, uids;
	uids := [T_LIST];
	Append(uids, List(o, e -> self.uidObj(e)));
	lkup := HashLookupUID(self, uids);
	if Same(lkup, false) then
	    return HashAdd(self, uids, o);
	else return lkup;
	fi;
    end,

    uidObj := (self,o) >> Cond(
    IsRec(o) and IsBound(o.uid), o.uid, 
    IsRec(o), Error("Unhashed object <o>"),
    IsList(o), self.listLookupAdd(o),
    self.nonrecLookupAdd(o)),

    objAdd := meth(self, res, h)
        local uid;
    uid := HashAdd(self, h, res);
    res.h := h;
    res.uid := uid;
    return res;
    end,

    singletonAdd := (self, o) >> self.objAdd(o, [BagAddr(o)]),

    _prList := meth(self, lst) 
        local p;
    for p in lst do
            if IsRec(p) then Print(ObjId(p));
        elif IsList(p) then Print("(",self._prList(p), ")");
        else Print(p);
        fi;
        Print(":",self.uidObj(p), " ");
        od;
    end,

    debug := true,

    objLookup := meth(self, objid, params) 
        local h,lkup,hcodes;
	h := [BagAddr(objid)];
	Append(h, List(params, p -> self.uidObj(p))); 
	lkup := HashLookup(self, h);

	if self.debug then 
	    if Same(lkup,false) then 
		Print("(", objid, " ", self._prList(params), ")\n");
	    else  
		Print(objid, " : hit ", lkup.uid, "\n"); 
	    fi;
	fi;
	    
	return [lkup,h];
    end,

    memClassFunc := meth(self, cls, orig, bck)
        Constraint(IsBound(cls.(orig)));
        #Constraint(not IsBound(cls.__call_no_memo__));
    cls.(bck) := cls.(orig);
    cls._hash := self;
    bck := RecName(bck); # this will make lookup a bit faster

    cls.(orig) := meth(arg)
            local clsself, params, lkup, res, h;
        clsself := arg[1]; params := Drop(arg,1);
        h := clsself._hash;
        if h<>false then 
        lkup := h.objLookup(clsself, params);
        if lkup[1] <> false then return lkup[1]; fi;
        fi;
        res := ApplyFunc(clsself.(bck), params);
        if h<>false then 
        return h.objAdd(res, lkup[2]);
        else return res;
        fi;
    end;
    end,

    memClass := (self,cls) >> self.memClassFunc(cls, "__call__", "__call_no_memo__")
));
