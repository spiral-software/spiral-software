
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details



# NOTE: a hack, because we essentially ignore doHashValues
_constHashPat := @@(1, Value, (e, cx) -> 
    e.t.doHashValues and not IsIntT(e.t) and e.t<>TString and e.t<>TBool
    and not Last(cx.parents) _is data
    and not ForAny(cx.parents, p -> p _is Value)
); 

# ^^ NOTE: strings remapped to vars cause trouble in make_env

# Below could be done automatically in Value constructor (Value.new),
# but HashTable object is not implemented efficiently, leading
# to long runtimes when huge data(..) blocks were used (e.g., SAR)
#
HashConstantsCode := (code, hashtab, hashadd_func, pat) -> SubstTopDownRulesNR(code, rec(
    hash_constant := Rule(pat, e -> hashadd_func(hashtab, e)
)));

#
# Hash for constants
#
NewConstantHash := () -> HashTable( (key,size) -> key[1].t.hash(key[1].v, size) );
GlobalConstantHash := NewConstantHash();

HashedValue := function(conhash, val)
    local hashed, h;
    if not val.t.doHashValues then return val; fi;
    hashed := HashLookup(conhash, [val, val.t]);
    if hashed = false then
        h := val;
        HashAdd(conhash, [val,val.t], h);
        return h;
    else 
      return hashed;
    fi;
end;

#
# Hash for constants remapped to variables 
#
NewConstantVarHash := () -> HashTable( (key,size) -> key[1].t.hash(key[1].v, size) );
GlobalConstantVarHash := NewConstantVarHash(); 

VHashedValue := function(conhash, val)
    local hashed, h;
    if not val.t.doHashValues then return val; fi;
    hashed := HashLookup(conhash, [val, val.t]);
    if hashed = false then
        h := var.fresh_t("C", val.t);
        h.value := val;
        HashAdd(conhash, [val, val.t], h);
        return h;
    else 
      return hashed;
    fi;
end;

# 
# Compiler interface
# 
FlushConsts := function()
    GlobalConstantHash    := NewConstantHash();
    GlobalConstantVarHash := NewConstantVarHash(); 
end;

HashConsts := function (c, opts)
    if IsBound(opts.declareConstants) and opts.declareConstants then
        c := HashConstantsCode(c, GlobalConstantVarHash, VHashedValue, _constHashPat);
    else
        c := HashConstantsCode(c, GlobalConstantHash, HashedValue, _constHashPat);
    fi;
    return c;
end;

#F DeclareConstantsHash(<c>, <htab>)
#F
_declareConstantsHash := function(c, htab)
    local hh, h;
    for hh in htab.entries do
        for h in hh do
	    c := data(h.data, h.key[1], c);
	od;
    od;	
    return c;
end;

#F DeclareConstantsCode(<c>, <opts>)
#F
#F Declares constants in the code as variables, without polluting the global hash.
#F Constants are declared at the top level, without regard of placement.
#F
#F To better localize use DeclareConstantsCodeLocally(c, [func], opts)
#F
DeclareConstantsCode := function(c, opts)
    local htab, hh, h;
    htab := NewConstantVarHash();
    c := HashConstantsCode(c, htab, VHashedValue, _constHashPat);
    return _declareConstantsHash(c, htab);
end;


_localDeclareConstants := function(x, opts) 
    local ch, c;
    ch := x.rChildren();
    ch := List(ch, c -> Cond(not IsCommand(c), c, DeclareConstantsCode(c, opts)));
    return x.from_rChildren(ch);
end;

#F DeclareConstantsCode(<c>, <patterns>, <opts>)
#F 
#F Applies DeclareConstantsCode locally to each of the patterns.
#F This function is non-recursive, i.e. if one of the patterns is matched, it won't 
#F go inside the body. So for best results all patterns should be mutually exclusive.
#F 
DeclareConstantsCodeLocally := (c, patterns, opts) -> SubstTopDownRulesNR(
    c, List(patterns, p -> Rule(p, _localDeclareConstants))
);
