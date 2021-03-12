
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# q:=HashLookup(bf, DFT(64))[1].ruletree;;
# s:= RCTDA(RC(SumsRuleTree(q)));;
# c:= _CodeSums(s, Y, X);;

# cc:=Copy(c);;
# cc:=Compile.pullDataAndDecls(cc);;
# cc:=Compile.fastScalarize(cc);;
# cc:=FlattenCode(UnrollCode(cc));;

# cc := CopyPropagate(cc);;

# IsLoop := c -> Same(ObjId(c),loop);


Declare(myUnrollLoop);

_myUnrollCode := function(c,decls)
     local op,d,len,tentry;
     op := ObjId(c);
     if Same(op,loop) then return myUnrollLoop(c,  decls);
     elif Same(op,chain) then return Concatenation(List(c.cmds, x->_myUnrollCode(x,decls)));
     elif Same(op,decl) then
	 Append(decls, c.vars); 
	 for d in c.vars do
	    if IsArray(d.t) then
		d.value := V(List([1..d.t.size], i -> var.fresh_t("s", d.t.t))); fi;
	 od;
	 return _myUnrollCode(c.cmd,  decls);
     elif Same(op,data) then
	 #datas.(c.var.id) := c.value;
	 c.var.value := c.value;
	 return _myUnrollCode(c.cmd,  decls);
     elif Same(op,assign) then return [c];
     else Error("Can't handle object of type '", op, "'");
     fi;
end;

myUnrollLoop := function(c,  decls)
    local result, cmd, cmds, ind, val, i;
    cmds := _myUnrollCode(c.cmd,decls);
    ind := c.var;
    result := [];
    for i in c.range do 
       val := V(i);
       Add(result, assign(ind, val));
       Append(result, Copy(cmds));
    od;
    return result;
end;

myUnrollCode := c -> _myUnrollCode(c,[]);
