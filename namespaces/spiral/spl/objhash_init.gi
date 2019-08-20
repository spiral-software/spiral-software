
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details


ObjLookup := arg -> ApplyFunc(ObjHash.objLookup, arg);
ObjAdd    := arg -> ApplyFunc(ObjHash.objAdd, arg);
MemClass  := arg -> ApplyFunc(ObjHash.memClass, arg);
MemClassFunc  := arg -> ApplyFunc(ObjHash.memClassFunc, arg);
SingletonAdd  := arg -> ApplyFunc(ObjHash.singletonAdd, arg);

objs := x -> Filtered(Collect(x, @), IsRec);
uids := x -> List(objs(x), o->o.uid);  
### init
SingletonAdd(TInt);
SingletonAdd(TComplex);
SingletonAdd(TUnknown);
SingletonAdd(TReal);


MemClass(TArray);
MemClass(TArrayBase);
MemClass(TVect);

#ClassSPL.hash := ObjHash;
#Function.hash := ObjHash;

########
MemClass(Exp);
MemClass(ListableExp);
MemClass(Lambda);
MemClass(FList);
MemClass(FData);
MemClass(nth);
MemClass(Value);

MemClassFunc(Value, "new", "new_no_memo");
MemClassFunc(Value, "newbase", "newbase_no_memo");
