#define TableBytes(nelems)     (SIZE_HD * ((nelems)+2))
#define TableSize(table)       (GET_SIZE_BAG(table)/SIZE_HD - 2)
#define TableNumEnt(table)     ((Int)HD_TO_INT(PTR_BAG(table)[TableSize(table)]))
#define SetTableNumEnt(table, nument)          (SET_BAG(table, TableSize(table), INT_TO_HD(nument)))
#define TableId(table)               (Bag)(PTR_BAG(table)[TableSize(table)+1])
#define SetTableId(table, val)       SET_BAG(table, TableSize(table)+1, (val))

/** Find an entry in function locals or return 0                           */
Obj  FuncLocalsLookup(Obj hdFunc, char * name, int* ptr_index);
 
/* look in idents.h for OFS_XXX offsets */

Obj  TableCreate(UInt size);
Obj  TableCreateId(UInt size, Obj id);
Obj  TableCreateT(UInt type, UInt size);

Obj  TableResize(Obj hdTable, UInt new_size, UInt name_ofs);

Obj  TableAdd(Obj hdTable, UInt pos, Obj what);

Obj  TableAddIdent(Obj hdTable, UInt pos, char * name);

Obj  TableAddRecnam(Obj hdTable, UInt pos, char * name);

UInt TableLookup(Obj hdTable, char * name, UInt name_ofs);

extern void TableRehash(Obj hdTable);
extern Obj TableToList(Obj hdTable);

