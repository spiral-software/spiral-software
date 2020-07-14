#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "function.h"
#include        "integer.h"
#include        "tables.h"
#include        "idents.h"
#include        "record.h"              /* SetRecname                      */
#include        "scanner.h"             /* MAX_IDENT_SIZE                  */
#include        "args.h"
#include        "list.h"
#include        "eval.h"

/** Find an entry in function locals or return 0                           */
Obj  FuncLocalsLookup(Obj hdFunc, char * name, int* ptr_index) {
    UInt nrEntries;
    UInt k;
    Int nrArg, nrLoc;
    ACT_NUM_ARGS_FUNC(hdFunc, nrArg);
    ACT_NUM_LOCALS_FUNC(hdFunc, nrLoc);
    nrEntries = nrArg + nrLoc;
    for ( k = 1; k <= nrEntries; ++k ) {
        if ( ! strcmp( name, (char*)(PTR_BAG((Bag)PTR_BAG(hdFunc)[k])+OFS_IDENT) ) ) {
            if (ptr_index) *ptr_index = k;
	    return (Bag)PTR_BAG(hdFunc)[k];
        }
    }
    return 0;
}

Obj  TableCreate(UInt size) {
    Obj hd = NewBag( T_NAMESPACE,  TableBytes(size) );
    SetTableNumEnt(hd, 0);
    SetTableId(hd, 0);
    return hd;
}

Obj  TableCreateId(UInt size, Obj hdId) {
    Obj hd = NewBag( T_NAMESPACE,  TableBytes(size) );
    SetTableNumEnt(hd, 0);
    SetTableId(hd, hdId);
    return hd;
}

Obj  TableCreateT(UInt type, UInt size) {
    Obj hd = NewBag( type,  TableBytes(size) );
    SetTableNumEnt(hd, 0);
    SetTableId(hd, 0);
    return hd;
}


static inline int __hash(Obj hdTable, char * name) {    
    if ( GET_TYPE_BAG(hdTable) == T_MAKELET )
	return 0;
    else {
	int k; char *p;
	for ( k = 0, p = name; *p != '\0'; ++p )  k = 65599 * k + *p;
	k = k % TableSize(hdTable);
	return k;
    }
}

/** Find an entry or first empty spot                                      */
UInt TableLookup(Obj hdTable, char * name, UInt name_ofs) {
    UInt k = __hash(hdTable, name);

    /* Look through the table, until you find a free slot or our name      */
    while ( PTR_BAG(hdTable)[k] != 0
         && strcmp( (char*)(PTR_BAG((Bag)PTR_BAG(hdTable)[k])+name_ofs), name ) ) {
        k = (k + 1) % TableSize(hdTable);
    }
    return k;
}

/** Enlarge the table and rehash all entries.                              */
Obj  TableResize(Obj hdTable, UInt new_size, UInt name_ofs) {
    UInt i, k, nument;
    Obj hd;
    Obj hdTableId;
    Obj hdSav = NewBag( GET_TYPE_BAG(hdTable), GET_SIZE_BAG(hdTable) );
    Obj* pHdSav;
    Obj* pHdTab;

    hdTableId = TableId(hdTable);
    
    pHdSav = PTR_BAG(hdSav);
    pHdTab = PTR_BAG(hdTable);
    
    /* NULL out and copy everything (including TableNumEnt and TableId) */
    for ( i = 0; i < GET_SIZE_BAG(hdTable)/SIZE_HD-2; ++i ) {
        if (*pHdTab) {
	    *pHdSav++ = *pHdTab;
	}
	*pHdTab++ = 0;
    }
    *pHdTab++ = 0;
    *pHdTab++ = 0;
    
    nument = pHdSav - PTR_BAG(hdSav);
    //  CHANGED_BAG(hdSav);
    //  CHANGED_BAG(hdTable);

    Resize( hdTable, TableBytes(new_size) );
    
    for ( i = 0; i < nument; ++i ) {
	hd = (Bag)PTR_BAG(hdSav)[i];
	if ( hd == 0 )  continue;

	k = __hash(hdTable, (char*)(PTR_BAG(hd) + name_ofs));
	while ( PTR_BAG(hdTable)[k] != 0 )
	    k = (k + 1) % TableSize(hdTable);

	SET_BAG(hdTable, k,  hd );
    }
    SetTableNumEnt(hdTable, nument);
    SetTableId(hdTable, hdTableId);
    return hdTable;
}

/** Add an already initialized object to the table  */
Obj  TableAdd(Obj hdTable, UInt pos, Obj what) {
    SET_BAG(hdTable, pos,  what ); 
    SetTableNumEnt(hdTable, TableNumEnt(hdTable)+1);
    /* If the identifer table is overcrowded enlarge it                */
    if ( 5*TableNumEnt(hdTable)/4 >= TableSize(hdTable) )
        TableResize(hdTable,  2 * TableSize(hdTable) + 1, OFS_IDENT);
    return what;
}

extern Obj Props(Obj var);

/** Add an identifier to the table                                         */
Obj  TableAddIdent(Obj hdTable, UInt pos, char * nam) {
    static char name[MAX_IDENT_SIZE];  
    Obj hd, props;
    UInt len;
    len = strlen(nam);
    /* name might disappear during garbage collection */
    strncpy(name, nam, len + 1);

    hd = NewBag( T_VAR, SIZE_HD * OFS_IDENT + len + 1);
    strncpy( VAR_NAME(hd), name, len + 1 );
    SET_BAG(hdTable, pos,  hd ); 
    SetTableNumEnt(hdTable, TableNumEnt(hdTable)+1);
    /* If the identifer table is overcrowded enlarge it                */
    if ( 5*TableNumEnt(hdTable)/4 >= TableSize(hdTable) )
        TableResize(hdTable,  2 * TableSize(hdTable) + 1, OFS_IDENT);
    props = Props(hd);
    SetRecname(props, HdPkgRecname, hdTable);
    return hd;
}

/** Add a record name to the table                                         */
Obj  TableAddRecnam(Obj hdTable, UInt pos, char * nam) {
    static char name[MAX_IDENT_SIZE];  
    Obj hd;
    UInt len;
    len = strlen(nam);
    /* name might disappear during garbage collection */
    strncpy(name, nam, len+1);

    hd = NewBag( T_RECNAM, SIZE_HD * OFS_RECNAM + len + 1);
    strncpy( (char*)(PTR_BAG(hd)+OFS_RECNAM), name, len + 1 );
    SET_BAG(hdTable, pos,  hd );
    SetTableNumEnt(hdTable, TableNumEnt(hdTable)+1);
    /* If the identifer table is overcrowded enlarge it                */
    if ( 5*TableNumEnt(hdTable)/4 >= TableSize(hdTable) )
        TableResize(hdTable,  2 * TableSize(hdTable) + 1, OFS_RECNAM);
    return hd;
}

void TableRehash(Obj hdTable) {
    /* figuring out name offset from objects type */
    Int i, name_offs = OFS_IDENT;
    for (i=0; i<TableSize(hdTable); i++) {
	if (PTR_BAG(hdTable)[i]) {
	    if (GET_TYPE_BAG(PTR_BAG(hdTable)[i])==T_RECNAM) name_offs = OFS_RECNAM;
	    break;
	}
    }
    TableResize(hdTable, TableSize(hdTable), name_offs);
}

Obj TableToList(Obj hdTable) {
    Obj hdList = NewList(0);
    Int i;
    for (i = 0; i <TableSize(hdTable); i++) {
        Obj hd = PTR_BAG(hdTable)[i];
        if (hd) {
            hd = EVAL(hd);
            ListAdd(hdList, hd); 
        }
    }
    return hdList;
}

