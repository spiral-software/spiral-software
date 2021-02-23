/****************************************************************************
**
*A  idents.c                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions for mapping  identifiers  to  variables.
**
*/
#include        <assert.h>
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */

#include        "idents.h"              /* declaration part of the package */
#include        "record.h"              /* SetRecname                      */
#include        "namespaces.h"
#include        "tables.h"
#include        "scanner.h"
#include        "list.h"
#include        "namespaces_bin.h"

Obj  HdPkgRecname;
extern Obj Props(Obj);
/****************************************************************************
**
*F  MakeIdent( <name> ) . . . . . . . . . . . . . . . . . . make new variable
**
**  'MakeIdent' creates the handle of  the  variable  bag  for  the  variable
**  with the identifier  <name>. It  does  not  look  in  any  tables, unlike 
**  FindIdent.
*/
Obj  MakeIdent ( char name [] ) {
    UInt i;
    Bag hd = NewBag( T_VAR, SIZE_HD * OFS_IDENT + strlen(name) + 1 );
    for ( i = 0; i < OFS_IDENT; ++i )
        SET_BAG(hd, i,  0 );
    strncpy( VAR_NAME(hd), name, strlen(name)+1  );
    return hd;
}


// Make a new ident from a string that is already in a bag, for example,
// the variable on the left-hand side of an assignment during parsing.
//
// 'ofs' is the offset to the actual source string in the particular instance

Obj  MakeIdentSafe ( Obj hdNam, UInt ofs ) {
    UInt i, len = strlen((char*)(PTR_BAG(hdNam)+ofs));
    Bag hd = NewBag( T_VAR, SIZE_HD * OFS_IDENT + len + 1 );
    for ( i = 0; i < OFS_IDENT; ++i )
        SET_BAG(hd, i,  0 );
    strncpy( VAR_NAME(hd), (char*)(PTR_BAG(hdNam)+ofs), len+1  );
    return hd;
}

/****************************************************************************
**
*V  HdIdenttab  . . . . . . . . . . . . . . handle of identifier table, local
**
**  'HdIdenttab' is the handle of the identifier table bag.  The table  is  a
**  list that contains all the variable bags.  The entries are  hashed  into
**  this table, i.e., for an identifier we compute a hash value and  then  we
**  put the variable bag for that identifier bag at this  position.  If  this
**  entry is already used by another variable, a situation we call collision,
**  we take the next free entry.
**
**  Note that we keep  the size of the table at least  twice as big as number
**  of occupied elements to reduce the number of collisions.
*/
Bag       HdIdenttab;
UInt   NrIdenttab;


/****************************************************************************
**
*V  HdStack . . . . . . . . .  handle of the function definition stack, local
*V  TopStack  . . . . . . . . . . top of the function definition stack, local
**
**  'HdStack' is the handle of the function definition stack.  This is a list
**  of all the functions that are currently being read organized as a  stack.
**  The entries of this list are the function  definition  bags.  'FindIdent'
**  first searches this stack to look for local  variables  before  searching
**  'HdIdenttab'.
**
**  'TopStack' is the index of the topmost entry in the  stack,  which  grows
**  upward.
*/
Bag             HdStack;
UInt   TopStack;
UInt   FuncDepth;

/****************************************************************************
**
*F  PushFunction( <hdFun> ) . . . . . . . . add another function to the stack
**
**  'PushFunction' adds another function to the  function  definition  stack.
**  It is called from the reader when the reader starts to parse a  function.
**  It makes the local variables known to 'FindIdent'.
*/
void            PushFunction (Bag hdFun)
{
    ++FuncDepth;
    SET_BAG(HdStack, ++TopStack,  hdFun );
}


/****************************************************************************
**
*F  PopFunction() . . . . . . . . . .  remove another function from the stack
**
**  'PopFunction'  removes  the  most  recent  function  from  the   function
**  definition stack.  It is called from the reader when the reader  finished
**  parsing a function.  It makes the local variables again unknown.
*/
void            PopFunction (void)
{
    SET_BAG(HdStack, TopStack--,  0 );
    --FuncDepth;
}


/****************************************************************************
**
*V  IsUndefinedGlobal . . . . .  true if a variable is an new global variable
**
**  'IsUndefinedGlobal' is set by 'FindIdent'  if  <name>  is  a  new  global
**  variable inside a function definition.  'RdVar' tests 'IsUndefinedGlobal'
**  and complains about such variables.
*/
UInt   IsUndefinedGlobal;

/****************************************************************************
**
*F GlobalIdent( <name> ) . . . . . . . . . . . . . . . find a global variable 
**
*/
Bag       GlobalIdent ( char name [] ) {
    UInt k;
    Bag     hd;
    k = TableLookup(HdIdenttab, name, OFS_IDENT);
    hd = PTR_BAG(HdIdenttab)[k];
    if ( hd == 0 )
        hd = TableAddIdent(HdIdenttab, k, name);
    return hd;
}

/****************************************************************************
**
*F  FindIdent( <name> ) . . . . . . . . . . . find variable for an identifier
**
**  'FindIdent' returns the handle of  the  variable  bag  for  the  variable
**  with the identifier  <name>.  'FindIdent'  first  searches  the  function
**  definition bags made added to the stack by 'PushFunction'.  If  no  local
**  variable has this identifier, 'FindIdent'  looks in the global identifier
**  table.  If the identifier is also not found in the  global  table  a  new
**  global variable is created.
*/
typedef enum { FI_READ, FI_WRITE } fi_mode_t;

Bag       _FindIdent ( char name [], fi_mode_t mode );

Bag       FindIdent ( char name [] ) {
    return _FindIdent(name, FI_READ);
}

Bag       FindIdentWr ( char name [] ) {
    return _FindIdent(name, FI_WRITE);
}

Bag       IdentAssign ( char name [] , Bag val ) {
    Bag hdIdent = _FindIdent(name, FI_WRITE);
    SET_BAG(hdIdent, 0,  val );
    return hdIdent;
}


extern Int     inBreakLoop();

Bag       _FindIdent ( char name [], fi_mode_t mode ) {
    Bag           hd, hdIdent, hdNS;
    UInt       i,  k;
    int                 mode_rd = (mode == FI_READ) ? 1 : 0;

    IsUndefinedGlobal = 0;

    /* search private namespaces before searching function stack */
	hdIdent = FindInPrivatePackages(name, mode_rd);
	if( hdIdent ) return hdIdent;

    /* Search the local tables stored on the function stack.         */
    for ( i = TopStack; i > 0; --i ) {
        int ptr_index;
        hd = PTR_BAG(HdStack)[i];
        hdIdent = FuncLocalsLookup(hd, name, &ptr_index);
        if ( hdIdent != 0 ) {
            UInt j;
            // mark variable
            if (i<TopStack)
                SET_FLAG_BAG(PTR_BAG(hd)[ptr_index], BF_ENV_VAR);
            // mark functions we depend from
            for (j=TopStack; j>i; j--)
                SET_FLAG_BAG(PTR_BAG(HdStack)[j], BF_ENVIRONMENT);
            return hdIdent;
        }
    }

    /* Search the current packages                                */
	if (mode_rd) {
		for (i = Input->packageTop; i > 0; --i) {
			hd = PTR_BAG(Input->packages)[i];
			k = TableLookup(hd, name, OFS_IDENT);
			hdIdent = PTR_BAG(hd)[k];
			if (hdIdent != 0) return hdIdent;
		}
	}

    /* Next  search imported namespaces, only if variable is for reading   */
    if ( mode_rd ) {
        for ( i = Input->importTop; i > 0; --i ) {
            hd = PTR_BAG(Input->imports)[i];
            k = TableLookup(hd, name, OFS_IDENT);
            hdIdent = PTR_BAG(hd)[k];
            if ( hdIdent != 0 ) return hdIdent;        
        }
    }
    
    /* At this point an identifier was looked for everywhere except the 
       global table, and not found. When reading a value, we just need to
       check in the global table; when writing we look in the package
       namespace for creating a new identifier */

    /* Look in the global or, when writing, in package namespace           */
    if ( mode_rd ) {
        hdNS = HdIdenttab;
        k = TableLookup(hdNS, name, OFS_IDENT);
        hd = PTR_BAG(hdNS)[k];
        /* If currently reading a function, give a warning                     */
        if ( FuncDepth != 0 && (hd==0 || VAR_VALUE(hd)==0)) IsUndefinedGlobal = 1;
        if ( hd == 0 ) hd = TableAddIdent(hdNS, k, name);
        return hd;
    }
    else { 
        /*  glob   pkg  identtab  FindIdentWr
          x 0      0       0         new->pkg
          x 0      0       1    !!   new->pkg
            0      1       0         pkg
            0      1       1         pkg
    
          x 1      0       0         new->pkg new->identtab
          x 1      0       1         identtab->pkg
            1      1       0         ERROR
            1      1       1         pkg */

        /* When in non-global mode (a package is being created), all writes go */
        /* to that package, even if global variable of same name exists        */
        if(Input->global == 0) {
            assert(Input->package != 0);
            hdNS = Input->package; 
            k = TableLookup(hdNS, name, OFS_IDENT);
            hd = PTR_BAG(hdNS)[k];
            if ( hd == 0 ) {
                if ( FuncDepth != 0 ) IsUndefinedGlobal = 1;
                if (inBreakLoop()) {
                    /* this is only to keep "let()" packages unchanged */
                    /* while in the break loop. */
                    hdNS = HdIdenttab;
                    k = TableLookup(hdNS, name, OFS_IDENT);
                    hd = PTR_BAG(hdNS)[k];
                    if ( hd == 0 ) hd = TableAddIdent(hdNS, k, name);
                    return hd;
                } else
                    return TableAddIdent(hdNS, k, name);
            } 
            else return hd;
        }
        /* for GlobalPackage() */
        else {
            char n[MAX_IDENT_SIZE];  
            UInt len = strlen(name);
            /* name might disappear during garbage collection */
            strncpy(n, name, len+1);

            if(Input->package == 0) GlobalPackageSpec(GlobalIdent(Input->name));
    
            /* check Globals */
            k = TableLookup(HdIdenttab, n, OFS_IDENT);
            hd = PTR_BAG(HdIdenttab)[k];
            if ( hd == 0 ) {
                /* not in Globals */
                UInt pkg_k;
                if ( FuncDepth != 0 ) IsUndefinedGlobal = 1;
                pkg_k = TableLookup(Input->package, n, OFS_IDENT);
                hd = TableAddIdent(Input->package, pkg_k, n);
                TableAdd(HdIdenttab, k, hd);
            }
            else {
                /* in Globals */
                Obj props;
                k = TableLookup(Input->package, n, OFS_IDENT);
                TableAdd(Input->package, k, hd);
                props = Props(hd);
                SetRecname(props, HdPkgRecname, Input->package);
            }

            return hd;
        }
    }
}


/****************************************************************************
**
*V  HdRectab  . . . . . . . . . . . . . .  handle of record name table, local
**
**  'HdRectab' is the handle of the record name table bag.  The  table  is  a
**  list that contains all the record name bags. The entries are hashed into
**  this table, i.e., for a record name compute a hash value and  then 
**  put the record name bag for that record name bag  at  this  position.  If
**  this entry is already used by another record name (hash collision), use the 
**  next free entry.
**
**  Field names for both records (rec()) and tables (tab()) are in this table.
**
**  Note that we keep  the size of the table at least  twice as big as number
**  of occupied elements to reduce the number of collisions.
*/
Bag       HdRectab;

/****************************************************************************
**
*F  FindRecname( <name> ) . . . .  get bag for a record/table field name
**
**  'FindRecname' returns the record name bag for  the  record  name  <name>.
**  If none exists, it creates a new bag and adds it to HdRectab.
*/
Bag       FindRecname (char *name)
{
    UInt       k;

	/* Look in the record name table, but where ?                          */
    k = TableLookup(HdRectab, name, OFS_RECNAM);

    /* If we found our record name, very good.                             */
    if ( PTR_BAG(HdRectab)[k] != 0 )
        return PTR_BAG(HdRectab)[k];
    /* If we find a free slot, still good.                                 */
    else 
        return TableAddRecnam(HdRectab, k, name);
}


/****************************************************************************
**
*F  ExistsRecname( <name> ) . . . . . . check if record field name bag exists
**
**  'ExistsRecname' returns 1 if the record name bag for the rec field <name>
**  exists, 0 otherwise.
*/
int       ExistsRecname (char *name)
{
    UInt       k;
    k = TableLookup(HdRectab, name, OFS_RECNAM);
    return PTR_BAG(HdRectab)[k] != 0;
}


/****************************************************************************
**
*F  completion( <name>, <len> ) . . . . . . . .  find the completions of name
*/
UInt   iscomplete (char *name, UInt len, UInt rn)
{
    char                * curr;
    UInt       i, k;
    Bag           hdTab;

    if ( ! rn )  hdTab = HdIdenttab;
    else         hdTab = HdRectab;

    for ( i = 0; i < TableSize(hdTab); i++ ) 
    {
        if ( PTR_BAG(hdTab)[i] == 0 )
            continue;

        if ( ! rn )  
            curr = VAR_NAME(PTR_BAG(hdTab)[i]);
        else         
            curr = RECNAM_NAME(PTR_BAG(hdTab)[i]);

        for ( k = 0; name[k] != 0 && curr[k] == name[k]; k++ )
            ;
        if ( k == len && curr[k] == '\0' )  
            return 1;
    }
    return 0;
}

UInt   completion (char *name, UInt len, UInt rn)
{
    char                * curr,  * next;
    UInt       i, k;
    Bag           hdTab;

    if ( ! rn )  hdTab = HdIdenttab;
    else         hdTab = HdRectab;

    next = 0;
    for ( i = 0; i < TableSize(hdTab); i++ ) {
        if ( PTR_BAG(hdTab)[i] == 0 )  continue;
        if ( ! rn )  curr =    VAR_NAME(PTR_BAG(hdTab)[i]);
        else         curr = RECNAM_NAME(PTR_BAG(hdTab)[i]);
        for ( k = 0; name[k] != 0 && curr[k] == name[k]; k++ ) ;
        if ( k < len || curr[k] <= name[k] )  continue;
        if ( next != 0 ) {
            for ( k = 0; curr[k] != '\0' && curr[k] == next[k]; k++ ) ;
            if ( k < len || next[k] < curr[k] )  continue;
        }
        next = curr;
    }

    if ( next != 0 ) {
        for ( k = 0; next[k] != '\0'; k++ )
            name[k] = next[k];
        name[k] = '\0';
    }

    return next != 0;
}


/****************************************************************************
**
*F  InitIdents()  . . . . . . . . . . . . . . . initialize identifier package
**
**  'InitIdents' initializes the identifier package. This must be done before
**  the  first  call  to  'FindIdent'  or  'FindRecname',  i.e.,  before  the
**  evaluator packages are initialized.
*/
void            InitIdents (void)
{
    TopStack = 0;
    FuncDepth = 0;
    HdIdenttab = TableCreate(997); 
    HdRectab   = TableCreate(997); 
    HdStack    = NewBag( T_LIST, 1024 * SIZE_HD );
    HdPkgRecname = FindRecname("pkg");
}


