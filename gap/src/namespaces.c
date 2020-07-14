/****************************************************************************
**
*A  namespaces.c                SPIRAL source                Yevgen Voronenko
**
*A  $Id: namespaces.c 7860 2009-02-12 05:54:01Z vlad $
**
**  A namespace is a table of name->value bindings.
*/
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* Bag, NewBag, T_STRING, .. */
#include        "idents.h"
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "scanner.h"             /* Pr(), Input                     */
#include        "objects.h"
#include		"string4.h"
#include        "function.h"
#include        "tables.h"
#include        "list.h"
#include        "plist.h"
#include        "record.h"
#include        "args.h"
#include        "read.h"                /* BinBag                          */
#include        "spiral_delay_ev.h"
#include        "namespaces.h"

#include		"GapUtils.h"

extern Obj       HdStack, HdIdenttab;
extern UInt      TopStack;

Obj HdLocal;

UInt IMPORTS_LEN = 16;
UInt PACKAGES_LEN = 16;
UInt NEW_TABLE_SIZE = 103;

Obj NN(Obj x) { if(x==0) return HdFalse; else return x; }

#define IS_GLOBAL(ns) GET_FLAG_BAG(ns, BF_NS_GLOBAL)
Obj GLOBAL(Obj ns) {
    SET_FLAG_BAG(ns, BF_NS_GLOBAL);
    return ns;
}
Obj LOCAL(Obj ns) {
    CLEAR_FLAG_BAG(ns, BF_NS_GLOBAL);
    return ns;
}

Obj  UpdateInputData ( TypInputFile * input) {
    if(input->data == 0)
        input->data = NewBag(T_REC, 0);
    SetRecField(input->data, "fileName", StringToHd(input->name));
    SetRecField(input->data, "lineText", StringToHd(input->line));
    SetRecField(input->data, "lineNumber", INT_TO_HD(input->number));

    SetRecField(input->data, "imports", NN(input->imports));
    SetRecField(input->data, "importsTop", INT_TO_HD(input->importTop));
    SetRecField(input->data, "packages", NN(input->packages));
    SetRecField(input->data, "packageTop", INT_TO_HD(input->packageTop));
    SetRecField(input->data, "pkg", NN(input->package));
    SetRecField(input->data, "global", INT_TO_HD(input->global));
    return input->data;
}

Obj  _NamespaceAdd ( Obj hdTo, Obj hdFrom, int destructive ) {
    UInt i;
    assert(hdTo != 0);
    assert(hdFrom != 0);
    for ( i = 0; i < TableSize(hdFrom); i++ ) {
        Obj var = PTR_BAG(hdFrom)[i];
        if ( var != 0 ) {
            UInt idx = TableLookup(hdTo, VAR_NAME(var), OFS_IDENT);
            Obj oldVar = PTR_BAG(hdTo)[idx];
            if ( destructive || (oldVar == 0 || VAR_VALUE(oldVar) == 0) )
                TableAdd(hdTo, idx, var);
        }
    }
    return hdTo;
}

Obj  NamespaceAdd ( Obj hdTo, Obj hdFrom ) { return _NamespaceAdd(hdTo, hdFrom, 1); }

Obj  NamespaceAddNonDestructive ( Obj hdTo, Obj hdFrom ) { return _NamespaceAdd(hdTo, hdFrom, 0); }

Obj  PushNamespaceGlobal ( Obj hdNS ) {
    return NamespaceAdd(HdIdenttab, hdNS);
}

Obj  PushNamespace ( Obj hdNS ) {
    assert(hdNS != 0);
    if ( Input->imports == 0 ) {
        Input->imports = NewBag(T_LIST, SIZE_PLEN_PLIST(IMPORTS_LEN));
        SET_LEN_PLIST(Input->imports, 1);
    }
    ++(Input->importTop);
    AssPlist(Input->imports, Input->importTop, hdNS);
    return hdNS;
 }

void PushNamespaces ( Obj hdNSList ) {
    UInt i;
    assert(hdNSList != 0);
    for(i=1; i <= LEN_LIST(hdNSList); ++i)
      PushNamespace(ELM_LIST(hdNSList, i));
 }

void PushPackages ( Obj hdNSList ) {
    UInt i;
    assert(hdNSList != 0);
    for(i=1; i <= LEN_LIST(hdNSList); ++i)
      PushPackage(ELM_LIST(hdNSList, i));
 }

Obj PopNamespace (void) {
    if( Input->importTop > 0 ) {
        Obj hdPopped = ELM_PLIST(Input->imports, Input->importTop);
        AssPlist(Input->imports, Input->importTop, 0);
        --(Input->importTop);
        SET_LEN_PLIST(Input->imports, Input->importTop);
        return hdPopped;
    }
    else return Error("No namespace on top of stack", 0, 0);
}

int PopAllNamespaces (void) { /* returns number of namespaces popped */
    int res = Input->importTop;
    if ( res != 0 ) {
        SET_LEN_PLIST(Input->imports, 0);
        Input->importTop = 0;
    }
    return res;
}


Obj PushPackage ( Obj hdNS ) {
    assert(hdNS != 0);
    if ( Input->packages == 0 ) {
        Input->packages = NewBag(T_LIST, SIZE_PLEN_PLIST(PACKAGES_LEN));
        SET_LEN_PLIST(Input->packages, 1);
    }
    ++(Input->packageTop);
    AssPlist(Input->packages, Input->packageTop, hdNS);
    Input->package = hdNS;
    //    VAR_VALUE(GlobalIdent("Local")) = hdNS;
    Input->global = IS_GLOBAL(hdNS);
    return hdNS;
 }

Obj PopPackage (void) {
    if( Input->packageTop > 0 ) {
        Obj hdPopped = Input->package;
        SET_ELM_PLIST(Input->packages, Input->packageTop, 0);
        SET_LEN_PLIST(Input->packages, --(Input->packageTop));

        if ( Input->packageTop > 0 ) {
            Input->package = ELM_PLIST(Input->packages, Input->packageTop);
            //VAR_VALUE(GlobalIdent("Local")) = Input->package;
            Input->global = IS_GLOBAL(Input->package);
        }
        else {
            Input->package = 0;
            Input->global = 1;
            //VAR_VALUE(GlobalIdent("Local")) = HdIdenttab;
        }
        return hdPopped;
    }
    else return Error("No package on top of stack", 0, 0);
}


Obj  NamespaceRec ( Obj hdRec ) {
    int i;
    int nfields = GET_SIZE_BAG(hdRec) / (2*SIZE_HD);
    Obj hd = TableCreate(nfields + 2);
    for(i = 0; i < nfields; ++i) {
        char * nam = RECNAM_NAME( PTR_BAG(hdRec)[2*i] );
        UInt k = TableLookup(hd, nam, OFS_IDENT);
        Obj hdVar = TableAddIdent(hd, k, nam);
        SET_BAG(hdVar, 0,  PTR_BAG(hdRec)[2*i+1] ); /* set the value */
    }
    return hd;
}

Obj  RecNamespace ( Obj hdNs ) {
    Obj hdRec;
    UInt i, nfields;
    UInt tpos = 0;
    nfields = TableNumEnt(hdNs);
    hdRec = NewBag(T_REC, (2 * nfields) * SIZE_HD );
    i = 0;
    while(tpos < TableSize(hdNs)) {
        Obj hdVar = PTR_BAG(hdNs)[tpos];
        if ( hdVar != 0 && PTR_BAG(hdVar)[0] != 0) {
            Obj hdRn = FindRecname(VAR_NAME(hdVar));
            SET_BAG(hdRec, 2*i,  hdRn );
            SET_BAG(hdRec, 2*i+1,  PTR_BAG(hdVar)[0] );
            ++i;
        }
        ++tpos;
    }
    if ( GET_SIZE_BAG(hdRec) != 2*i*SIZE_HD )
        Resize(hdRec, 2*i*SIZE_HD);

    return hdRec;
}


Obj _createNamespaces(Obj hd) {
    if(GET_TYPE_BAG(hd)==T_VAR) {
        Obj v = VAR_VALUE(hd);
        if(v == 0 || GET_TYPE_BAG(v) != T_NAMESPACE) {
            Obj ns = TableCreateId(NEW_TABLE_SIZE, hd);
            SET_VAR_VALUE(hd, ns);
            return ns;
        }
        else return v;
    }
    else {
        Obj lhs;
        {assert(GET_TYPE_BAG(hd)==T_RECELM);}
        lhs = _createNamespaces(PTR_BAG(hd)[0]);
        if(GET_TYPE_BAG(lhs)!=T_NAMESPACE)
            return Error("StartPackage(a.b....s) form expects <a> to be a package, "
                         "and <b>..<s> - subpackage fields", 0, 0);
        else {
            Obj fld = RecnameObj(PTR_BAG(hd)[1]);
            UInt idx = TableLookup(lhs, (char*)PTR_BAG(fld), OFS_IDENT);
            if(PTR_BAG(lhs)[idx]==0 || GET_TYPE_BAG(VAR_VALUE(PTR_BAG(lhs)[idx])) != T_NAMESPACE) {
                Obj var = TableAddIdent(lhs, idx, (char*)PTR_BAG(fld));
                Obj ns = TableCreateId(NEW_TABLE_SIZE, hd);
                SET_VAR_VALUE(var, ns);
                return ns;
            }
            else return VAR_VALUE(PTR_BAG(lhs)[idx]);
        }
    }
}

Obj _StartPackage(Obj par) {
    char * usage = "Usage: StartPackage ( [ <var> | <pkgname> | <ns> ] )\n (%s given)";
    Obj v;
    switch(GET_TYPE_BAG(par)) {
    case T_NAMESPACE:   return par;
    case T_DELAY:
        v = PTR_BAG(par)[0];
        if ( GET_TYPE_BAG(v) != T_VAR )
            return _StartPackage(v); /*Error("StartPackage: only D(<var>) allowed", 0, 0);*/
        else if ( GET_TYPE_BAG(VAR_VALUE(v)) != T_NAMESPACE ) {
            Obj ns = TableCreateId(NEW_TABLE_SIZE, v);
            SET_VAR_VALUE(v, ns);
            return VAR_VALUE(v);
        }
        else return _StartPackage(VAR_VALUE(v));
    case T_STRING:      return _StartPackage(FindIdentWr(HD_TO_STRING(par)));
    case T_VAR:
        v = VAR_VALUE(par);
        if (v == 0) {
            Obj ns = TableCreateId(NEW_TABLE_SIZE, par);
            SET_VAR_VALUE(par, ns);
            return VAR_VALUE(par);
        }
        else return _StartPackage(v);
    case T_RECELM:
        { Obj last = PTR_BAG(par)[0];
          while(GET_TYPE_BAG(last)==T_RECELM) last = PTR_BAG(last)[0];

          if(GET_TYPE_BAG(last)!=T_VAR)
              return Error("StartPackage(a.b....s) form expects <a> to be a package, "
                            "and <b>..<s> - subpackage fields", 0, 0);
          if(VAR_VALUE(last)!=0 && GET_TYPE_BAG(VAR_VALUE(last)) != T_NAMESPACE)
              return Error("StartPackage(a.b....s) form expects <a> to be a package",
                           0, 0);
          return _createNamespaces(par);
        }
    default:
        if(GET_TYPE_BAG(par) < T_VAR) return Error(usage, (Int)TNAM_BAG(par), 0);
        else return _StartPackage(EVAL(par));
    }
}

Obj StartPackage(char * name) {
    Obj hdName = StringToHd(name);
    return PushPackage(LOCAL(_StartPackage(hdName)));
}
Obj StartPackageSpec(Obj spec) {
    return PushPackage(LOCAL(_StartPackage(spec)));
}
Obj GlobalPackage(char * name) {
    Obj hdName = StringToHd(name);
    return PushPackage(GLOBAL( _StartPackage(hdName) ));
}
Obj GlobalPackageSpec(Obj spec) {
    return PushPackage(GLOBAL(_StartPackage(spec)));
}

Obj StartPackage2(char * super, char * sub) {
    Obj rec = FindIdentWr(super);
    Obj elm = FindRecname(sub);
    Obj ns = _StartPackage(BinBag(T_RECELM, rec, elm));
    return PushPackage(LOCAL(ns));
}

Obj GlobalPackage2(char * super, char * sub) {
    Obj rec = FindIdentWr(super);
    Obj elm = FindRecname(sub);
    Obj ns = _StartPackage(BinBag(T_RECELM, rec, elm));
    return PushPackage(GLOBAL(ns));
}

Obj EndPackage(void) {
    return PopPackage();
}

Obj  FunStartPackage ( Obj hdCall ) {
    char * usage = "Usage: StartPackage ( [ <var> | <pkgname> | <ns> ] )";
    Obj ns;
    if ( GET_SIZE_BAG(hdCall) > 2 * SIZE_HD ) return Error(usage, 0, 0);
    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD )
        ns = _StartPackage(PTR_BAG(hdCall)[1]);
    else ns = TableCreate(NEW_TABLE_SIZE);
    return PushPackage(LOCAL(ns));
}

Obj  FunGlobalPackage ( Obj hdCall ) {
    char * usage = "Usage: GlobalPackage ( [ <var> | <pkgname> | <ns> ] )";
    Obj ns;
    if ( GET_SIZE_BAG(hdCall) > 2 * SIZE_HD ) return Error(usage, 0, 0);
    if ( GET_SIZE_BAG(hdCall) == 2 * SIZE_HD )
        ns = _StartPackage(PTR_BAG(hdCall)[1]);
    else ns = TableCreate(NEW_TABLE_SIZE);
    return PushPackage(GLOBAL(ns));
}

Obj  FunEmptyPackage ( Obj hdCall ) {
    char * usage = "Usage: EmptyPackage ( [ <var> | <pkgname> | <ns> ] )";
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    return _StartPackage(PTR_BAG(hdCall)[1]);
}

Obj  FunGetPackage ( Obj hdCall ) {
    char * usage = "Usage: GetPackage ( )";
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD ) return Error(usage, 0, 0);
    return NN(Input->package);
}

Obj  FunEndPackage ( Obj hdCall ) {
    char * usage = "Usage: EndPackage ( )";
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD ) return Error(usage, 0, 0);
    return NN(EndPackage());

}

Obj  FunCurrentFile ( Obj hdCall ) {
    char * usage = "Usage: CurrentFile ( )";
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD ) return Error(usage, 0, 0);
    return UpdateInputData(Input);
}

Obj  FunCurrentFileStack ( Obj hdCall ) {
    char * usage = "Usage: CurrentFileStack ( )";
    Obj res;
    Int i=0;
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD ) return Error(usage, 0, 0);

    res = NewBag(T_LIST, SIZE_PLEN_PLIST(SCANNER_INPUTS));
    SET_LEN_PLIST(res, 0);
    while ( &InputFiles[i-1] != Input) {
        AssPlist(res, i+1, UpdateInputData(&InputFiles[i]));
        ++i;
    }
    return res;
}

Obj  FunCurrentDir ( Obj hdCall )
{
    char * usage = "Usage: CurrentDir ( )";
    char * file = Input->name;

    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD )
		return Error(usage, 0, 0);

    if ( strcmp(file, "*stdin*")==0 || strcmp(file, "*stderr*")==0 )
        return StringToHd(config_demand_val("spiral_dir")->strval );

    else {
        Int pos;
		char path_sep = config_demand_val("path_sep")->strval[0];
		Obj hd = StringToHd(file);

		pos = strlen(file);
		while ( pos >=0 && file[pos] != path_sep )
			--pos;

		if ( pos > 0 ) {
			CHARS_STRING(hd)[pos+1] = '\0';
			Resize(hd, pos+1);
		}

		return hd;
    }
}

Obj  FunPopNamespace ( Obj hdCall ) {
    char * usage = "Usage: PopNamespace ( )";
    if ( GET_SIZE_BAG(hdCall) == SIZE_HD ) return PopNamespace();
    else return Error(usage, 0, 0);
}

Obj  FunPopAllNamespaces ( Obj hdCall ) {
    char * usage = "Usage: PopAllNamespaces ( <ns-rec> )";
    if ( GET_SIZE_BAG(hdCall) != 1 * SIZE_HD ) return Error(usage, 0, 0);
    return INT_TO_HD(PopAllNamespaces());
}

Obj  FunPushNamespace ( Obj hdCall ) {
    char * usage = "Usage: PushNamespace ( <ns> )";
    Obj ns;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    ns = EVAL(PTR_BAG(hdCall)[1]);
    ns = INJECTION_D(ns);
    if ( GET_TYPE_BAG(ns) != T_NAMESPACE ) return Error(usage, 0, 0);
    PushNamespace(ns);
    return HdVoid;
}

Obj  FunPushNamespaceGlobal ( Obj hdCall ) {
    char * usage = "Usage: PushNamespaceGlobal ( <ns> )";
    Obj ns;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    ns = EVAL(PTR_BAG(hdCall)[1]);
    ns = INJECTION_D(ns);
    if ( GET_TYPE_BAG(ns) != T_NAMESPACE ) return Error(usage, 0, 0);
    PushNamespaceGlobal(ns);
    return HdVoid;
}

Obj  FunNamespaceAdd ( Obj hdCall ) {
    char * usage = "Usage: NamespaceAdd ( <to>, <from> ) - copies all identifiers from <from> to <to>";
    Obj from, to;
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD ) return Error(usage, 0, 0);
    to = EVAL(PTR_BAG(hdCall)[1]);
    from = EVAL(PTR_BAG(hdCall)[2]);
    to = INJECTION_D(to);
    from = INJECTION_D(from);
    if ( GET_TYPE_BAG(to) != T_NAMESPACE ) return Error(usage, 0, 0);
    if ( GET_TYPE_BAG(from) != T_NAMESPACE ) return Error(usage, 0, 0);
    return NamespaceAddNonDestructive(to, from);
}

Obj  FunRecNamespace ( Obj hdCall ) {
    char * usage = "Usage: RecNamespace ( <ns> )";
    Obj ns;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    ns = EVAL(PTR_BAG(hdCall)[1]);
    ns = INJECTION_D(ns);
    if ( GET_TYPE_BAG(ns) != T_NAMESPACE ) return Error(usage, 0, 0);
    return RecNamespace(ns);
}

Obj  FunNamespaceRec ( Obj hdCall ) {
    char * usage = "Usage: NamespaceRec ( <rec> )";
    Obj hdRec;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    hdRec = EVAL(PTR_BAG(hdCall)[1]);
    if ( GET_TYPE_BAG(hdRec) != T_REC ) return Error(usage, 0, 0);
    return NamespaceRec(hdRec);
}

Obj FunIsNamespace ( Obj hdCall ) {
    char * usage = "Usage: IsNamespace ( <obj> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    return (GET_TYPE_BAG(hd)==T_NAMESPACE) ? HdTrue : HdFalse;
}

Obj  EvNS ( Obj hd ) { return hd; }

void PrNS ( Obj hd ) {
    int i, first = 1;
    if ( TableId(hd) != 0 )
        Pr("%g", (Int)TableId(hd), 0);
    else {
        Pr("%2>UnnamedNS(",0,0);
        for ( i = 0; i < TableSize(hd); i++ ) {
            if ( PTR_BAG(hd)[i] == 0 || VAR_VALUE(PTR_BAG(hd)[i]) == 0 )  continue;
            if ( ! first ) Pr(", ", 0, 0);
            first = 0;
            Pr("%g", (Int)PTR_BAG(hd)[i], 0);
        }
        Pr(")%2<",0,0);
    }
}

Obj FunNSId ( Obj hdCall ) {
    char * usage = "Usage: NSId ( <ns> )";
    Obj hd, hdId;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    hd = INJECTION_D(hd);
    if ( GET_TYPE_BAG(hd) != T_NAMESPACE ) return Error(usage, 0, 0);
    hdId = TableId(hd);
    if (hdId == 0) return HdFalse;
    else return hdId;
}

Obj PathNSSpec(Obj hd, char path_sep) {
    if(GET_TYPE_BAG(hd)==T_VAR || GET_TYPE_BAG(hd)==T_VARAUTO)
      return StringVar(hd);
    else if(GET_TYPE_BAG(hd)==T_RECELM) {
        Obj lhs_st, rhs;
        UInt rhs_len, lhs_len;
        lhs_st = PathNSSpec(PTR_BAG(hd)[0], path_sep);
        rhs = RecnameObj(PTR_BAG(hd)[1]);
        rhs_len = strlen(RECNAM_NAME(rhs));
        lhs_len = GET_SIZE_BAG(lhs_st) - 1;
        Resize(lhs_st, lhs_len + rhs_len + 2);
        CHARS_STRING(lhs_st)[lhs_len] = path_sep;
        strncat(CHARS_STRING(lhs_st) + lhs_len + 1, RECNAM_NAME(rhs), rhs_len);
        return lhs_st;
    }
    else return Error("pkg.subpkg... expected", 0, 0);
}

Obj FunPathNSSpec ( Obj hdCall ) {
    char * usage = "Usage: PathNSSpec ( <ns> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    hd = INJECTION_D(hd);
    return PathNSSpec(hd, CHARS_STRING(VAR_VALUE(FindIdent("PATH_SEP")))[0]);
}

Obj Fun_WriteVar ( Obj hdCall ) {
    char * usage = "Usage: _WriteVar ( <delayed-var> )";
    Obj hd, hdTab, hdVar;
    UInt pos;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error(usage, 0, 0);
    hd = EVAL(PTR_BAG(hdCall)[1]);
    if ( GET_TYPE_BAG(hd) != T_DELAY ) return Error(usage, 0, 0);
    hd = INJECTION_D(hd);
    if ( GET_TYPE_BAG(hd) != T_VAR && GET_TYPE_BAG(hd) != T_VARAUTO )
        return Error("Delayed expression is not a variable", 0, 0);
    hdTab = Input->package;
    if ( hdTab == 0 ) hdTab = HdIdenttab;

    pos = TableLookup(hdTab, VAR_NAME(hd), OFS_IDENT);
    hdVar = PTR_BAG(hdTab)[pos];
    if ( hdVar == 0 ) {
        hdVar = TableAddIdent(hdTab, pos, VAR_NAME(hd));
        SET_BAG(hdTab, pos,  hdVar );
    }
    return PROJECTION_D(hdVar);
}


Bag       FunNSFields (Bag hdCall)
{
    Bag           hdNS,  hdRes;
    UInt                i, listpos;

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD ) return Error("usage: NSFields( <ns> )",0,0);
    hdNS = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdNS) != T_NAMESPACE ) return Error("NSFields: <ns> must be a namespace",0,0);

    hdRes = NewList(TableNumEnt(hdNS));
    listpos = 1;
    for ( i = 0; i < TableSize(hdNS); i++ ) {
        Obj ent = PTR_BAG(hdNS)[i];
        if ( ent == 0 || VAR_VALUE(ent)==0 )  continue;
        else {
            Obj hdNam = NEW_STRING(strlen(VAR_NAME(ent)));
            strncpy(CHARS_STRING(hdNam), VAR_NAME(ent), strlen(VAR_NAME(ent))+1);
            SET_BAG(hdRes, listpos++,  hdNam );
        }
    }
    return hdRes;
}

void InitNamespaces (void) {
  Obj call, pkg;
    GlobalPackage2("spiral", "namespaces");

    InstIntFunc("IsNamespace",    FunIsNamespace);
    InstIntFunc("PushNamespace",    FunPushNamespace);
    InstIntFunc("PushNamespaceGlobal", FunPushNamespaceGlobal);
    InstIntFunc("PopNamespace",     FunPopNamespace);
    InstIntFunc("RecNamespace",     FunRecNamespace);
    InstIntFunc("NamespaceRec",     FunNamespaceRec);
    InstIntFunc("NamespaceAdd",     FunNamespaceAdd);
    InstIntFunc("PopAllNamespaces", FunPopAllNamespaces);

    InstIntFunc("CurrentFile", FunCurrentFile);
    InstIntFunc("CurrentFileStack", FunCurrentFileStack);
    InstIntFunc("CurrentDir", FunCurrentDir);

    InstIntFunc("StartPackage",     FunStartPackage);
    InstIntFunc("GlobalPackage",     FunGlobalPackage);
    InstIntFunc("EmptyPackage",     FunEmptyPackage);
    InstIntFunc("GetPackage",       FunGetPackage);
    InstIntFunc("EndPackage",       FunEndPackage);

    InstIntFunc("NSId",     FunNSId);
    InstIntFunc("PathNSSpec",     FunPathNSSpec);
    InstIntFunc("NSFields",     FunNSFields);
    InstIntFunc("_WriteVar",     Fun_WriteVar);

	SET_VAR_VALUE(FindIdentWr("Global"), HdIdenttab);
    SetTableId(HdIdenttab, FindIdent("Global"));

    HdLocal = FindIdentWr("Local");
    SET_FLAG_BAG(HdLocal, BF_VAR_AUTOEVAL);

    call = VAR_VALUE(FindIdent("CurrentFile"));
    call = UniBag(T_FUNCCALL, call);
    pkg = FindRecname("pkg");
    call = BinBag(T_RECELM, call, pkg);

    SET_VAR_VALUE(HdLocal, call);

    InstEvFunc(T_NAMESPACE, EvNS);
    InstPrFunc(T_NAMESPACE, PrNS);

    EndPackage();
}
/* -*- Mode: c; c-basic-offset: 4 -*- */
