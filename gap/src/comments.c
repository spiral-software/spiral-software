/****************************************************************************
**
*A  comments.c                  SPIRAL source                Yevgen Voronenko
**
*A  $Id: comments.c 9810 2010-08-16 20:40:53Z vlad $
**
**  This package implements special handling of comments.
*/
#include        <stdlib.h>
#include        <time.h>
#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* Bag, NewBag, T_STRING, .. */
#include        "plist.h"
#include        "idents.h"
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */
#include        "scanner.h"             /* Pr()                            */
#include        "objects.h"
#include		"string4.h"
#include        "comments.h"
#include        "spiral.h"
#include        "args.h"
#include        "record.h"
#include        "namespaces.h"
#include        "tables.h"
#include        "function.h"
#include		"GapUtils.h"

/*V  HdCommentBuffer */
/*V  HdDocRecname */
UInt SAVE_DEF_LINE = 1;


char DEFINED_IN_FMT[] = "[[ defined in %s:%d ]]\n";

Int ExtractDefinedIn( char* text, char* fileName, Int* line) {
    char *end, *start = strstr(text, "[[ defined in ");
    int count;
    if (start==0) return 0;
    end = strstr(start, " ]]\n");
    if (end==0) return 0;
    start += 14;
    while (--end>start && isdigit(*end));
    if (*end!=':') return 0;
    count = (int)(end - start);
    strncpy(fileName, start, count);
    fileName[count] = 0;
    *line = atoi(++end);
    return 1;
}
 
Obj  HdCommentBuffer;
Obj  HdDocRecname;

Obj FunDocumentVariable(Obj);

void InitCommentBuffer(void) {
    ClearCommentBuffer();
    InitGlobalBag(&HdCommentBuffer, "HdCommentBuffer");
    InitGlobalBag(&HdDocRecname, "HdDocRecname");
    HdDocRecname = FindRecname("__doc__");
    GlobalPackage2("spiral", "doc");
    InstIntFunc("CommentBuffer", FunCommentBuffer);
    InstIntFunc("ClearCommentBuffer", FunClearCommentBuffer);
    InstIntFunc("Doc", FunDoc);
    InstIntFunc("DocLoc", FunDocLoc);
    InstIntFunc("DocumentVariable", FunDocumentVariable);
    InstIntFunc("SaveDefLine", FunSaveDefLine);
    EndPackage();
}

static Bag NoCopy(Bag obj) {
    SET_FLAG_BAG(obj, BF_NO_COPY);
    return obj;
}

void ClearCommentBuffer(void) {
    HdCommentBuffer = NoCopy(NewBag(T_STRING, 4));
    CSTR_STRING(HdCommentBuffer)[0] = '\0';
}

void AppendCommentBuffer(char *text, int tlen) {
    int buflen = strlen(CSTR_STRING(HdCommentBuffer));
    int bagsize = GET_SIZE_BAG(HdCommentBuffer);
    if (tlen >=3 && text[1]=='@' && text[2]=='P') {
        if ( Input->package != 0 )
            DocumentValue(Input->package, HdCommentBuffer);
        ClearCommentBuffer();
    }
    else {
        if(tlen + buflen + 1 >= bagsize) {
            /* allocate more then we need for efficiency */
            bagsize = 2*(tlen + buflen);
            Resize(HdCommentBuffer, bagsize);
        }
        if ( tlen >= 2 && text[0] == '#' && text[1] == ' ') { tlen-=2; text+=2; }

	strncpy(CSTR_STRING(HdCommentBuffer) + buflen, text, tlen);
        CSTR_STRING(HdCommentBuffer)[tlen+buflen+1] = '\0';
        /*printf("append: %d (%d/%d)\n", tlen, buflen, bagsize);*/
    }
}

Obj  GetCommentBuffer(void) {
    return HdCommentBuffer;
}

void DocumentValue( Obj val, Obj comment ) {
    if ( comment == 0 ) return;
    if ( GET_TYPE_BAG(val) == T_REC ) {
        SetRecname(val, HdDocRecname, NoCopy(comment));
    }
    else if ( GET_TYPE_BAG(val) == T_NAMESPACE ) {
        SetNSField(val, "__doc__", NoCopy(comment));
    }
}

void DocumentVariable( Obj var, Obj comment ) {
    if( comment == 0) return;
    SetRecname(Props(var), HdDocRecname, NoCopy(comment));
}

Obj FunDocumentVariable(Obj hdCall) {
    Obj hdVar;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD ) return Error("Usage: DocumentVariable(<variable>)", 0, 0);
    hdVar = PTR_BAG(hdCall)[1];
    if ( GET_TYPE_BAG(hdVar) != T_VAR && GET_TYPE_BAG(hdVar) != T_VARAUTO )
        return Error("Usage: ProtectVar(<variable>)", 0, 0);
    DocumentVariable(hdVar, GetCommentBuffer());
    return HdVoid;
}


Obj  FunCommentBuffer( Obj hdCall ) {
    return HdCommentBuffer;
}

Obj  FunClearCommentBuffer( Obj hdCall ) {
    ClearCommentBuffer();
    return HdVoid;
}

Obj  MakeDefString() {
    Obj result = 0;
    if (SAVE_DEF_LINE) {
		char *str = GuMakeMessage(DEFINED_IN_FMT, Input->name, Input->number);
		C_NEW_STRING(result, str);
		NoCopy(result);
		free(str);
    }
    return result;
}

Obj  FunSaveDefLine( Obj hdCall ) {
    char * usage = "usage: SaveDefLine( <true> | <false> ) -- toggle automatic linenumber saving \n"
                   "       docstrings will not contain definition linenumber.\n"
                   "usage: SaveDefLine( ) -- appends current linunumber to current docstring\n";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) > 2 * SIZE_HD )    return Error(usage, 0,0);
    else if ( GET_SIZE_BAG(hdCall) == SIZE_HD ) {
        char *str = GuMakeMessage(DEFINED_IN_FMT, Input->name, Input->number);
        AppendCommentBuffer(str, (int)strlen(str));
        free(str);
    }
    else {
        hd = EVAL(PTR_BAG(hdCall)[1]);
        if ( GET_TYPE_BAG(hd) != T_BOOL )    return Error(usage, 0,0);
        if ( hd == HdTrue ) SAVE_DEF_LINE = 1;
        else SAVE_DEF_LINE = 0;
    }
    return HdVoid;
}

/***********************************************************************
**
*F FindDocString( Obj hd ) . . . . . finds documentation for given object
**
** Returns 0 if not found. Documentation expected to be in T_STRING object.
**
*/

Obj  FindDocString( Obj hd ) {
    
    Obj	doc = NULL, temp;
    Obj *docfld;
    Int type = GET_TYPE_BAG(hd);
    
    if(type == T_VAR || type == T_VARAUTO) {
        if(type == T_VARAUTO)
            EVAL(hd);
        docfld = FindRecnameRec(Props(hd), HdDocRecname, &temp);
        if(docfld!=NULL) doc = docfld[1];
        hd = VAR_VALUE(hd);
    }
    else
	hd = EVAL(hd);

    type = GET_TYPE_BAG(hd);
    if ( doc == NULL && ( type == T_FUNCTION 
	    || type == T_METHOD || type == T_MAKEFUNC 
	    || type == T_MAKEMETH)) 
    {
	int numArg, numLoc;
	ACT_NUM_ARGS_FUNC(hd, numArg);
	ACT_NUM_LOCALS_FUNC(hd, numLoc);
	doc = PTR_BAG(hd)[numArg + numLoc + 2];
    }
    
    /* try to look in __doc__ */
    if ( doc == NULL && (hd!=0 && (GET_TYPE_BAG(hd)==T_REC || GET_TYPE_BAG(hd)==T_NAMESPACE) )) {
        if ( GET_TYPE_BAG(hd) == T_REC ) {
            docfld = FindRecnameRec(hd, HdDocRecname, &temp);
            if(docfld!=NULL) doc = docfld[1];
        }
        else {
            Int n = TableLookup(hd, "__doc__", OFS_IDENT);
            if ( PTR_BAG(hd)[n] != 0 )
                doc = VAR_VALUE(PTR_BAG(hd)[n]);
        }
    }
    return doc;
}

Int FindDocAndExtractLoc(Bag obj, char* fileName, Int* line) {
    Obj doc = FindDocString(obj);
    if(doc != 0 && GET_TYPE_BAG(doc)==T_STRING) {
        if (ExtractDefinedIn(CSTR_STRING(doc), fileName, line)) {
            return 1;
        } else
            return -1;
    } else
        return 0;
}

/****************************************************************************
**
*F  Doc( <func> ) . . . . . . . . . .  print documentation for given function
**
**  FunDoc implements internal function 'Doc'.
*/

Obj  FunDoc( Obj hdCall ) {
    char * usage = "usage: Doc( <var> )";
    Obj doc;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )    return Error(usage, 0,0);

    doc = FindDocString(PTR_BAG(hdCall)[1]);

    if(doc != NULL && GET_TYPE_BAG(doc)==T_STRING)
        Pr("%s", (Int)CSTR_STRING(doc), 0);
    else if(doc == NULL)
        Pr("--no documentation--\n", 0, 0);
    else
        Pr("--documentation corrupt (not a string, but %s)--\n", (Int)TNAM_BAG(doc), 0);

    return HdVoid;
}

Bag FunDocLoc(Bag hdCall) {
    char * usage = "usage: DocLoc( <obj> )";
    Obj hd;
    Obj doc; 

    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )    return Error(usage, 0,0);

    hd  = NewList(2);
    doc = FindDocString(PTR_BAG(hdCall)[1]);
    if(doc != 0 && GET_TYPE_BAG(doc)==T_STRING) {
        char fileName[1024];
        Int  line;
        if (ExtractDefinedIn(CSTR_STRING(doc), fileName, &line)) {
	    Obj s = NoCopy(NEW_STRING(strlen(fileName)));
	    strcpy(CSTR_STRING(s), fileName);
	    SET_ELM_PLIST(hd, 1, s);
	    SET_ELM_PLIST(hd, 2, INT_TO_HD(line));
            return hd;
        }
    }
    SET_ELM_PLIST(hd, 1, NEW_STRING(0));
    SET_ELM_PLIST(hd, 2, INT_TO_HD(0));
    return hd;
}


/* -*- Mode: c ; c-basic-offset:4 -*- */
