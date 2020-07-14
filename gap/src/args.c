/****************************************************************************
**
*A  args.c                       SPIRAL source               Yevgen Voronenko
**
**
**  Defines functions useful for dealing with GAP data types, and evaluating
**  elements of interpreter function calls.
**
*/

#include	<string.h>

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "objects.h"
#include		"string4.h"
#include        "scanner.h"             /* Pr()                            */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* HD_TO_INT / INT_TO_HD           */
#include        "read.h"                /* ReadIt()                        */
#include        "gstring.h"
#include        "idents.h"
#include        "plist.h"
#include        "record.h"
#include        "tables.h"
#include        "args.h"
#include        "double.h"

char   HdToChar ( Obj hdVal, char * errMsg, Int a, Int b ) {
    Obj hd = EVAL(hdVal);
    if( GET_TYPE_BAG(hd) != T_CHAR ) Error(errMsg,a,b);
    return *(unsigned char*)PTR_BAG(hdVal);
}

char * HdToString ( Obj hdVal, char * errMsg, Int a, Int b ) {
    Obj hd = EVAL(hdVal);
    if( ! IsString(hd) ) Error(errMsg,a,b);
    return (char *) PTR_BAG(hd);
}

Int   HdToInt ( Obj hdVal, char * errMsg, Int a, Int b ) {
    Obj hd = EVAL(hdVal);
    if( GET_TYPE_BAG(hd) != T_INT ) Error(errMsg,a,b);
    return HD_TO_INT(hd);
}

double HdToDouble ( Obj hdVal, char * errMsg, Int a, Int b ) {
    Obj hd = EVAL(hdVal);
    if( GET_TYPE_BAG(hd) != T_DOUBLE ) Error(errMsg, a, b);
    return DBL_OBJ(hd);
}

Obj    StringToHd (char *st) { 
    Obj result;
    C_NEW_STRING(result, st);
    return result;
}

Obj    DoubleToHd (double d) {
    return ObjDbl(d);
}

Obj    CharToHd   (char c) {
    return HdChars[(int)c];
}

Obj    StringVar (Obj var) {
  UInt len = strlen(VAR_NAME(var));
  Obj st = NEW_STRING(len);
  strncpy(CHARS_STRING(st), VAR_NAME(var), len);
  return st;
}

Obj    NewList   (int length) {
    Obj result = NewBag(T_LIST,  SIZE_PLEN_PLIST(length));
    SET_LEN_PLIST(result, length);
    return result;
}

Obj    NewListT   (int type, int length) {
    Obj result = NewBag(type,  SIZE_PLEN_PLIST(length));
    SET_LEN_PLIST(result, length);
    return result;
}

Obj*   FindRecField  ( Obj hdRec, char *field ) {
    /* if the right operand is a record look for the 'operations' element  */
    Obj bag;
    if ( GET_TYPE_BAG(hdRec) != T_REC )
        Error("FindRecField: Record expected", 0, 0);    
    return FindRecnameRec(hdRec, FindRecname(field), &bag);
}

Obj    RecFieldValue ( Obj hdRec, char * field ){
    Obj * ptRec;
    if ( GET_TYPE_BAG(hdRec) != T_REC )	
	return Error("RecFieldValue: Record expected", 0, 0);
    ptRec = FindRecField(hdRec, field);
    if ( ptRec != NULL ) return ptRec[1];
    else return Error("RecFieldValue: '%s' field must have an assigned value", (Int)field, 0);
}

Obj    SetRecField ( Obj hdRec, char * field, Obj hdVal ) {
    return SetRecname( hdRec, FindRecname(field), hdVal );
}

Obj    SetNSField ( Obj hdRec, char * field, Obj hdVal ) {
    UInt pos = TableLookup(hdRec, field, OFS_IDENT);
    Obj hd = PTR_BAG(hdRec)[pos];
    if ( hd == 0 ) 
	hd = TableAddIdent(hdRec, pos, field);
    return SET_VAR_VALUE(hd, hdVal);
}
