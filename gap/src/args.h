/****************************************************************************
**
*A  args.h                       SPIRAL source               Yevgen Voronenko
**
**
**  Defines functions useful for dealing with GAP data types, and evaluating
**  elements of interpreter function calls.
**
*/

/* Get the function argument from T_FUNCCALL bag */
#define ELM_ARGLIST(hdCall, i) PTR_BAG(hdCall)[i]

/* Get the function pointer from T_FUNCINT bag */
#define HD_TO_FUNCINT(hd) (*  (Obj(**)(Obj)) PTR_BAG(hd))
#define HD_TO_STRING(hd) ((char*)PTR_BAG(hd))

char   HdToChar   ( Obj val, char * errMsg, Int, Int );
char * HdToString ( Obj val, char * errMsg, Int, Int );
Int   HdToInt    ( Obj val, char * errMsg, Int, Int );
double HdToDouble ( Obj val, char * errMsg, Int, Int );

#define IntToHd INT_TO_HD
Obj    StringToHd (char *st);
Obj    DoubleToHd (double d);
Obj    CharToHd   (char c);

Obj    StringVar (Obj var);
Obj    NewList   (int length);
Obj    NewListT   (int type, int length);

Obj*   FindRecField  ( Obj hdRec, char *field );
Obj    RecFieldValue ( Obj hdRec, char *field );
Obj    SetRecField   ( Obj hdRec, char *field, Obj hdVal );

Obj    SetNSField ( Obj hdNS, char * field, Obj hdVal );

#define INT_FLD(hdRec, fld)  HdToInt(RecFieldValue(hdRec, fld), \
			"Record '%s' field must be an integer",(Int)fld,NUM_TO_INT(0))

#define STR_FLD(hdRec, fld)  HdToString(RecFieldValue(hdRec, fld), \
			"Record '%s' field must be a string",(Int)fld,NUM_TO_INT(0))

#define CHAR_FLD(hdRec, fld) HdToChar(RecFieldValue(hdRec, fld), \
			"Record '%s' field must be a character",(Int)fld,NUM_TO_INT(0))

#define DBL_FLD(hdRec, fld) HdToDouble(RecFieldValue(hdRec, fld), \
			"Record '%s' field must be an IntPair (use IntPairFloat)",(Int)fld,NUM_TO_INT(0))
